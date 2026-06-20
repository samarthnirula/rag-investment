"""Stripe webhook receiver — runs as a standalone FastAPI service.

Handles Stripe events and syncs subscription state to the PostgreSQL
subscriptions table so the app can enforce plan limits.

Supported events:
  checkout.session.completed          → create/activate subscription
  customer.subscription.created       → upsert subscription
  customer.subscription.updated       → upsert subscription
  customer.subscription.deleted       → mark cancelled
  invoice.payment_succeeded           → mark active
  invoice.payment_failed              → mark past_due

Run (standalone):
  uvicorn insightlens.webhook.stripe_handler:app --host 0.0.0.0 --port 8502

Environment variables required:
  DATABASE_URL
  STRIPE_WEBHOOK_SECRET   — from Stripe Dashboard → Webhooks → signing secret
  STRIPE_SECRET_KEY       — Stripe secret key (sk_live_... or sk_test_...)
"""
from __future__ import annotations

import logging
import os

import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

load_dotenv()

_log = logging.getLogger(__name__)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

app = FastAPI(title="Atticus Stripe Webhook", docs_url=None, redoc_url=None)


def _get_billing_repo():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from insightlens.config import load_config
    from insightlens.storage.billing_repository import BillingRepository
    from insightlens.storage.snowflake_client import open_connection
    cfg = load_config()
    return cfg, BillingRepository, open_connection


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="stripe-signature"),
):
    payload = await request.body()

    if not _WEBHOOK_SECRET:
        # SECURITY: previously this fell back to trusting the unsigned request
        # body (anyone who finds this URL could POST a fake
        # "checkout.session.completed" event and grant themselves a paid
        # plan). Fail closed instead -- misconfiguration should break
        # billing sync loudly, not silently accept forged events.
        _log.error("STRIPE_WEBHOOK_SECRET not set — refusing to process unsigned webhook")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook signing secret not configured.",
        )

    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, _WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature")

    event_type: str = event["type"]
    data = event["data"]["object"]

    _log.info("Stripe event: %s", event_type)

    try:
        cfg, BillingRepo, open_conn = _get_billing_repo()
        with open_conn(cfg.db) as conn:
            repo = BillingRepo(conn)
            _dispatch(event_type, data, repo)
    except Exception as exc:
        _log.exception("Error processing Stripe event %s: %s", event_type, exc)
        # Return 200 so Stripe doesn't retry indefinitely for non-transient errors
        return JSONResponse({"error": str(exc)}, status_code=200)

    return {"received": True}


def _dispatch(event_type: str, data: dict, repo) -> None:
    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data, repo)
    elif event_type in ("customer.subscription.created", "customer.subscription.updated"):
        _handle_subscription_upsert(data, repo)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(data, repo)
    elif event_type == "invoice.payment_succeeded":
        _handle_payment_succeeded(data, repo)
    elif event_type == "invoice.payment_failed":
        _handle_payment_failed(data, repo)
    else:
        _log.debug("Unhandled Stripe event type: %s", event_type)


def _handle_checkout_completed(session: dict, repo) -> None:
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")
    client_ref = session.get("client_reference_id")  # our user_id, set on checkout creation
    mode = session.get("mode")

    if mode != "subscription" or not client_ref:
        return

    repo.upsert_subscription(
        user_id=client_ref,
        plan_name=_resolve_plan_name(subscription_id),
        status="active",
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
    )
    _log.info("Subscription activated for user %s (customer %s)", client_ref, customer_id)


def _handle_subscription_upsert(subscription: dict, repo) -> None:
    customer_id = subscription.get("customer")
    subscription_id = subscription.get("id")
    sub_status = subscription.get("status", "active")
    plan_name = _plan_from_subscription(subscription)
    metadata = subscription.get("metadata", {})
    user_id = metadata.get("user_id")

    if not user_id:
        _log.debug("Subscription %s has no user_id in metadata — skipping", subscription_id)
        return

    repo.upsert_subscription(
        user_id=user_id,
        plan_name=plan_name,
        status=sub_status,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
    )


def _handle_subscription_deleted(subscription: dict, repo) -> None:
    metadata = subscription.get("metadata", {})
    user_id = metadata.get("user_id")
    if not user_id:
        return
    repo.upsert_subscription(
        user_id=user_id,
        plan_name="Starter",
        status="cancelled",
        stripe_customer_id=subscription.get("customer"),
        stripe_subscription_id=subscription.get("id"),
    )
    _log.info("Subscription cancelled for user %s", user_id)


def _handle_payment_succeeded(invoice: dict, repo) -> None:
    subscription_id = invoice.get("subscription")
    if not subscription_id:
        return
    try:
        sub = stripe.Subscription.retrieve(subscription_id)
        _handle_subscription_upsert(sub, repo)
    except Exception as exc:
        _log.warning("Could not retrieve subscription %s: %s", subscription_id, exc)


def _handle_payment_failed(invoice: dict, repo) -> None:
    subscription_id = invoice.get("subscription")
    metadata = invoice.get("subscription_details", {}).get("metadata", {})
    user_id = metadata.get("user_id")
    if not user_id:
        return
    repo.upsert_subscription(
        user_id=user_id,
        plan_name="Starter",
        status="past_due",
        stripe_customer_id=invoice.get("customer"),
        stripe_subscription_id=subscription_id,
    )


def _plan_from_subscription(subscription: dict) -> str:
    items = subscription.get("items", {}).get("data", [])
    if items:
        nickname = items[0].get("price", {}).get("nickname") or ""
        if nickname:
            return nickname
    return "Pro"


def _resolve_plan_name(subscription_id: str | None) -> str:
    if not subscription_id:
        return "Pro"
    try:
        sub = stripe.Subscription.retrieve(subscription_id)
        return _plan_from_subscription(sub)
    except Exception:
        return "Pro"
