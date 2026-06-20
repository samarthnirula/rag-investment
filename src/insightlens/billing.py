"""Plan limits and gross-margin guardrails for Atticus.

The public prices can stay simple while the operating limits protect the
business.  The default rule is a 60% gross margin, which means variable AI and
storage costs should stay under 40% of monthly revenue.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


TARGET_GROSS_MARGIN = float(os.getenv("ATTICUS_TARGET_GROSS_MARGIN", "0.60"))


@dataclass(frozen=True)
class PlanLimits:
    name: str
    monthly_price_usd: float
    monthly_query_limit: int
    monthly_upload_limit: int
    max_upload_mb: int
    max_pages_per_pdf: int

    @property
    def monthly_variable_cost_cap_usd(self) -> float:
        return round(self.monthly_price_usd * (1 - TARGET_GROSS_MARGIN), 2)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 0)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, 0.0)


def default_plan() -> PlanLimits:
    """Return the default plan used until Stripe/subscription data exists."""
    return PlanLimits(
        name=os.getenv("ATTICUS_DEFAULT_PLAN", "Starter"),
        monthly_price_usd=_env_float("ATTICUS_STARTER_PRICE_USD", 29.0),
        monthly_query_limit=_env_int("ATTICUS_MONTHLY_QUERY_LIMIT", 300),
        monthly_upload_limit=_env_int("ATTICUS_MONTHLY_UPLOAD_LIMIT", 20),
        max_upload_mb=_env_int("ATTICUS_MAX_UPLOAD_MB", 50),
        max_pages_per_pdf=_env_int("ATTICUS_MAX_PAGES_PER_PDF", 500),
    )


def max_upload_bytes(plan: PlanLimits | None = None) -> int:
    active_plan = plan or default_plan()
    return active_plan.max_upload_mb * 1024 * 1024


def estimate_query_cost_usd(
    *,
    query_text: str,
    response_text: str,
    chunks_retrieved: int,
) -> float:
    """Cheap, conservative telemetry estimate for margin dashboards.

    This is not provider billing-grade accounting.  It is intentionally simple:
    approximate tokens from characters and attach a small retrieval overhead.
    The value helps flag heavy users before exact provider-level cost tracking
    is wired in.
    """
    estimated_tokens = max(1, (len(query_text) + len(response_text)) // 4)
    generation_cost = estimated_tokens * _env_float("ATTICUS_EST_COST_PER_TOKEN", 0.000003)
    retrieval_cost = chunks_retrieved * _env_float("ATTICUS_EST_COST_PER_SOURCE", 0.00001)
    return round(generation_cost + retrieval_cost, 6)


def estimate_ingestion_cost_usd(*, pages: int, file_size_bytes: int) -> float:
    """Estimate variable ingestion cost for upload guardrails and alerts."""
    per_page = _env_float("ATTICUS_EST_INGEST_COST_PER_PAGE", 0.002)
    per_mb = _env_float("ATTICUS_EST_STORAGE_COST_PER_MB_MONTH", 0.0005)
    return round((pages * per_page) + ((file_size_bytes / (1024 * 1024)) * per_mb), 6)


def format_limit_summary(plan: PlanLimits | None = None) -> str:
    active_plan = plan or default_plan()
    return (
        f"{active_plan.monthly_query_limit} AI queries/month, "
        f"{active_plan.monthly_upload_limit} uploads/month, "
        f"{active_plan.max_upload_mb} MB/file, "
        f"{active_plan.max_pages_per_pdf} pages/PDF"
    )
