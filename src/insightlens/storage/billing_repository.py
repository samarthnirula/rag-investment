"""Subscription and admin margin reporting."""
from __future__ import annotations

from dataclasses import dataclass

import psycopg2

from insightlens.storage.snowflake_client import SnowflakeConnection


@dataclass(frozen=True)
class UserMarginRow:
    user_id: str
    queries_this_month: int
    uploads_this_month: int
    estimated_cost_usd: float


class BillingRepository:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self._conn = conn

    def upsert_subscription(
        self,
        *,
        user_id: str,
        plan_name: str,
        status: str = "trialing",
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
    ) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """INSERT INTO subscriptions
                   (user_id, plan_name, status, stripe_customer_id, stripe_subscription_id)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (user_id) DO UPDATE SET
                     plan_name = EXCLUDED.plan_name,
                     status = EXCLUDED.status,
                     stripe_customer_id = EXCLUDED.stripe_customer_id,
                     stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                     updated_at = NOW()""",
                (user_id, plan_name, status, stripe_customer_id, stripe_subscription_id),
            )
        except psycopg2.Error:
            pass
        finally:
            cur.close()

    def admin_margin_rows(self, limit: int = 100) -> list[UserMarginRow]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """WITH q AS (
                       SELECT user_id,
                              COUNT(*) FILTER (
                                WHERE DATE_TRUNC('month', logged_at) = DATE_TRUNC('month', CURRENT_DATE)
                              ) AS queries_this_month,
                              COALESCE(SUM(estimated_cost_usd) FILTER (
                                WHERE DATE_TRUNC('month', logged_at) = DATE_TRUNC('month', CURRENT_DATE)
                              ), 0) AS query_cost
                       FROM query_log
                       WHERE user_id IS NOT NULL
                       GROUP BY user_id
                   ),
                   u AS (
                       SELECT user_id,
                              COUNT(*) AS uploads_this_month,
                              COALESCE(SUM(estimated_cost_usd), 0) AS upload_cost
                       FROM upload_events
                       WHERE DATE_TRUNC('month', uploaded_at) = DATE_TRUNC('month', CURRENT_DATE)
                       GROUP BY user_id
                   )
                   SELECT COALESCE(q.user_id, u.user_id) AS user_id,
                          COALESCE(q.queries_this_month, 0) AS queries_this_month,
                          COALESCE(u.uploads_this_month, 0) AS uploads_this_month,
                          COALESCE(q.query_cost, 0) + COALESCE(u.upload_cost, 0) AS estimated_cost
                   FROM q FULL OUTER JOIN u ON q.user_id = u.user_id
                   ORDER BY estimated_cost DESC
                   LIMIT %s""",
                (limit,),
            )
            return [
                UserMarginRow(
                    user_id=row[0],
                    queries_this_month=int(row[1] or 0),
                    uploads_this_month=int(row[2] or 0),
                    estimated_cost_usd=float(row[3] or 0),
                )
                for row in cur.fetchall()
            ]
        except psycopg2.Error:
            return []
        finally:
            cur.close()
