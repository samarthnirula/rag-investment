from insightlens.billing import (
    TARGET_GROSS_MARGIN,
    default_plan,
    estimate_ingestion_cost_usd,
    estimate_query_cost_usd,
    max_upload_bytes,
)


def test_default_plan_enforces_margin_cost_cap():
    plan = default_plan()
    assert plan.monthly_price_usd == 29.0
    assert plan.monthly_variable_cost_cap_usd == round(
        plan.monthly_price_usd * (1 - TARGET_GROSS_MARGIN),
        2,
    )


def test_upload_bytes_uses_plan_limit():
    plan = default_plan()
    assert max_upload_bytes(plan) == plan.max_upload_mb * 1024 * 1024


def test_cost_estimators_return_positive_values():
    query_cost = estimate_query_cost_usd(
        query_text="Summarize this agreement.",
        response_text="This agreement has a termination clause.",
        chunks_retrieved=4,
    )
    ingest_cost = estimate_ingestion_cost_usd(pages=10, file_size_bytes=1024 * 1024)

    assert query_cost > 0
    assert ingest_cost > 0
