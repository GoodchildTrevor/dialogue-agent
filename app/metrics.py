"""
Prometheus metrics collector for the dialogue-agent tracing table.

Exposes metrics derived from the `traces` table:
  - agent_node_latency_ms   → Histogram: latency per step_name + model_used
  - agent_route_total       → Counter: route_decision distribution
  - agent_tool_error_total  → Counter: tool errors (rows with rejection_reason set)
  - agent_model_calls_total → Counter: calls per model_used
"""
from __future__ import annotations
import logging
from typing import Sequence
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import select

from app.db.models import TraceRecord
from app.db.session import get_session_maker

logger = logging.getLogger(__name__)

# Isolated registry — avoids collision with default process metrics
REGISTRY = CollectorRegistry(auto_describe=False)

node_latency = Histogram(
    "agent_node_latency_ms",
    "Latency in milliseconds per graph node (step_name)",
    labelnames=["step_name", "model_used"],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
    registry=REGISTRY,
)

route_counter = Counter(
    "agent_route_total",
    "Number of requests per route decision",
    labelnames=["route_decision"],
    registry=REGISTRY,
)

tool_error_counter = Counter(
    "agent_tool_error_total",
    "Number of tool errors per step_name and rejection_reason",
    labelnames=["step_name", "rejection_reason"],
    registry=REGISTRY,
)

model_calls_counter = Counter(
    "agent_model_calls_total",
    "Number of LLM calls per model",
    labelnames=["model_used"],
    registry=REGISTRY,
)

# High-watermark: track unix timestamp of last processed row
_last_scraped_at: float = 0.0


async def init_watermark() -> None:
    """Seed the watermark from the most recent trace row so we don't replay history on restart.
    
    :param None: This function takes no parameters
    :returns: None
    :raises: May raise exceptions related to database connectivity or SQLAlchemy operations
    """
    global _last_scraped_at
    from sqlalchemy import select, func
    from datetime import datetime, timezone
    
    session_maker = get_session_maker()
    async with session_maker() as session:
        result = await session.execute(
            select(func.max(TraceRecord.created_at))
        )
        latest = result.scalar_one_or_none()
        if latest is not None:
            _last_scraped_at = latest.timestamp()
        else:
            # If no records exist yet, set to current time to avoid processing old records
            _last_scraped_at = datetime.now(timezone.utc).timestamp()


async def collect_metrics() -> None:
    """Pull new TraceRecord rows since last scrape and update all metrics.
    
    :param None: This function takes no parameters
    :returns: None
    :raises: May raise exceptions related to database connectivity or SQLAlchemy operations
    """
    global _last_scraped_at

    session_maker = get_session_maker()
    async with session_maker() as session:
        stmt = (
            select(TraceRecord)
            .where(
                TraceRecord.created_at > datetime.fromtimestamp(_last_scraped_at, tz=timezone.utc)
            )
            .order_by(TraceRecord.created_at.asc())
            .limit(1000)   # prevent full table scan on first scrape
        )
        result = await session.execute(stmt)
        rows: Sequence[TraceRecord] = result.scalars().all()

    if not rows:
        return

    for row in rows:
        step  = row.step_name      or "unknown"
        model = row.model_used     or "unknown"
        route = row.route_decision or "unknown"

        node_latency.labels(step_name=step, model_used=model).observe(row.latency_ms)

        if row.route_decision:
            route_counter.labels(route_decision=route).inc()

        if row.rejection_reason:
            tool_error_counter.labels(step_name=step, rejection_reason=row.rejection_reason).inc()

        if row.model_used:
            model_calls_counter.labels(model_used=model).inc()

    _last_scraped_at = rows[-1].created_at.timestamp()
    logger.debug("metrics: processed %d new trace rows", len(rows))


def get_metrics_output() -> tuple[bytes, str]:
    """Generate the latest metrics data and return it with the appropriate content type.
    
    :param None: This function takes no parameters
    :returns: A tuple containing the metrics data as bytes and the content type string
    :rtype: tuple[bytes, str]
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
