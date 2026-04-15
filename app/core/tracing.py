from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any

from app.db.models import TraceRecord
from app.db.session import async_session_maker

logger = logging.getLogger(__name__)


def _safe_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {"value": value}


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:4000]


@dataclass(slots=True)
class TraceHandle:
    step_name: str
    user_id: str
    request_id: str
    input: Any
    output: Any = None
    estimated_tokens: int | None = None
    started_at: float = field(default_factory=time.perf_counter)
    # F8 — queryable top-level fields
    route_decision: str | None = None
    model_used: str | None = None
    rejection_reason: str | None = None
    input_hash: str | None = None
    tool_names: list[str] | None = None


class trace(AbstractAsyncContextManager[TraceHandle]):
    def __init__(self, *, step_name: str, user_id: str, request_id: str, input: Any) -> None:
        self._handle = TraceHandle(
            step_name=step_name,
            user_id=user_id,
            request_id=request_id,
            input=input,
        )

    async def __aenter__(self) -> TraceHandle:
        return self._handle

    async def __aexit__(self, exc_type, exc, tb) -> None:
        latency_ms = int((time.perf_counter() - self._handle.started_at) * 1000)
        if exc is not None:
            self._handle.output = {"error": str(exc)}
        asyncio.create_task(_persist_trace(self._handle, latency_ms))


async def _persist_trace(handle: TraceHandle, latency_ms: int) -> None:
    try:
        async with async_session_maker() as session:
            record = TraceRecord(
                user_id=handle.user_id,
                request_id=handle.request_id,
                step_name=handle.step_name,
                input_text=_safe_text(handle.input),
                output_text=_safe_text(handle.output),
                input_payload=_safe_payload(handle.input),
                output_payload=_safe_payload(handle.output),
                latency_ms=latency_ms,
                estimated_tokens=handle.estimated_tokens,
                route_decision=handle.route_decision,
                model_used=handle.model_used,
                rejection_reason=handle.rejection_reason,
                input_hash=handle.input_hash,
                tool_names=handle.tool_names,
            )
            session.add(record)
            await session.commit()
    except Exception:
        logger.exception("Failed to persist trace for step=%s", handle.step_name)
