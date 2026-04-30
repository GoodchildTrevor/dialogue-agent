from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader

from app.api.schemas import ChatRequest, ChatResponse
from app.core.config import Settings
from app.graph.graph_runtime import GraphRuntime

router = APIRouter()
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key")


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


async def get_api_key(
    api_key: str = Depends(api_key_header),
    settings: Settings = Depends(get_settings_dep),
) -> str:
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def get_runtime(request: Request) -> GraphRuntime:
    return request.app.state.runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _save_and_embed(
    runtime: GraphRuntime,
    *,
    user_id: str,
    user_content: str,
    assistant_content: str,
) -> None:
    """Save user + assistant messages to PG and embed them asynchronously.

    Runs as a fire-and-forget task — never raises to the caller.
    """
    try:
        user_msg_id = await runtime.history_service.save_message(
            user_id=user_id, role="user", content=user_content
        )
        asst_msg_id = await runtime.history_service.save_message(
            user_id=user_id, role="assistant", content=assistant_content
        )
        await runtime.pg_ingester.ingest([user_msg_id, asst_msg_id])
    except Exception as exc:
        logger.warning("save_and_embed failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> ChatResponse:
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    runtime = get_runtime(request)
    state = runtime.build_initial_state(
        user_id=payload.user_id,
        message=payload.message,
        request_id=request_id,
    )
    result = await runtime.run(state)
    answer = result.get("final_answer", "")

    asyncio.create_task(
        _save_and_embed(
            runtime,
            user_id=payload.user_id,
            user_content=payload.message,
            assistant_content=answer,
        )
    )

    return ChatResponse(answer=answer)


@router.post("/stream")
async def stream(
    payload: ChatRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> StreamingResponse:
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    runtime = get_runtime(request)
    queue: asyncio.Queue[str] = asyncio.Queue()
    initial_state = runtime.build_initial_state(
        user_id=payload.user_id,
        message=payload.message,
        request_id=request_id,
        status_queue=queue,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(runtime.run(initial_state))
        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    status = await asyncio.wait_for(queue.get(), timeout=0.2)
                    yield f"event: status\ndata: {json.dumps({'message': status}, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    if task.done():
                        break
                    continue

            result = await task
            answer = result.get("final_answer", "")

            asyncio.create_task(
                _save_and_embed(
                    runtime,
                    user_id=payload.user_id,
                    user_content=payload.message,
                    assistant_content=answer,
                )
            )

            for token in answer.split():
                yield f"event: token\ndata: {json.dumps({'token': token + ' '}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'answer': answer}, ensure_ascii=False)}\n\n"

        except Exception as exc:
            logger.error("Stream error", exc_info=exc)
            yield f"event: error\ndata: {json.dumps({'message': 'Internal server error'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
