from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader

from app.api.schemas import ChatRequest, ChatResponse
from app.core.config import Settings
from app.graph.nodes import GraphRuntime

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key")

def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings

async def get_api_key(api_key: str = Depends(api_key_header), settings: Settings = Depends(get_settings_dep)) -> str:
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def get_runtime(request: Request) -> GraphRuntime:
    return request.app.state.runtime


@router.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request, api_key: str = Depends(get_api_key)) -> ChatResponse:
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    runtime = get_runtime(request)
    state = runtime.build_initial_state(
        user_id=payload.user_id,
        message=payload.message,
        request_id=request_id,
    )
    result = await runtime.run(state)
    return ChatResponse(answer=result.get("final_answer", ""))


@router.post("/stream")
async def stream(payload: ChatRequest, request: Request, api_key: str = Depends(get_api_key)) -> StreamingResponse:
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
            for token in answer.split():
                yield f"event: token\ndata: {json.dumps({'token': token + ' '}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'answer': answer}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("Stream error", exc_info=exc)
            yield f"event: error\ndata: {json.dumps({'message': 'Internal server error'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
