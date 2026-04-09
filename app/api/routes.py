from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.graph.nodes import GraphRuntime

router = APIRouter()


def get_runtime(request: Request) -> GraphRuntime:
    return request.app.state.runtime


@router.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    runtime = get_runtime(request)
    state = runtime.build_initial_state(user_id=payload.user_id, message=payload.message)
    result = await runtime.run(state)
    return ChatResponse(answer=result.get("final_answer", ""))


@router.post("/stream")
async def stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    runtime = get_runtime(request)
    queue: asyncio.Queue[str] = asyncio.Queue()
    initial_state = runtime.build_initial_state(
        user_id=payload.user_id,
        message=payload.message,
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
                    continue
            result = await task
            answer = result.get("final_answer", "")
            for token in answer.split():
                yield f"event: token\ndata: {json.dumps({'token': token + ' '}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'answer': answer}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
