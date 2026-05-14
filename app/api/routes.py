from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Request, Depends, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from sqlalchemy import select

from app.api.schemas import ChatRequest, ChatResponse
from app.core.config import Settings
from app.db.models import File as FileModel
from app.db.session import get_session_maker
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


# Headers that prevent nginx / any intermediary proxy from buffering the stream.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def _sse(payload: dict) -> str:
    """Encode *payload* as a plain SSE data frame (no event name)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_MAX_FILENAME_LEN = 200


def _sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > _MAX_FILENAME_LEN:
        name = name[:_MAX_FILENAME_LEN]
    return name or "file"


def _uploaded_files_to_dicts(uploaded_files) -> list[dict[str, str]]:
    """Convert list of UploadedFile pydantic models to plain dicts for state."""
    return [{"file_id": f.file_id, "filename": f.filename} for f in uploaded_files]


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
        uploaded_files=_uploaded_files_to_dicts(payload.uploaded_files),
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
        uploaded_files=_uploaded_files_to_dicts(payload.uploaded_files),
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        task = asyncio.create_task(runtime.run(initial_state))
        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    status = await asyncio.wait_for(queue.get(), timeout=0.2)
                    yield _sse({"status": status})
                except asyncio.TimeoutError:
                    continue

            while not queue.empty():
                status = queue.get_nowait()
                yield f"event: status\ndata: {json.dumps({'message': status}, ensure_ascii=False)}\n\n"

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

            yield _sse({"answer": answer})
            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.error("Stream error [%s]", request_id, exc_info=exc)
            yield _sse({"error": "Internal server error"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/upload")
async def upload_files(
    request: Request,
    user_id: str = Form(...),
    files: list[UploadFile] = File(...),
    api_key: str = Depends(get_api_key),
) -> JSONResponse:
    settings = request.app.state.settings
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for upload in files:
        if upload.content_type not in settings.ALLOWED_MIME_TYPES:
            raise HTTPException(415, f"Unsupported type: {upload.content_type}")

    results = []
    for upload in files:
        file_id = uuid4()
        safe_name = _sanitize_filename(upload.filename or "upload")
        path = Path(settings.UPLOAD_STORAGE_DIR) / user_id / f"{file_id}_{safe_name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        size_bytes = 0

        async with aiofiles.open(path, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    await f.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, f"{upload.filename} exceeds size limit")
                await f.write(chunk)

        async with get_session_maker()() as session:
            db_file = FileModel(
                id=file_id,
                user_id=user_id,
                original_name=upload.filename,
                mime_type=upload.content_type,
                storage_path=str(path),
                size_bytes=size_bytes,
                status="pending",
            )
            session.add(db_file)
            await session.commit()

        asyncio.create_task(
            request.app.state.file_ingestion.process(file_id)
        )
        results.append({"file_id": str(file_id), "filename": safe_name, "status": "pending"})

    return JSONResponse({"files": results})


@router.get("/upload/{file_id}/status")
async def file_status(
    file_id: uuid.UUID,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> JSONResponse:
    async with get_session_maker()() as session:
        stmt = select(
            FileModel.status,
            FileModel.error_message,
            FileModel.inline_text,
            FileModel.original_name,
        ).where(FileModel.id == file_id)
        result = await session.execute(stmt)
        row = result.fetchone()

    if not row:
        raise HTTPException(404, "File not found")

    status, error_message, inline_text, original_name = row
    return JSONResponse({
        "file_id": str(file_id),
        "filename": original_name,
        "status": status,
        "error": error_message,
        "inline_text": inline_text,  # None if file was chunked into Qdrant
    })
