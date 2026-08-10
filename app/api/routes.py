from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Request, Depends, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from sqlalchemy import select

from app.api.schemas import ChatRequest, ChatResponse, UploadedFile
from app.core.config import Settings
from app.db.models import File as FileModel
from app.db.session import get_session_maker
from app.graph.graph_runtime import GraphRuntime

router = APIRouter()
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key")

# MIME type for .xlsx spreadsheets: analyzed via pandas/openpyxl by the Excel
# MCP tools, never chunked into Qdrant (there is nothing to embed).
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def get_settings_dep(request: Request) -> Settings:
    """Retrieve application settings from the FastAPI app state.

    :param request: The FastAPI Request object containing app state.
    :returns: The Settings instance configured at application startup.
    """
    return request.app.state.settings


async def get_api_key(
    api_key: str = Depends(api_key_header),
    settings: Settings = Depends(get_settings_dep),
) -> str:
    """Validate the API key provided in the X-API-Key request header.

    :param api_key: The API key extracted from the ``X-API-Key`` header.
    :param settings: Application settings containing the expected API key.
    :returns: The validated API key string.
    :raises HTTPException: 403 Forbidden if the API key is invalid.
    """
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def get_runtime(request: Request) -> GraphRuntime:
    """Retrieve the GraphRuntime singleton from the FastAPI app state.

    :param request: The FastAPI Request object providing access to ``app.state``.
    :returns: The GraphRuntime instance used for chat/dialogue orchestration.
    """
    return request.app.state.runtime


async def _save_and_embed(
    runtime: GraphRuntime,
    *,
    user_id: str,
    user_content: str,
    assistant_content: str,
) -> None:
    """Save user + assistant messages to PostgreSQL and embed them asynchronously.

    Runs as a fire-and-forget task — never raises to the caller.

    :param runtime: The GraphRuntime instance providing service access.
    :param user_id: The identifier of the user whose messages are being saved.
    :param user_content: The user's message text to save and embed.
    :param assistant_content: The assistant's response text to save and embed.
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
    """Encode *payload* as a plain SSE data frame with no event name.

    :param payload: The dictionary to serialize as JSON within the ``data:`` line.
    :returns: A string formatted as an SSE data frame ending with double newlines.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_MAX_FILENAME_LEN = 200


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename by extracting its base name, replacing invalid characters,
    and truncating to the maximum allowed length.

    :param name: The original filename string, which may include path components.
    :returns: A sanitized filename safe for filesystem storage.
    """
    name = Path(name).name
    name = re.sub(r"[^\w.\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > _MAX_FILENAME_LEN:
        name = name[:_MAX_FILENAME_LEN]
    return name or "file"


async def _enrich_uploaded_files(uploaded_files) -> list[dict]:
    """Fetch inline_text from the database for each uploaded file.

    The client only sends file_id + filename. This function resolves inline_text
    server-side so the orchestrator can decide whether to use inline text or call
    document_searcher.

    :param uploaded_files: An iterable of UploadedFile objects containing file_id values.
    :returns: A list of dictionaries with keys ``file_id``, ``filename``, and ``inline_text``.
    """
    if not uploaded_files:
        return []

    file_ids = [uuid.UUID(f.file_id) for f in uploaded_files]

    async with get_session_maker()() as session:
        stmt = select(
            FileModel.id,
            FileModel.inline_text,
        ).where(FileModel.id.in_(file_ids))
        result = await session.execute(stmt)
        inline_map = {str(row.id): row.inline_text for row in result.fetchall()}

    return [
        {
            "file_id": f.file_id,
            "filename": f.filename,
            "inline_text": inline_map.get(f.file_id),
        }
        for f in uploaded_files
    ]


async def _auto_attach_recent_files(user_id: str, minutes: int) -> list[UploadedFile]:
    """Find the user's recently indexed files and return them as an UploadedFile list.

    When the client does not explicitly pass uploaded_files, this fallback queries
    the database for files belonging to this user that were successfully indexed
    within the last ``minutes`` minutes. This allows the system to automatically
    associate a previously uploaded file with the current chat request.

    :param user_id: The identifier of the user whose recent files to find.
    :param minutes: The time window in minutes to search for recently indexed files.
    :returns: A list of UploadedFile objects for recent files, or an empty list if none found.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    async with get_session_maker()() as session:
        stmt = (
            select(FileModel.id, FileModel.original_name)
            .where(
                FileModel.user_id == user_id,
                FileModel.status == "indexed",
                FileModel.created_at >= cutoff,
            )
            .order_by(FileModel.created_at.desc())
        )
        result = await session.execute(stmt)
        rows = result.fetchall()

    if not rows:
        return []

    files = [
        UploadedFile(file_id=str(row.id), filename=row.original_name)
        for row in rows
    ]
    logger.info(
        "Auto-attached %d recent file(s) for user %s (window=%d min): %s",
        len(files),
        user_id,
        minutes,
        [f.file_id for f in files],
    )
    return files


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/healthz")
async def healthz() -> JSONResponse:
    """Health check endpoint that returns the service status.

    :returns: A JSONResponse with ``{"status": "ok"}`` indicating the service is healthy.
    """
    return JSONResponse({"status": "ok"})


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> ChatResponse:
    """Process a chat message and return the assistant's response.

    Handles non-streaming chat requests by building an initial state, running
    the graph runtime, and asynchronously saving the conversation to the database.

    :param payload: The chat request containing user_id, message, and optional uploaded_files.
    :param request: The FastAPI Request object providing access to app state and settings.
    :param api_key: Validated API key from the X-API-Key header (auto-checked via dependency).
    :returns: A ChatResponse containing the assistant's answer, images, and sources.
    """
    settings: Settings = request.app.state.settings
    logger.info("CHAT uploaded_files: %s", payload.uploaded_files)

    # Auto-attach recent files if the client didn't pass any.
    uploaded_files = payload.uploaded_files
    if not uploaded_files and settings.FILE_AUTO_ATTACH_MINUTES > 0:
        uploaded_files = await _auto_attach_recent_files(
            payload.user_id, settings.FILE_AUTO_ATTACH_MINUTES
        )

    enriched_files = await _enrich_uploaded_files(uploaded_files)
    logger.info("CHAT enriched_files: %s", enriched_files)
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    runtime = get_runtime(request)
    state = runtime.build_initial_state(
        user_id=payload.user_id,
        message=payload.message,
        request_id=request_id,
        uploaded_files=enriched_files,
    )
    result = await runtime.run(state)
    answer = result.get("final_answer", "")
    images = result.get("images") or []
    sources = result.get("sources") or []

    asyncio.create_task(
        _save_and_embed(
            runtime,
            user_id=payload.user_id,
            user_content=payload.message,
            assistant_content=answer,
        )
    )

    return ChatResponse(answer=answer, images=images, sources=sources)


@router.post("/stream")
async def stream(
    payload: ChatRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> StreamingResponse:
    """Process a chat message and stream the assistant's response via Server-Sent Events.

    Handles streaming chat requests by building an initial state, running the graph
    runtime concurrently, and yielding status updates followed by the final answer.

    :param payload: The chat request containing user_id, message, and optional uploaded_files.
    :param request: The FastAPI Request object providing access to app state and settings.
    :param api_key: Validated API key from the X-API-Key header (auto-checked via dependency).
    :returns: A StreamingResponse with SSE-formatted status updates and the final answer.
    """
    settings: Settings = request.app.state.settings
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    runtime = get_runtime(request)
    queue: asyncio.Queue[str] = asyncio.Queue()

    # Auto-attach recent files if the client didn't pass any.
    uploaded_files = payload.uploaded_files
    if not uploaded_files and settings.FILE_AUTO_ATTACH_MINUTES > 0:
        uploaded_files = await _auto_attach_recent_files(
            payload.user_id, settings.FILE_AUTO_ATTACH_MINUTES
        )

    enriched_files = await _enrich_uploaded_files(uploaded_files)
    initial_state = runtime.build_initial_state(
        user_id=payload.user_id,
        message=payload.message,
        request_id=request_id,
        status_queue=queue,
        uploaded_files=enriched_files,
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
            images = result.get("images") or []
            sources = result.get("sources") or []

            asyncio.create_task(
                _save_and_embed(
                    runtime,
                    user_id=payload.user_id,
                    user_content=payload.message,
                    assistant_content=answer,
                )
            )

            yield _sse({"answer": answer, "images": images, "sources": sources})
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
    """Upload one or more files and register them in the database for processing.

    Validates MIME types against allowed settings, saves files to disk, creates
    database records, and triggers asynchronous file ingestion - except for
    .xlsx spreadsheets, which are analyzed via pandas/openpyxl by the Excel MCP
    tools and are marked "indexed" immediately since there is nothing to chunk.

    :param request: The FastAPI Request object providing access to app state and settings.
    :param user_id: The identifier of the user uploading files (from form data).
    :param files: A list of UploadFile objects to be saved and processed.
    :param api_key: Validated API key from the X-API-Key header (auto-checked via dependency).
    :returns: A JSONResponse containing file IDs, filenames, and processing status.
    :raises HTTPException: 415 if a file has an unsupported MIME type, 413 if a file exceeds the size limit.
    """
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

        is_xlsx = upload.content_type == _XLSX_MIME

        async with get_session_maker()() as session:
            db_file = FileModel(
                id=file_id,
                user_id=user_id,
                original_name=upload.filename,
                mime_type=upload.content_type,
                storage_path=str(path),
                size_bytes=size_bytes,
                status="indexed" if is_xlsx else "pending",
            )
            session.add(db_file)
            await session.commit()

        if not is_xlsx:
            asyncio.create_task(
                request.app.state.file_ingestion.process(file_id)
            )
        results.append({
            "file_id": str(file_id),
            "filename": safe_name,
            "status": db_file.status,
        })

    return JSONResponse({"files": results})


@router.get("/upload/{file_id}/status")
async def file_status(
    file_id: uuid.UUID,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> JSONResponse:
    """Retrieve the processing status of an uploaded file.

    Queries the database for the file record matching the given UUID and returns
    its status, error message (if any), inline text, and original filename.

    :param file_id: The UUID of the uploaded file to look up.
    :param request: The FastAPI Request object providing access to app state.
    :param api_key: Validated API key from the X-API-Key header (auto-checked via dependency).
    :returns: A JSONResponse with file_id, filename, status, error, and inline_text.
    :raises HTTPException: 404 Not Found if no file matches the given ID.
    """
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
