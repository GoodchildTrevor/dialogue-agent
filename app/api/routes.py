from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Request, Depends, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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

# Some upstream bot bridges (e.g. BotX) report a generic content_type for
# extensions their local mimetypes database doesn't know about, instead of
# the real MIME type. Fall back to guessing from the filename extension in
# that case rather than rejecting a legitimate file with 415.
_GENERIC_CONTENT_TYPES = {None, "", "application/octet-stream"}
_EXTENSION_MIME_FALLBACK = {
    ".xlsx": _XLSX_MIME,
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# Excel MCP tools whose successful result means a file on disk was modified
# and should be offered back to the user as a download/attachment. Matches
# every write tool dialogue-agent-mcp's excel.py exposes.
_EXCEL_WRITE_TOOLS = {
    "fill_cells",
    "fill_cell_by_index",
    "delete_rows",
    "insert_rows",
    "find_replace",
}
_EXCEL_WRITE_STATUSES = {"ok", "updated", "appended"}


def _resolve_content_type(filename: str | None, content_type: str | None) -> str | None:
    """Return the MIME type to actually use for a given upload.

    If the client-reported content_type is missing or a generic
    "application/octet-stream" placeholder, guess it from the filename
    extension instead (stdlib mimetypes first, then a small local fallback
    table for extensions not always present in mimetypes.types_map).
    """
    if content_type not in _GENERIC_CONTENT_TYPES:
        return content_type

    suffix = Path(filename or "").suffix.lower()
    guessed, _ = mimetypes.guess_type(filename or "")
    return guessed or _EXTENSION_MIME_FALLBACK.get(suffix, content_type)


def _unwrap_mcp_payload(result: dict | None) -> dict:
    """Unwrap the raw MCP tool-call envelope into the tool's actual JSON payload.

    tool_executor stores MCP results as-is: {"content": "<json text>",
    "images": [...]}. The useful fields the tool returned (e.g. "status",
    "file_id", "url") are inside that JSON-encoded string under "content",
    not at the top level of the dict. Reading `result.get("file_id")`
    directly always returns None and silently drops every successful edit -
    this was why modified_files came back empty even when
    fill_cell_by_index/fill_cells succeeded (confirmed ok=True in
    tool_executor logs).
    """
    if not isinstance(result, dict):
        return {}
    content = result.get("content")
    if not isinstance(content, str):
        return result
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_modified_files(
    tool_results: list[dict] | None,
    uploaded_files: list[dict] | None,
) -> list[dict]:
    """Determine which uploaded files were actually modified this turn.

    Inspects the graph's tool_results (ground truth from real tool calls,
    never trusting the model's own claims in free text) for successful Excel
    write-tool calls and returns the corresponding {file_id, filename, url}
    entries, deduplicated by file_id.

    The download url is read directly from the MCP tool's own JSON result -
    dialogue-agent-mcp's write tools already copy the edited file into the
    shared export volume and build this url themselves (see
    dialogue-agent-mcp's excel.py:_publish_for_download), so there is no
    second publish step here: this agent used to look storage_path back up
    in Postgres and copy the file a second time into /shared-exports just to
    build its own url, which was a redundant duplicate of work the MCP tool
    already did. url may be None if PUBLIC_FILES_BASE_URL isn't configured
    on the MCP side - callers must treat it as optional.
    """
    filename_by_id = {f["file_id"]: f["filename"] for f in (uploaded_files or [])}
    modified: dict[str, dict] = {}

    for tr in tool_results or []:
        if tr.get("tool") not in _EXCEL_WRITE_TOOLS or not tr.get("ok"):
            continue
        payload = _unwrap_mcp_payload(tr.get("result"))
        if payload.get("status") not in _EXCEL_WRITE_STATUSES:
            continue
        file_id = payload.get("file_id")
        if not file_id:
            continue
        modified[file_id] = {
            "file_id": file_id,
            "filename": filename_by_id.get(file_id, f"{file_id}.xlsx"),
            "url": payload.get("url"),
        }

    return list(modified.values())


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

    Runs as a fire-and-forget task - never raises to the caller.

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
    :returns: A ChatResponse containing the assistant's answer, images, sources, and any
        files that were modified as a result of this turn's tool calls.
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
    modified_files = _extract_modified_files(result.get("tool_results"), enriched_files)

    asyncio.create_task(
        _save_and_embed(
            runtime,
            user_id=payload.user_id,
            user_content=payload.message,
            assistant_content=answer,
        )
    )

    return ChatResponse(answer=answer, images=images, sources=sources, modified_files=modified_files)


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
            modified_files = _extract_modified_files(result.get("tool_results"), enriched_files)

            asyncio.create_task(
                _save_and_embed(
                    runtime,
                    user_id=payload.user_id,
                    user_content=payload.message,
                    assistant_content=answer,
                )
            )

            yield _sse({"answer": answer, "images": images, "sources": sources, "modified_files": modified_files})
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

    resolved_types = {
        id(upload): _resolve_content_type(upload.filename, upload.content_type)
        for upload in files
    }
    for upload in files:
        resolved = resolved_types[id(upload)]
        if resolved not in settings.ALLOWED_MIME_TYPES:
            raise HTTPException(415, f"Unsupported type: {resolved}")

    results = []
    for upload in files:
        content_type = resolved_types[id(upload)]
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

        is_xlsx = content_type == _XLSX_MIME

        async with get_session_maker()() as session:
            db_file = FileModel(
                id=file_id,
                user_id=user_id,
                original_name=upload.filename,
                mime_type=content_type,
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


@router.get("/files/{file_id}")
async def download_file(
    file_id: uuid.UUID,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> FileResponse:
    """Download the current bytes of a previously uploaded/modified file.

    Serves directly from storage_path on disk - the same file that the Excel
    MCP tools edit in place, so this always reflects the latest saved state
    without any extra round trip to the MCP service.

    :param file_id: The UUID of the file to download.
    :param request: The FastAPI Request object providing access to app state.
    :param api_key: Validated API key from the X-API-Key header (auto-checked via dependency).
    :returns: A FileResponse streaming the file's current bytes.
    :raises HTTPException: 404 Not Found if no file matches the given ID or it is missing on disk.
    """
    async with get_session_maker()() as session:
        stmt = select(
            FileModel.storage_path,
            FileModel.original_name,
            FileModel.mime_type,
        ).where(FileModel.id == file_id)
        result = await session.execute(stmt)
        row = result.fetchone()

    if not row:
        raise HTTPException(404, "File not found")

    storage_path, original_name, mime_type = row
    path = Path(storage_path)
    if not path.exists():
        raise HTTPException(404, "File not found on disk")

    return FileResponse(path, filename=original_name, media_type=mime_type)
