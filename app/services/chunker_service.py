from __future__ import annotations

import httpx

from typing import Any


class ChunkerServiceClient:
    """HTTP client for document-chunker service.

    Endpoints:
      POST /chunk       multipart/form-data — parse and chunk a file
      POST /chunk-text  application/json    — chunk plain text (model response / user message)

    Both return ChunkResponse:
      {
        "file_name": str,
        "file_format": str,
        "creation_date": str,
        "modification_date": str,
        "chunks": [
          { "raw": str, "lemmas": str, "meta": {} },
          ...
        ]
      }

    For /chunk-text, if the text fits within chunk_size tokens it is returned
    as a single chunk without lemmatization (fast path, meta.single_chunk=true).
    If it exceeds chunk_size, the full sentence-split + lemmatize pipeline runs.

    Supported file formats for /chunk: .pdf, .docx, .doc, .xlsx
    """

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def chunk(
        self,
        *,
        file_name: str,
        file_content: bytes,
        content_type: str = "application/octet-stream",
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, Any]:
        """Send a file to document-chunker and return the parsed ChunkResponse dict.

        Args:
            file_name:    Original filename including extension (e.g. "report.pdf").
                          The extension determines the parser used by the chunker.
            file_content: Raw bytes of the file.
            content_type: MIME type for the upload part (default: application/octet-stream).
            chunk_size:   Max tokens per chunk. Uses chunker service default if omitted.
            overlap:      Token overlap between chunks. Uses chunker service default if omitted.

        Returns:
            Parsed JSON response matching document-chunker ChunkResponse schema.

        Raises:
            httpx.HTTPStatusError: on 4xx/5xx responses from the chunker.
        """
        data: dict[str, Any] = {}
        if chunk_size is not None:
            data["chunk_size"] = chunk_size
        if overlap is not None:
            data["overlap"] = overlap

        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post(
                "/chunk",
                files={"file": (file_name, file_content, content_type)},
                data=data,
            )
            response.raise_for_status()
            return response.json()

    async def chunk_text(
        self,
        *,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> dict[str, Any]:
        """Chunk plain text (e.g. a model response or user message).

        Fast path (inside the chunker service): if the text fits within
        chunk_size tokens, it is returned as a single chunk without running
        the full lemmatization pipeline.

        Args:
            text:       Raw text to chunk.
            chunk_size: Max tokens per chunk. Uses chunker service default if omitted.
            overlap:    Sentence overlap between chunks. Uses chunker service default if omitted.

        Returns:
            Parsed JSON response matching document-chunker ChunkResponse schema.

        Raises:
            httpx.HTTPStatusError: on 4xx/5xx responses from the chunker.
        """
        payload: dict[str, Any] = {"text": text}
        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if overlap is not None:
            payload["overlap"] = overlap

        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post("/chunk-text", json=payload)
            response.raise_for_status()
            return response.json()
