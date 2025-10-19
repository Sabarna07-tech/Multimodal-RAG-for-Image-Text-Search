from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from pydantic import BaseModel

from app.cache import chat_cache
from app.celery_app import celery_app
from app.core.redis_client import get_redis
from app.ingest.pdf import extract_pdf_nodes
from app.ml.generate import generate_response
from app.ml.index_build import index_text_nodes, index_image_nodes
from app.ml.retrieve import retrieve
from app.settings import settings
from app.storage.note_store import NoteStore
from app.storage.schema import MetadataStore, Document as MetaDocument, Chunk as MetaChunk
from app.tasks import ingest_youtube_task
from app.utils.text_chunk import chunk_text  # noqa: F401 - retained for backwards compatibility

router = APIRouter()


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
note_store = NoteStore(settings.paths.notes_db_path)
metadata_store = MetadataStore(os.path.join(settings.paths.lancedb_dir, "metadata.sqlite3"))


def key_by_user(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    return settings.api.api_keys.get(api_key) or get_remote_address(request)


limiter = Limiter(key_func=key_by_user)


def get_user_id(api_key: str = Security(api_key_header)) -> str:
    user_id = settings.api.api_keys.get(api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user_id


class _InMemoryCache:
    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:  # pragma: no cover - TTL ignored
        self._store[key] = value


_redis_client: Optional[Any] = None


def _redis() -> Any:
    global _redis_client
    if _redis_client is None:
        try:
            client = get_redis()
            client.ping()
            _redis_client = client
        except Exception:
            _redis_client = _InMemoryCache()
    return _redis_client


def _validate_upload(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in settings.uploads.allowed_exts:
        raise HTTPException(status_code=400, detail=f"Only {settings.uploads.allowed_exts} allowed")


def _persist_upload(src_path: Path, user_id: str, doc_token: str) -> Path:
    target_dir = Path(settings.paths.ingest_cache_dir) / "uploads" / user_id / doc_token
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_path = target_dir / src_path.name
    shutil.copy2(src_path, dest_path)
    return dest_path


@router.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}


@router.post("/process-pdf")
@router.post("/process-pdf/")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def process_pdf(request: Request, file: UploadFile = File(...), user_id: str = Depends(get_user_id)):
    _validate_upload(file.filename)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / file.filename
        with open(pdf_path, "wb") as buffer:
            buffer.write(await file.read())

        doc_token = str(uuid.uuid4())
        persisted_pdf = _persist_upload(pdf_path, user_id, doc_token)

    text_nodes, image_nodes = extract_pdf_nodes(persisted_pdf, user_id, doc_token)
    if not text_nodes and not image_nodes:
        raise HTTPException(status_code=400, detail="No extractable text/images found")

    document_record = MetaDocument(
        id=doc_token,
        user_id=user_id,
        source_type="pdf",
        source_uri=str(persisted_pdf),
        title=file.filename,
        status="processing",
    )
    metadata_store.upsert_document(document_record)

    indexed_text = index_text_nodes(user_id, text_nodes)
    if indexed_text:
        chunks = [
            MetaChunk(
                id=item["chunk_id"],
                document_id=doc_token,
                modality="text",
                text=item.get("text"),
                page_no=item["metadata"].get("page_no"),
                file_path=str(persisted_pdf),
                meta=item["metadata"],
            )
            for item in indexed_text
        ]
        metadata_store.upsert_chunks(chunks)

    indexed_images = index_image_nodes(user_id, image_nodes)
    if indexed_images:
        chunks = [
            MetaChunk(
                id=item["chunk_id"],
                document_id=doc_token,
                modality="image",
                file_path=item["metadata"].get("file_path"),
                page_no=item["metadata"].get("page_no"),
                meta=item["metadata"],
            )
            for item in indexed_images
        ]
        metadata_store.upsert_chunks(chunks)

    metadata_store.upsert_document(document_record.copy(update={"status": "ready"}))

    return {
        "status": "ok",
        "text_chunks_indexed": len(indexed_text),
        "images_indexed": len(indexed_images),
    }


def _enqueue_youtube(request: Request, user_id: str, url: str) -> JSONResponse:
    if not url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")

    idem = request.headers.get("Idempotency-Key")
    redis_client = _redis()
    if idem:
        cache_key = f"idempotency:{user_id}:{idem}"
        existing = redis_client.get(cache_key)
        if existing:
            result = celery_app.AsyncResult(existing)
            return JSONResponse(status_code=202, content={"job_id": existing, "state": result.state})

    task = ingest_youtube_task.delay(user_id, url)

    if idem:
        redis_client.setex(f"idempotency:{user_id}:{idem}", 3600, task.id)

    return JSONResponse(status_code=202, content={"job_id": task.id, "state": "PENDING"})


@router.post("/process-youtube/")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def process_youtube_form(request: Request, url: str = Form(...), user_id: str = Depends(get_user_id)):
    return _enqueue_youtube(request, user_id, url)


@router.post("/ingest-youtube")
@router.post("/ingest/youtube")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def enqueue_youtube(request: Request, payload: YouTubeIngestRequest, user_id: str = Depends(get_user_id)):
    return _enqueue_youtube(request, user_id, payload.url)


def _progress_from_state(state_meta: Dict[str, Any]) -> int:
    stage = state_meta.get("stage")
    mapping = {
        "begin": 5,
        "metadata": 20,
        "transcript": 55,
        "frames": 80,
        "ready": 100,
        "extract": 35,
    }
    return mapping.get(stage, 0)


@router.get("/yt_status/{job_id}")
@router.get("/ingest/status/{job_id}")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def ingest_status(request: Request, job_id: str, user_id: str = Depends(get_user_id)):
    result = celery_app.AsyncResult(job_id)
    body: Dict[str, Any] = {"job_id": job_id, "state": result.state}
    meta = result.info if isinstance(result.info, dict) else {}
    if meta:
        body.update(meta)
        body["progress_pct"] = _progress_from_state(meta)
    if result.state == "SUCCESS":
        payload = result.result if isinstance(result.result, dict) else {"result": str(result.result)}
        body.update(payload)
        body["progress_pct"] = 100
    return body


@router.get("/videos/")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def list_videos(request: Request, user_id: str = Depends(get_user_id)):
    return {"videos": note_store.list_videos(user_id)}


@router.get("/videos/{video_id}/notes")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def get_video_notes(request: Request, video_id: str, user_id: str = Depends(get_user_id)):
    record = note_store.get_video(user_id, video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video notes not found")
    return record


@router.get("/videos/{video_id}/quiz")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def get_video_quiz(request: Request, video_id: str, user_id: str = Depends(get_user_id)):
    record = note_store.get_quiz(user_id, video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return record


class ChatRequest(BaseModel):  # type: ignore[name-defined]
    message: str
    thread_id: Optional[str] = None
    video_id: Optional[str] = None


class YouTubeIngestRequest(BaseModel):  # type: ignore[name-defined]
    url: str


@router.post("/chat_pro")
@router.post("/chat/")
@limiter.limit(f"{settings.rate_limit.per_minute}/minute")
async def chat(request: Request, chat_req: ChatRequest, user_id: str = Depends(get_user_id)):
    if not chat_req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    filters = {"video_id": chat_req.video_id} if chat_req.video_id else None

    @chat_cache()
    def _generate(user_id: str, query: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        items = retrieve(user_id, query)
        if filters:
            items = [item for item in items if item["metadata"].get("video_id") == filters.get("video_id")]
        return generate_response(query, items)

    payload = _generate(user_id, chat_req.message, filters)
    return payload
