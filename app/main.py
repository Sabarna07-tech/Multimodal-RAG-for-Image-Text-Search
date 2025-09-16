import os
import re
import uuid
import tempfile
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Request, Form
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback for environments without loguru
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("app")

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.core.redis_client import get_redis
from app.data_extraction.pdf_extractor import extract_pdf_data
from app.embedding.embedder import Embedder
from app.vector_store.chroma_store import ChromaStore
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.utils.text_chunk import chunk_pages

from app.celery_app import celery_app
from app.tasks import ingest_youtube_task


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title=settings.APP_NAME)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    logger.info(f"{request_id} {request.method} {request.url.path} start")
    try:
        response = await call_next(request)
        logger.info(f"{request_id} {request.method} {request.url.path} done {response.status_code}")
        return response
    except Exception:
        logger.exception(f"{request_id} unhandled error")
        raise


def key_by_user(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    return settings.API_KEYS.get(api_key) or get_remote_address(request)


limiter = Limiter(key_func=key_by_user)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ---------------------------------------------------------------------------
# Authentication and singletons
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_user_id(api_key: str = Security(api_key_header)) -> str:
    user_id = settings.API_KEYS.get(api_key)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user_id


_embedder = Embedder()
_generator = Generator(settings.GOOGLE_API_KEY)


# Lazy Redis client with in-memory fallback for local dev/test
class _InMemoryCache:
    def __init__(self):
        self._store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:  # pragma: no cover - TTL ignored
        self._store[key] = value


_cached_redis = None


def _redis_client():
    global _cached_redis
    if _cached_redis is None:
        try:
            client = get_redis()
            client.ping()
            _cached_redis = client
        except Exception as exc:  # pragma: no cover - executed only when Redis unavailable
            logger.warning(f"Redis unavailable, using in-memory fallback: {exc}")
            _cached_redis = _InMemoryCache()
    return _cached_redis


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str


class YouTubeIngestRequest(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_upload(filename: str) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_UPLOAD_EXTS:
        raise HTTPException(status_code=400, detail=f"Only {settings.ALLOWED_UPLOAD_EXTS} allowed")


def _ensure_store(user_id: str) -> ChromaStore:
    return ChromaStore(db_path=settings.CHROMA_DB_PATH, user_id=user_id)


def _parse_page_from_image_filename(path: str) -> Optional[int]:
    match = re.search(r"page(\d+)_img", os.path.basename(path))
    return int(match.group(1)) if match else None


def _enqueue_youtube(request: Request, user_id: str, url: str) -> JSONResponse:
    if not url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")

    idem = request.headers.get("Idempotency-Key")
    redis_client = _redis_client()
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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------
@app.post("/process-pdf/")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MIN}/minute")
async def process_pdf(request: Request, file: UploadFile = File(...), user_id: str = Depends(get_user_id)):
    _validate_upload(file.filename)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, file.filename)
        with open(pdf_path, "wb") as buf:
            buf.write(await file.read())

        pages, image_paths = extract_pdf_data(pdf_path)
        if not pages and not image_paths:
            raise HTTPException(status_code=400, detail="No extractable text/images found")

        text_chunks = chunk_pages(pages, settings.CHUNK_SIZE_CHARS, settings.CHUNK_OVERLAP_CHARS)

    store = _ensure_store(user_id)

    if text_chunks:
        documents = [chunk["text"] for chunk in text_chunks]
        metadatas = [
            {
                "source": "pdf",
                "file_name": file.filename,
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "user_id": user_id,
            }
            for chunk in text_chunks
        ]
        ids = [str(uuid.uuid4()) for _ in text_chunks]
        vectors = _embedder.embed_text(documents)
        store.text_collection.add(documents=documents, metadatas=metadatas, embeddings=vectors.tolist(), ids=ids)

    if image_paths:
        image_vectors = _embedder.embed_images(image_paths)
        metadatas_img = [
            {
                "source": "pdf",
                "file_name": file.filename,
                "page": _parse_page_from_image_filename(path),
                "image_path": path,
                "user_id": user_id,
            }
            for path in image_paths
        ]
        ids_img = [str(uuid.uuid4()) for _ in image_paths]
        store.image_collection.add(
            documents=[os.path.basename(meta["image_path"]) for meta in metadatas_img],
            metadatas=metadatas_img,
            embeddings=[vec.tolist() for vec in image_vectors],
            ids=ids_img,
        )

    return {"status": "ok", "text_chunks_indexed": len(text_chunks), "images_indexed": len(image_paths)}


@app.post("/process-youtube/")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MIN}/minute")
async def process_youtube_form(request: Request, url: str = Form(...), user_id: str = Depends(get_user_id)):
    response = _enqueue_youtube(request, user_id, url)
    return response


@app.post("/ingest/youtube")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MIN}/minute")
async def enqueue_youtube(request: Request, payload: YouTubeIngestRequest, user_id: str = Depends(get_user_id)):
    return _enqueue_youtube(request, user_id, payload.url)


@app.get("/ingest/status/{job_id}")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MIN}/minute")
async def ingest_status(request: Request, job_id: str, user_id: str = Depends(get_user_id)):
    result = celery_app.AsyncResult(job_id)
    body: Dict[str, Any] = {"job_id": job_id, "state": result.state}
    if result.info and isinstance(result.info, dict):
        body.update(result.info)
    if result.state == "SUCCESS":
        payload = result.result if isinstance(result.result, dict) else {"result": str(result.result)}
        body.update(payload)
    return body


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat/")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MIN}/minute")
async def chat(request: Request, chat_req: ChatRequest, user_id: str = Depends(get_user_id)):
    if not chat_req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    store = _ensure_store(user_id)
    retriever = Retriever(_embedder, store.text_collection, store.image_collection)

    results = retriever.retrieve(
        query=chat_req.message,
        n_results=settings.N_RESULTS,
        rerank=settings.RERANK,
        rerank_top_k=settings.RERANK_TOP_K,
    )

    answer = _generator.generate_answer(chat_req.message, results.get("text"), results.get("image"))
    text_citations = results.get("text", {}).get("metadatas", [[]])[0] if results.get("text") else []
    image_citations = results.get("image", {}).get("metadatas", [[]])[0] if results.get("image") else []

    return {"response": answer, "citations": {"text": text_citations, "images": image_citations}}


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def ui_index():
    return FileResponse("app/static/index.html")


app.mount("/static", StaticFiles(directory="app/static"), name="static")
