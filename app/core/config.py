import os
from typing import Dict, List
from pydantic_settings import BaseSettings

def _default_cache_dir() -> str:
    # Windows-friendly: prefer LocalAppData, else output/cache
    return os.path.join(os.getenv("LOCALAPPDATA", "output"), "mrag_cache")

class Settings(BaseSettings):
    APP_NAME: str = "Multimodal RAG SaaS"

    # API auth
    API_KEYS: Dict[str, str] = {"test-key": "test-user"}

    # LLM (Gemini)
    GOOGLE_API_KEY: str

    # Vector store / checkpoints
    CHROMA_DB_PATH: str = "output/chroma_db"
    CHECKPOINT_DIR: str = "output/checkpoints"

    # Rate limiting
    RATE_LIMIT_PER_MIN: int = 60

    # Uploads
    ALLOWED_UPLOAD_EXTS: List[str] = [".pdf"]

    # Retrieval
    N_RESULTS: int = 4
    RERANK: bool = True
    RERANK_TOP_K: int = 8

    # Chunking
    CHUNK_SIZE_CHARS: int = 1200
    CHUNK_OVERLAP_CHARS: int = 200

    # Background worker
    REDIS_URL: str = "redis://localhost:6379/0"

    # Ingestion cache/output (place generated files outside repo to avoid reload restarts)
    INGEST_CACHE_DIR: str = _default_cache_dir()

    # YouTube ingestion policy / tuning
    YT_PREFER_TRANSCRIPT: bool = True
    YT_MAX_DURATION_MIN: int = 40            # reject very long videos
    YT_DOWNLOAD_FORMAT: str = "bv*[height<=360]+ba/b[height<=360]"
    YT_TIMEOUT_SEC: int = 120                # each stage timeout boundary
    YT_RETRIES: int = 2

    # Frames (lazy off by default to keep it simpler; flip to True if you want)
    YT_LAZY_FRAMES: bool = True              # if True, skip pre-sampling
    YT_FRAME_STRIDE_SEC: int = 5
    YT_MAX_FRAMES: int = 120

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
