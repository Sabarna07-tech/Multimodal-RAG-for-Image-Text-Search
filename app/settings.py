from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from config import AppDefaults

DEFAULTS = AppDefaults()


def _read_env(env: Optional[Mapping[str, str]], key: str, default: str) -> str:
    if env is not None and key in env:
        return env[key]
    return os.getenv(key, default)


def _read_int(env: Optional[Mapping[str, str]], key: str, default: int) -> int:
    raw = _read_env(env, key, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Environment variable {key} must be an integer, got '{raw}'.")


def _read_float(env: Optional[Mapping[str, str]], key: str, default: float) -> float:
    raw = _read_env(env, key, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Environment variable {key} must be a float, got '{raw}'.")


def _read_bool(env: Optional[Mapping[str, str]], key: str, default: bool) -> bool:
    raw = _read_env(env, key, str(default))
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _read_json_dict(env: Optional[Mapping[str, str]], key: str, default_json: str) -> Dict[str, str]:
    raw = _read_env(env, key, default_json)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error path.
        raise ValueError(f"Environment variable {key} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Environment variable {key} must be a JSON object.")
    return {str(k): str(v) for k, v in parsed.items()}


def _read_csv(env: Optional[Mapping[str, str]], key: str, default: str) -> tuple[str, ...]:
    raw = _read_env(env, key, default)
    items = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    normalized = tuple(item if item.startswith(".") else f".{item}" for item in items)
    return normalized or (default,)


@dataclass(frozen=True)
class ModelSettings:
    """Model identifiers for embeddings and rerankers."""

    text: str
    clip: str
    reranker: str


@dataclass(frozen=True)
class GeminiSettings:
    """Gemini API configuration."""

    api_key: str
    model: str


@dataclass(frozen=True)
class PathSettings:
    """Filesystem layout for persisted data."""

    lancedb_dir: str
    media_dir: str
    thumbs_dir: str
    ingest_cache_dir: str
    notes_db_path: str
    checkpoint_dir: str


@dataclass(frozen=True)
class RateLimitSettings:
    """Rate limiting policy."""

    per_minute: int


@dataclass(frozen=True)
class RetrievalSettings:
    """Retrieval, reranking, and output sizing configuration."""

    use_rerank: bool
    index_topk_text: int
    index_topk_image: int
    rerank_topk: int
    final_n: int
    confidence_tau: float


@dataclass(frozen=True)
class ChunkSettings:
    """Chunking configuration for textual sources."""

    size_chars: int
    overlap_chars: int


@dataclass(frozen=True)
class UploadSettings:
    """File upload constraints."""

    allowed_exts: tuple[str, ...]


@dataclass(frozen=True)
class NotesSettings:
    """Study notes and quiz generation parameters."""

    context_chars: int
    quiz_questions: int


@dataclass(frozen=True)
class YouTubeSettings:
    """YouTube ingestion configuration."""

    prefer_transcript: bool
    frame_scene_threshold: float
    max_frames: int
    frame_dedup_delta: int
    frame_stride_sec: int
    lazy_frames: bool
    frame_extractor: str
    max_duration_min: int
    download_format: str
    retries: int
    timeout_sec: int


@dataclass(frozen=True)
class ApiSettings:
    """Authentication and infrastructure settings."""

    app_name: str
    api_keys: Dict[str, str]
    redis_url: str


@dataclass(frozen=True)
class AppSettings:
    """Top-level application settings composed from environment variables."""

    models: ModelSettings
    gemini: GeminiSettings
    paths: PathSettings
    rate_limit: RateLimitSettings
    retrieval: RetrievalSettings
    chunks: ChunkSettings
    uploads: UploadSettings
    notes: NotesSettings
    youtube: YouTubeSettings
    api: ApiSettings


def load_settings(env: Optional[Mapping[str, str]] = None) -> AppSettings:
    """Load configuration from environment variables with validation and defaults."""

    models = ModelSettings(
        text=_read_env(env, "MODEL_TEXT", DEFAULTS.models.text),
        clip=_read_env(env, "MODEL_CLIP", DEFAULTS.models.clip),
        reranker=_read_env(env, "RERANKER_MODEL", DEFAULTS.models.reranker),
    )

    gemini = GeminiSettings(
        api_key=_read_env(env, "GEMINI_API_KEY", ""),
        model=_read_env(env, "GEMINI_MODEL", DEFAULTS.gemini.model),
    )

    paths = PathSettings(
        lancedb_dir=_read_env(env, "LANCEDB_DIR", DEFAULTS.paths.lancedb_dir),
        media_dir=_read_env(env, "MEDIA_DIR", DEFAULTS.paths.media_dir),
        thumbs_dir=_read_env(env, "THUMBS_DIR", DEFAULTS.paths.thumbs_dir),
        ingest_cache_dir=_read_env(env, "INGEST_CACHE_DIR", DEFAULTS.paths.ingest_cache_dir),
        notes_db_path=_read_env(env, "NOTES_DB_PATH", DEFAULTS.paths.notes_db_path),
        checkpoint_dir=_read_env(env, "CHECKPOINT_DIR", DEFAULTS.paths.checkpoint_dir),
    )

    rate_limit = RateLimitSettings(
        per_minute=_read_int(env, "RATE_LIMIT_PER_MIN", DEFAULTS.rate_limit.per_minute)
    )

    retrieval = RetrievalSettings(
        use_rerank=_read_bool(env, "RERANK_ENABLED", DEFAULTS.retrieval.use_rerank),
        index_topk_text=_read_int(env, "INDEX_TOPK_TEXT", DEFAULTS.retrieval.index_topk_text),
        index_topk_image=_read_int(env, "INDEX_TOPK_IMG", DEFAULTS.retrieval.index_topk_image),
        rerank_topk=_read_int(env, "RERANK_TOPK", DEFAULTS.retrieval.rerank_topk),
        final_n=_read_int(env, "FINAL_N", DEFAULTS.retrieval.final_n),
        confidence_tau=_read_float(env, "CONFIDENCE_TAU", DEFAULTS.retrieval.confidence_tau),
    )

    chunks = ChunkSettings(
        size_chars=_read_int(env, "CHUNK_SIZE_CHARS", DEFAULTS.chunks.size_chars),
        overlap_chars=_read_int(env, "CHUNK_OVERLAP_CHARS", DEFAULTS.chunks.overlap_chars),
    )

    uploads = UploadSettings(
        allowed_exts=_read_csv(env, "ALLOWED_UPLOAD_EXTS", DEFAULTS.uploads.allowed_exts),
    )

    notes = NotesSettings(
        context_chars=_read_int(env, "NOTE_CONTEXT_CHARS", DEFAULTS.notes.context_chars),
        quiz_questions=_read_int(env, "QUIZ_QUESTION_COUNT", DEFAULTS.notes.quiz_questions),
    )

    youtube = YouTubeSettings(
        prefer_transcript=_read_bool(env, "YT_PREFER_TRANSCRIPT", DEFAULTS.youtube.prefer_transcript),
        frame_scene_threshold=_read_float(env, "YT_FRAME_SCENE_THRESH", DEFAULTS.youtube.frame_scene_threshold),
        max_frames=_read_int(env, "YT_MAX_FRAMES", DEFAULTS.youtube.max_frames),
        frame_dedup_delta=_read_int(env, "YT_FRAME_DEDUP_DELTA", DEFAULTS.youtube.frame_dedup_delta),
        frame_stride_sec=_read_int(env, "YT_FRAME_STRIDE_SEC", DEFAULTS.youtube.frame_stride_sec),
        lazy_frames=_read_bool(env, "YT_LAZY_FRAMES", DEFAULTS.youtube.lazy_frames),
        frame_extractor=_read_env(env, "YT_FRAME_EXTRACTOR", DEFAULTS.youtube.frame_extractor),
        max_duration_min=_read_int(env, "YT_MAX_DURATION_MIN", DEFAULTS.youtube.max_duration_min),
        download_format=_read_env(env, "YT_DOWNLOAD_FORMAT", DEFAULTS.youtube.download_format),
        retries=_read_int(env, "YT_RETRIES", DEFAULTS.youtube.retries),
        timeout_sec=_read_int(env, "YT_TIMEOUT_SEC", DEFAULTS.youtube.timeout_sec),
    )

    api = ApiSettings(
        app_name=_read_env(env, "APP_NAME", DEFAULTS.app_name),
        api_keys=_read_json_dict(env, "API_KEYS", DEFAULTS.api.api_keys),
        redis_url=_read_env(env, "REDIS_URL", DEFAULTS.api.redis_url),
    )

    return AppSettings(
        models=models,
        gemini=gemini,
        paths=paths,
        rate_limit=rate_limit,
        retrieval=retrieval,
        chunks=chunks,
        uploads=uploads,
        notes=notes,
        youtube=youtube,
        api=api,
    )


# [MIGRATE] Global singleton used across the application. Import from `app.settings`.
settings = load_settings()
