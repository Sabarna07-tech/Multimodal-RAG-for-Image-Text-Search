from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefaults:
    """Default model identifiers used across the application."""

    text: str = "sentence-transformers/all-MiniLM-L6-v2"  # MiniLM encoder for text chunks/queries.
    clip: str = "openai/clip-vit-base-patch32"  # CLIP vision encoder for image embeddings.
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder reranker for text hits.


@dataclass(frozen=True)
class GeminiDefaults:
    """Default Gemini configuration."""

    model: str = "gemini-pro-vision"  # Multimodal Gemini model supporting images.


@dataclass(frozen=True)
class PathDefaults:
    """Locations for persisted artifacts."""

    lancedb_dir: str = "output/lance_db"  # LanceDB storage root.
    media_dir: str = "output/media"  # Uploaded media (PDFs, audio, raw assets).
    thumbs_dir: str = "output/thumbs"  # Derived thumbnails and previews.
    ingest_cache_dir: str = "output/cache"  # Temporary ingest cache (frames/transcripts).
    notes_db_path: str = "output/notes.db"  # SQLite database for summaries/quizzes.
    checkpoint_dir: str = "output/checkpoints"  # Model checkpoints or LangGraph snapshots.


@dataclass(frozen=True)
class RateLimitDefaults:
    """HTTP rate limit defaults."""

    per_minute: int = 60  # Default requests per minute per API key.


@dataclass(frozen=True)
class RetrievalDefaults:
    """Retrieval hyperparameters."""

    use_rerank: bool = True  # Whether to apply cross-encoder reranking.
    index_topk_text: int = 50  # Candidate text nodes retrieved before rerank.
    index_topk_image: int = 12  # Candidate image nodes retrieved before rerank.
    rerank_topk: int = 8  # Cross-encoder rerank depth.
    final_n: int = 4  # Final context size returned to generators.
    confidence_tau: float = 0.25  # Similarity confidence threshold for fallback behaviour.


@dataclass(frozen=True)
class YouTubeDefaults:
    """YouTube ingestion tuning."""

    prefer_transcript: bool = True  # Attempt transcript-first ingestion for latency savings.
    frame_scene_threshold: float = 0.4  # Scene change threshold for frame sampling.
    max_frames: int = 120  # Upper bound on persisted frames per video.
    frame_dedup_delta: int = 6  # Perceptual hash distance to treat frames as duplicates.
    frame_stride_sec: int = 5  # Baseline uniform stride before scene-change overrides.
    lazy_frames: bool = True  # Defer frame extraction until needed.
    frame_extractor: str = "ffmpeg"  # Primary frame extraction backend.
    max_duration_min: int = 40  # Reject very long videos by default (minutes).
    download_format: str = "bv*[height<=360]+ba/b[height<=360]"  # yt-dlp format selector.
    retries: int = 2  # yt-dlp retry attempts.
    timeout_sec: int = 120  # Stage timeout guard.


@dataclass(frozen=True)
class ApiDefaults:
    """API/auth defaults."""

    api_keys: str = '{"test-key": "test-user"}'  # JSON map of API keys -> user IDs.
    redis_url: str = "redis://localhost:6379/0"  # Redis endpoint for rate limiting / Celery.


@dataclass(frozen=True)
class UploadDefaults:
    """Upload constraints."""

    allowed_exts: str = ".pdf"  # Comma-separated list of allowed document extensions.


@dataclass(frozen=True)
class ChunkDefaults:
    """Chunking controls for PDF/text ingestion."""

    size_chars: int = 1200  # Character count per chunk.
    overlap_chars: int = 200  # Character overlap between sequential chunks.


@dataclass(frozen=True)
class NotesDefaults:
    """Study notes/quizzes generation configuration."""

    context_chars: int = 4000  # Context budget when generating notes.
    quiz_questions: int = 5  # Default number of quiz questions.


@dataclass(frozen=True)
class AppDefaults:
    """Aggregate of all default settings."""

    models: ModelDefaults = ModelDefaults()
    gemini: GeminiDefaults = GeminiDefaults()
    paths: PathDefaults = PathDefaults()
    rate_limit: RateLimitDefaults = RateLimitDefaults()
    retrieval: RetrievalDefaults = RetrievalDefaults()
    youtube: YouTubeDefaults = YouTubeDefaults()
    api: ApiDefaults = ApiDefaults()
    uploads: UploadDefaults = UploadDefaults()
    chunks: ChunkDefaults = ChunkDefaults()
    notes: NotesDefaults = NotesDefaults()
    app_name: str = "Multimodal RAG SaaS"
