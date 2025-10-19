import os
import uuid
from pathlib import Path
from typing import List

import yt_dlp

from app.celery_app import celery_app
from app.core.redis_client import get_redis
from app.generation.generator import Generator
from app.ingest.youtube import (
    transcript_or_fallback,
    download_video,
    YouTubeIngestError,
    resolve_video_id,
    TranscriptSegment,
)
from app.ingest.frames import extract_scene_frames, FrameExtractionError
from app.ml.index_build import index_text_nodes, index_image_nodes
from app.settings import settings
from app.storage.note_store import NoteStore
from app.storage.schema import MetadataStore, Document as MetaDocument, Chunk as MetaChunk
from app.utils.note_builder import build_notes_payload

_generator = (
    Generator(settings.gemini.api_key, model_name=settings.gemini.model)
    if settings.gemini.api_key
    else None
)
redis = get_redis()
note_store = NoteStore(settings.paths.notes_db_path)
os.makedirs(settings.paths.lancedb_dir, exist_ok=True)
metadata_store = MetadataStore(os.path.join(settings.paths.lancedb_dir, "metadata.sqlite3"))


def _fetch_video_info(url: str) -> dict:
    """Retrieve lightweight metadata for the supplied YouTube URL."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as exc:  # pragma: no cover - network dependent
        raise YouTubeIngestError(f"Failed to fetch metadata for {url}") from exc


def _already_indexed_key(user_id: str, vid: str) -> str:
    return f"yt:indexed:{user_id}:{vid}"


@celery_app.task(bind=True)
def ingest_youtube_task(self, user_id: str, url: str):
    vid = resolve_video_id(url) or "unknown"
    self.update_state(state="STARTED", meta={"stage": "begin", "video_id": vid})

    if vid != "unknown" and redis.get(_already_indexed_key(user_id, vid)):
        record = note_store.get_video(user_id, vid)
        return {
            "status": "ok",
            "skipped": True,
            "text_chunks_indexed": 0,
            "images_indexed": 0,
            "notes": record or {},
        }

    try:
        # [MIGRATE] Transcript-first YouTube ingestion via new ingest modules.
        self.update_state(state="PROGRESS", meta={"stage": "metadata"})
        info = _fetch_video_info(url)

        doc_token = info.get("id") or vid or str(uuid.uuid4())
        vid = info.get("id") or vid or "unknown"
        video_link = info.get("webpage_url", url)

        document_record = MetaDocument(
            id=doc_token,
            user_id=user_id,
            source_type="youtube",
            source_uri=video_link,
            title=info.get("title"),
            status="processing",
        )
        metadata_store.upsert_document(document_record)

        self.update_state(state="PROGRESS", meta={"stage": "transcript"})
        segments, _audio_path, transcript_info = transcript_or_fallback(url)
        transcript_info = transcript_info or {}

        transcript_segments: List[TranscriptSegment] = segments or []
        text_nodes: List[dict] = []
        for idx, segment in enumerate(transcript_segments):
            meta = {
                "doc_id": doc_token,
                "user_id": user_id,
                "modality": "text",
                "source": "youtube",
                "video_url": video_link,
                "video_id": vid,
                "start_ts": segment.start,
                "end_ts": segment.end,
            }
            text_nodes.append(
                {
                    "id": f"{doc_token}:ts{idx}",
                    "text": segment.text,
                    "metadata": meta,
                }
            )

        indexed_text = index_text_nodes(user_id, text_nodes)
        text_chunks_indexed = len(indexed_text)
        if indexed_text:
            chunk_records = [
                MetaChunk(
                    id=item["chunk_id"],
                    document_id=doc_token,
                    modality="text",
                    text=item.get("text"),
                    start_ts=item["metadata"].get("start_ts"),
                    end_ts=item["metadata"].get("end_ts"),
                    file_path=video_link,
                    meta=item["metadata"],
                )
                for item in indexed_text
            ]
            metadata_store.upsert_chunks(chunk_records)

        video_path_str = transcript_info.get("video_path") if transcript_info else None
        if not video_path_str:
            video_path_str = str(download_video(url))
        video_path = Path(video_path_str)

        frames = []
        if not settings.youtube.lazy_frames:
            try:
                frames = extract_scene_frames(
                    video_path,
                    user_id,
                    doc_token,
                    settings.youtube.frame_scene_threshold,
                    settings.youtube.max_frames,
                    settings.youtube.frame_dedup_delta,
                )
            except FrameExtractionError:
                frames = []

        images_indexed = 0
        if frames:
            self.update_state(state="PROGRESS", meta={"stage": "frames", "count": len(frames)})
            image_nodes = [
                {
                    "id": f"{doc_token}:frame{idx}",
                    "metadata": {
                        "doc_id": doc_token,
                        "user_id": user_id,
                        "modality": "image",
                        "source": "youtube",
                        "video_url": video_link,
                        "video_id": vid,
                        "file_path": frame["file_path"],
                        "start_ts": frame.get("start_ts"),
                        "end_ts": frame.get("end_ts"),
                    },
                }
                for idx, frame in enumerate(frames)
            ]
            indexed_images = index_image_nodes(user_id, image_nodes)
            images_indexed = len(indexed_images)
            if indexed_images:
                chunk_records = [
                    MetaChunk(
                        id=item["chunk_id"],
                        document_id=doc_token,
                        modality="image",
                        start_ts=item["metadata"].get("start_ts"),
                        end_ts=item["metadata"].get("end_ts"),
                        file_path=item["metadata"].get("file_path"),
                        meta=item["metadata"],
                    )
                    for item in indexed_images
                ]
                metadata_store.upsert_chunks(chunk_records)

        metadata_store.upsert_document(document_record.copy(update={"status": "ready"}))

        transcript_payload = [
            {
                "text": segment.text,
                "start": segment.start,
                "duration": max(segment.end - segment.start, 0.0),
            }
            for segment in transcript_segments
        ]

        notes_payload = build_notes_payload(transcript_payload, info, vid, video_link, _generator)
        note_store.upsert(user_id, vid, notes_payload)

        if vid != "unknown":
            redis.setex(_already_indexed_key(user_id, vid), 86400, "1")

        return {
            "status": "ok",
            "text_chunks_indexed": text_chunks_indexed,
            "images_indexed": images_indexed,
            "notes": notes_payload,
        }

    except Exception as exc:
        self.update_state(state="FAILURE", meta={"stage": "error", "error": str(exc)})
        raise
