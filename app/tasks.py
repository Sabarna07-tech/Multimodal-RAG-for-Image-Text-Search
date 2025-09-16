import os
import uuid

from app.celery_app import celery_app
from app.core.config import settings
from app.core.redis_client import get_redis
from app.data_extraction.youtube_extractor import extract_youtube_data, _video_id
from app.embedding.embedder import Embedder
from app.generation.generator import Generator
from app.storage.note_store import NoteStore
from app.utils.note_builder import build_notes_payload
from app.utils.text_chunk import chunk_text
from app.vector_store.chroma_store import ChromaStore

_embedder = Embedder()  # worker-local (avoid GPU init on web)
_generator = Generator(settings.GOOGLE_API_KEY) if settings.GOOGLE_API_KEY else None
redis = get_redis()
note_store = NoteStore(settings.NOTES_DB_PATH)


def _already_indexed_key(user_id: str, vid: str) -> str:
    return f"yt:indexed:{user_id}:{vid}"


@celery_app.task(bind=True)
def ingest_youtube_task(self, user_id: str, url: str):
    vid = _video_id(url) or "unknown"
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
        store = ChromaStore(db_path=settings.CHROMA_DB_PATH, user_id=user_id)

        self.update_state(state="PROGRESS", meta={"stage": "extract"})
        _, transcript, frame_paths, info = extract_youtube_data(url, user_id=user_id)

        full_text = " ".join([item.get("text", "") for item in transcript]) if transcript else ""
        chunks = chunk_text(full_text, settings.CHUNK_SIZE_CHARS, settings.CHUNK_OVERLAP_CHARS)
        text_chunks_indexed = 0
        if chunks:
            vectors = _embedder.embed_text(chunks)
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [
                {
                    "source": "youtube",
                    "video_url": url,
                    "user_id": user_id,
                    "video_id": vid,
                }
                for _ in chunks
            ]
            store.text_collection.add(
                documents=chunks,
                embeddings=vectors.tolist(),
                metadatas=metadatas,
                ids=ids,
            )
            text_chunks_indexed = len(chunks)

        images_indexed = 0
        if frame_paths:
            self.update_state(state="PROGRESS", meta={"stage": "frames", "count": len(frame_paths)})
            image_vectors = _embedder.embed_images(frame_paths)
            ids = [str(uuid.uuid4()) for _ in frame_paths]
            metadatas = [
                {
                    "source": "youtube",
                    "video_url": url,
                    "image_path": path,
                    "user_id": user_id,
                    "video_id": vid,
                }
                for path in frame_paths
            ]
            store.image_collection.add(
                documents=[os.path.basename(path) for path in frame_paths],
                embeddings=[vec.tolist() for vec in image_vectors],
                metadatas=metadatas,
                ids=ids,
            )
            images_indexed = len(frame_paths)

        notes_payload = build_notes_payload(transcript, info, vid, info.get("webpage_url", url), _generator)
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
