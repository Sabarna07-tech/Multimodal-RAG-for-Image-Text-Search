import io
import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# Configure environment before importing the app module
TMP_DIR = tempfile.mkdtemp(prefix="mrag-tests-")
os.environ["GOOGLE_API_KEY"] = "dummy-test-api-key"
os.environ["API_KEYS"] = json.dumps({"test-key-user-1": "user-1"})
os.environ["CHROMA_DB_PATH"] = os.path.join(TMP_DIR, "chroma")
os.environ["CHECKPOINT_DIR"] = os.path.join(TMP_DIR, "checkpoints")
os.environ["INGEST_CACHE_DIR"] = os.path.join(TMP_DIR, "cache")
os.environ["NOTES_DB_PATH"] = os.path.join(TMP_DIR, "notes.db")

from app import main as app_main  # noqa: E402


class DummyEmbedder:
    def embed_text(self, items):
        import numpy as np
        return np.ones((len(items), 3))

    def embed_text_for_images(self, items):
        import numpy as np
        return np.ones((len(items), 3))

    def embed_images(self, items):
        import numpy as np
        return np.ones((len(items), 3))


class DummyGenerator:
    def __init__(self):
        self.model = object()

    def generate_answer(self, query, text, images):
        return f"answer for {query}"


class DummyRedis:
    def __init__(self):
        self.storage = {}

    def get(self, key):
        return self.storage.get(key)

    def setex(self, key, ttl, value):
        self.storage[key] = value


@pytest.fixture(scope="module")
def client(monkeypatch):
    monkeypatch.setattr(app_main, "_embedder", DummyEmbedder())
    monkeypatch.setattr(app_main, "_generator", DummyGenerator())
    monkeypatch.setattr(app_main, "_redis_client", lambda: DummyRedis())
    app_main._note_store = app_main.NoteStore(app_main.settings.NOTES_DB_PATH)

    def _fake_delay(user_id, url):
        return SimpleNamespace(id="job-123")

    def _fake_async_result(job_id):
        return SimpleNamespace(state="SUCCESS", result={"text_chunks_indexed": 5, "images_indexed": 1})

    monkeypatch.setattr(app_main.ingest_youtube_task, "delay", _fake_delay)
    monkeypatch.setattr(app_main.celery_app, "AsyncResult", _fake_async_result)

    # Seed a video record for note-related tests
    app_main._note_store.upsert(
        "user-1",
        "vid-1",
        {
            "video_id": "vid-1",
            "video_url": "https://youtu.be/demo",
            "title": "Demo Lecture",
            "duration": 1200,
            "summary": "Summary",
            "key_points": ["Point A", "Point B"],
            "timeline": [{"timestamp": 0, "note": "Intro"}],
            "quiz": [{"question": "Q1", "answer": "A1"}],
        },
    )

    return TestClient(app_main.app)


def teardown_module(module):  # noqa: D401
    """Remove temporary directories created for the tests."""
    shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_process_pdf(client, monkeypatch):
    sample_pages = [
        {"page": 1, "text": "This is a test page."},
        {"page": 2, "text": "Another page for testing."},
    ]
    sample_images = [os.path.join(TMP_DIR, "img1.jpg")]
    os.makedirs(os.path.dirname(sample_images[0]), exist_ok=True)
    with open(sample_images[0], "wb") as handle:
        handle.write(b"fake-image-bytes")

    monkeypatch.setattr(app_main, "extract_pdf_data", lambda path: (sample_pages, sample_images))

    response = client.post(
        "/process-pdf/",
        headers={"X-API-Key": "test-key-user-1"},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["text_chunks_indexed"] > 0


def test_video_library_endpoints(client):
    headers = {"X-API-Key": "test-key-user-1"}
    listing = client.get("/videos/", headers=headers)
    assert listing.status_code == 200
    videos = listing.json()["videos"]
    assert videos and videos[0]["video_id"] == "vid-1"

    notes = client.get("/videos/vid-1/notes", headers=headers)
    assert notes.status_code == 200
    assert notes.json()["title"] == "Demo Lecture"

    quiz = client.get("/videos/vid-1/quiz", headers=headers)
    assert quiz.status_code == 200
    assert quiz.json()["quiz"][0]["answer"] == "A1"


def test_chat_returns_answer(client, monkeypatch):
    expected = {
        "text": {"documents": [["doc1"]], "metadatas": [[{"source": "pdf"}]]},
        "image": {"documents": [[]], "metadatas": [[]]},
    }
    monkeypatch.setattr(app_main.Retriever, "retrieve", lambda self, **_: expected)
    response = client.post(
        "/chat/",
        headers={"X-API-Key": "test-key-user-1"},
        json={"thread_id": "t1", "message": "hello", "video_id": "vid-1"},
    )
    assert response.status_code == 200
    assert response.json()["response"].startswith("answer for")


def test_process_youtube_enqueues_job(client):
    response = client.post(
        "/process-youtube/",
        headers={"X-API-Key": "test-key-user-1"},
        data={"url": "https://youtu.be/dummy"},
    )
    assert response.status_code == 202
    assert response.json()["job_id"] == "job-123"

    status = client.get(
        "/ingest/status/job-123",
        headers={"X-API-Key": "test-key-user-1"},
    )
    assert status.status_code == 200
    assert status.json()["state"] == "SUCCESS"
