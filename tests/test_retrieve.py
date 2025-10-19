from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from app.cache import clear_all_caches
from app.ml import retrieve
from app.storage.schema import Chunk


class DummyStore:
    def __init__(self, text_rows: List[Dict[str, Any]], image_rows: List[Dict[str, Any]]):
        self._text = text_rows
        self._image = image_rows

    def search_text(self, user_id: str, vec: List[float], top_k: int):
        return self._text[:top_k]

    def search_image(self, user_id: str, vec: List[float], top_k: int):
        return self._image[:top_k]


class DummyMetadata:
    def __init__(self, chunks: Dict[str, Chunk]):
        self._chunks = chunks

    def get_chunk(self, chunk_id: str):
        return self._chunks.get(chunk_id)


class DummyCrossEncoder:
    def predict(self, pairs):
        return np.linspace(0.1, 0.9, len(pairs))


@pytest.fixture(autouse=True)
def clear_cache_state():
    clear_all_caches()
    yield
    clear_all_caches()


def test_retrieve_fusion(monkeypatch):
    text_rows = [
        {"chunk_id": "t1", "score": 0.8, "meta": {}},
        {"chunk_id": "t2", "score": 0.6, "meta": {}},
    ]
    image_rows = [
        {"chunk_id": "i1", "score": 0.7, "meta": {}},
    ]

    store = DummyStore(text_rows, image_rows)
    chunks = {
        "t1": Chunk(id="t1", document_id="doc1", modality="text", text="alpha", meta={}),
        "t2": Chunk(id="t2", document_id="doc2", modality="text", text="beta", meta={}),
        "i1": Chunk(id="i1", document_id="doc3", modality="image", meta={"file_path": "/tmp/img.jpg"}),
    }
    metadata = DummyMetadata(chunks)

    monkeypatch.setattr(retrieve, "_LANCEDB_STORE", store)
    monkeypatch.setattr(retrieve, "_METADATA_STORE", metadata)
    monkeypatch.setattr(retrieve, "embed_text_batch", lambda texts: np.ones((1, 384), dtype=np.float32))
    monkeypatch.setattr(retrieve, "embed_query_for_images", lambda query: np.ones(512, dtype=np.float32))
    monkeypatch.setattr(retrieve, "_get_cross_encoder", lambda: DummyCrossEncoder())
    monkeypatch.setattr(retrieve, "get_index_version", lambda user_id: 1)

    fused = retrieve.retrieve("user", "example query")
    assert fused
    assert fused[0]["chunk_id"] == "t1"
    assert fused[0]["modality"] == "text"


def test_retrieval_cache_invalidation(monkeypatch):
    store = DummyStore([], [])
    metadata = DummyMetadata({})
    monkeypatch.setattr(retrieve, "_LANCEDB_STORE", store)
    monkeypatch.setattr(retrieve, "_METADATA_STORE", metadata)
    monkeypatch.setattr(retrieve, "embed_text_batch", lambda texts: np.ones((1, 384), dtype=np.float32))
    monkeypatch.setattr(retrieve, "embed_query_for_images", lambda query: np.ones(512, dtype=np.float32))
    monkeypatch.setattr(retrieve, "_get_cross_encoder", lambda: False)

    monkeypatch.setattr(retrieve, "get_index_version", lambda user_id: 1)
    retrieve.retrieve("user", "question")

    monkeypatch.setattr(retrieve, "get_index_version", lambda user_id: 2)
    fused = retrieve.retrieve("user", "question")
    assert fused == []

