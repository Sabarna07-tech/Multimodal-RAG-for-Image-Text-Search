from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from app.ml import index_build


class _DummyStore:
    def __init__(self) -> None:
        self.text_rows: List = []
        self.image_rows: List = []

    def upsert_text_vectors(self, rows) -> None:
        self.text_rows.extend(rows)

    def upsert_image_vectors(self, rows) -> None:
        self.image_rows.extend(rows)


def test_index_text_nodes(monkeypatch, tmp_path: Path) -> None:
    store = _DummyStore()
    monkeypatch.setattr(index_build, "_LANCEDB_STORE", store)
    monkeypatch.setattr(index_build, "_VERSION_FILE", tmp_path / "versions.json")
    monkeypatch.setattr(index_build, "embed_text_batch", lambda texts: np.ones((len(texts), 384), dtype=np.float32))

    nodes = [
        {
            "id": "doc-1",
            "text": "This is a short document for testing purposes.",
            "metadata": {"source": "pdf"},
        }
    ]

    indexed = index_build.index_text_nodes("user-1", nodes)

    assert indexed
    assert store.text_rows
    assert (tmp_path / "versions.json").exists()


def test_index_image_nodes(monkeypatch, tmp_path: Path) -> None:
    store = _DummyStore()
    monkeypatch.setattr(index_build, "_LANCEDB_STORE", store)
    monkeypatch.setattr(index_build, "_VERSION_FILE", tmp_path / "versions.json")
    monkeypatch.setattr(index_build, "embed_images_batch", lambda paths: np.ones((len(list(paths)), 512), dtype=np.float32))

    tmp_file = tmp_path / "image.png"
    tmp_file.write_bytes(b"fake")

    nodes = [
        {
            "id": "img-1",
            "metadata": {
                "file_path": str(tmp_file),
                "doc_id": "doc-1",
                "source": "youtube",
            },
        }
    ]

    indexed = index_build.index_image_nodes("user-1", nodes)

    assert indexed
    assert store.image_rows
