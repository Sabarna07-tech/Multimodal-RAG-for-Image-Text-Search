from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.ml import embeddings


class _DummyTextModel:
    def to(self, device: str):
        return self

    def encode(self, texts, batch_size=None, convert_to_tensor=None, device=None, show_progress_bar=None):
        return torch.ones((len(texts), 384), dtype=torch.float32)


class _DummyClipModel:
    def __init__(self):
        self.device = "cpu"

    def to(self, device: str):
        self.device = device
        return self

    def get_image_features(self, **inputs):
        batch = inputs["pixel_values"].shape[0]
        return torch.ones((batch, 512), dtype=torch.float32)

    def get_text_features(self, **inputs):
        batch = inputs["input_ids"].shape[0]
        return torch.ones((batch, 512), dtype=torch.float32)


class _DummyProcessor:
    def to(self, device: str):
        return self

    def __call__(self, *, images=None, text=None, return_tensors="pt", padding=None):
        class _Namespace(SimpleNamespace):
            def to(self, device: str):
                return self

        if images is not None:
            batch = len(images)
            return _Namespace(pixel_values=torch.ones((batch, 3, 224, 224), dtype=torch.float32))
        if text is not None:
            batch = len(text)
            return _Namespace(
                input_ids=torch.ones((batch, 77), dtype=torch.int64),
                attention_mask=torch.ones((batch, 77), dtype=torch.int64),
            )
        raise ValueError("Unsupported processor call")


def test_embed_text_batch_normalized(monkeypatch):
    monkeypatch.setattr(embeddings, "_TEXT_MODEL", _DummyTextModel())
    vecs = embeddings.embed_text_batch(["hello", "world"])  # type: ignore[arg-type]
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_embed_images_batch(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(embeddings, "_CLIP_MODEL", _DummyClipModel())
    monkeypatch.setattr(embeddings, "_CLIP_PROCESSOR", _DummyProcessor())

    image_paths = []
    for idx, color in enumerate([(255, 255, 255), (0, 0, 0)]):
        path = tmp_path / f"img_{idx}.png"
        array = np.full((32, 32, 3), color, dtype=np.uint8)
        Image.fromarray(array).save(path)
        image_paths.append(path)

    vecs = embeddings.embed_images_batch(image_paths)
    assert vecs.shape == (2, 512)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_embed_query_for_images(monkeypatch):
    monkeypatch.setattr(embeddings, "_CLIP_MODEL", _DummyClipModel())
    monkeypatch.setattr(embeddings, "_CLIP_PROCESSOR", _DummyProcessor())

    vec = embeddings.embed_query_for_images("test query")
    assert vec.shape == (512,)
    assert pytest.approx(np.linalg.norm(vec), rel=1e-6) == 1.0
