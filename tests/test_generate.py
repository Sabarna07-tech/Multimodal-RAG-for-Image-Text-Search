from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ml import generate


class DummyModel:
    def __init__(self, text: str):
        self._text = text

    def generate_content(self, payload):
        return SimpleNamespace(text=self._text)


def test_generate_low_confidence(monkeypatch):
    items = [
        {"modality": "text", "score": 0.1, "metadata": {"doc_id": "doc1"}, "text": "sample"}
    ]
    result = generate.generate_response("question", items)
    assert "insufficient" in result["response"].lower()


def test_generate_with_model(monkeypatch):
    items = [
        {"modality": "text", "score": 1.0, "combined_score": 1.0, "metadata": {"doc_id": "doc1"}, "text": "fact"}
    ]

    monkeypatch.setattr(generate, "_ensure_model", lambda: DummyModel("answer text"))
    result = generate.generate_response("question", items)
    assert result["response"] == "answer text"
    assert result["citations"]["text"]

