from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from app.settings import settings

_TEXT_MODEL: Optional[SentenceTransformer] = None
_CLIP_MODEL: Optional[CLIPModel] = None
_CLIP_PROCESSOR: Optional[CLIPProcessor] = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_text_model() -> SentenceTransformer:
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        _TEXT_MODEL = SentenceTransformer(settings.models.text)
        _TEXT_MODEL.to(_device())
    return _TEXT_MODEL


def _ensure_clip() -> CLIPModel:
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        _CLIP_MODEL = CLIPModel.from_pretrained(settings.models.clip)
        _CLIP_MODEL.to(_device())
    return _CLIP_MODEL


def _ensure_processor() -> CLIPProcessor:
    global _CLIP_PROCESSOR
    if _CLIP_PROCESSOR is None:
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(settings.models.clip)
    return _CLIP_PROCESSOR


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


@torch.no_grad()
def embed_text_batch(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed text strings using MiniLM and return L2-normalized embeddings.

    # [MIGRATE] Shared embedding helpers replace ad-hoc model usage.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = _ensure_text_model()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_tensor=True,
        device=_device(),
        show_progress_bar=False,
    )
    array = embeddings.detach().cpu().float().numpy()
    return _normalize(array)


@torch.no_grad()
def embed_images_batch(paths: Sequence[Path], batch_size: int = 8) -> np.ndarray:
    """Embed images using CLIP's vision tower and return normalized vectors."""
    if not paths:
        return np.empty((0, 512), dtype=np.float32)
    model = _ensure_clip()
    processor = _ensure_processor()

    embeddings: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(_device())
        vision_feats = model.get_image_features(**inputs)
        for img in images:
            img.close()
        embeddings.append(vision_feats.detach().cpu().float().numpy())
    stacked = np.vstack(embeddings)
    return _normalize(stacked)


@torch.no_grad()
def embed_query_for_images(query: str) -> np.ndarray:
    """Encode a text query into the CLIP text space for image retrieval."""
    if not query.strip():
        return np.zeros((512,), dtype=np.float32)
    model = _ensure_clip()
    processor = _ensure_processor()
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(_device())
    text_features = model.get_text_features(**inputs)
    array = text_features.detach().cpu().float().numpy()
    normalized = _normalize(array)[0]
    return normalized


__all__ = ["embed_text_batch", "embed_images_batch", "embed_query_for_images"]

