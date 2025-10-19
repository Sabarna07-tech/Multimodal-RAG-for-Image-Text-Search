from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer

from app.settings import settings


class Embedder:
    """Text, image, and query embedding helper compatible with LlamaIndex."""

    # [MIGRATE] Align embedding stack with MiniLM + CLIP for the new pipeline.
    def __init__(
        self,
        text_model_name: Optional[str] = None,
        clip_model_name: Optional[str] = None,
    ) -> None:
        # [MIGRATE] Model identifiers now flow from the centralized settings module.
        text_model = text_model_name or settings.models.text
        clip_model = clip_model_name or settings.models.clip

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._text_model = SentenceTransformer(text_model)
        self._text_model.to(device)

        self._image_model = SentenceTransformer(clip_model)
        self._image_model.to(device)

        self._clip_text_model = SentenceTransformer(clip_model)
        self._clip_text_model.to(device)

        self._llama_text_embed: BaseEmbedding = HuggingFaceEmbedding(model_name=text_model)

    def llama_text_embedder(self) -> BaseEmbedding:
        """Expose the LlamaIndex-compatible text embedding model."""
        return self._llama_text_embed

    def embed_text(self, text_list: Sequence[str]) -> np.ndarray:
        """Embed free-form text queries/chunks with MiniLM."""
        if not text_list:
            return np.empty((0, 0))
        embeddings = self._text_model.encode(list(text_list), convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def embed_text_for_images(self, text_list: Sequence[str]) -> np.ndarray:
        """Embed textual prompts against the CLIP text encoder."""
        if not text_list:
            return np.empty((0, 0))
        embeddings = self._clip_text_model.encode(list(text_list), convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def embed_images(self, image_paths: Iterable[str]) -> np.ndarray:
        """Embed images via CLIP ViT-B/32."""
        pil_images: List[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as img:
                pil_images.append(img.convert("RGB"))
        if not pil_images:
            return np.empty((0, 0))
        embeddings = self._image_model.encode(pil_images, convert_to_tensor=True)
        return embeddings.cpu().numpy()
