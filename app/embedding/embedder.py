from __future__ import annotations

from typing import List

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class Embedder:
    """Encapsulates text + image encoders used across the service."""

    def __init__(self) -> None:
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.image_model = SentenceTransformer("clip-ViT-B-32")
        self.clip_text_model = SentenceTransformer("clip-ViT-B-32")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model.to(device)
        self.image_model.to(device)
        self.clip_text_model.to(device)

    def embed_text(self, text_list: List[str]):
        if not text_list:
            return []
        embeddings = self.text_model.encode(text_list, convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def embed_text_for_images(self, text_list: List[str]):
        if not text_list:
            return []
        embeddings = self.clip_text_model.encode(text_list, convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def embed_images(self, image_paths: List[str]):
        if not image_paths:
            return []
        pil_images = []
        for path in image_paths:
            with Image.open(path) as img:
                pil_images.append(img.convert("RGB"))
        embeddings = self.image_model.encode(pil_images, convert_to_tensor=True)
        return embeddings.cpu().numpy()