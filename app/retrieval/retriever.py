from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.embedding.embedder import Embedder

_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder

            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _CROSS_ENCODER = False
    return _CROSS_ENCODER


class Retriever:
    """Retrieves top-K text and image matches from ChromaDB."""

    def __init__(self, embedder: Embedder, text_collection, image_collection) -> None:
        self.embedder = embedder
        self.text_collection = text_collection
        self.image_collection = image_collection

    def _rerank(self, query: str, results: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        if not results or not results.get("documents"):
            return results

        documents = results["documents"][0]
        if not documents:
            return results

        ce = _get_cross_encoder()
        if not ce:
            return results

        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        pairs = [(query, doc) for doc in documents]
        scores = ce.predict(pairs)
        order = np.argsort(-scores)[:top_k]
        return {
            "documents": [[documents[i] for i in order]],
            "metadatas": [[metadatas[i] for i in order]],
            "ids": [[ids[i] for i in order]],
            "scores": [scores[order].tolist()],
        }

    def retrieve(
        self,
        query: str,
        n_results: int = 4,
        rerank: bool = True,
        rerank_top_k: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text_embeddings = self.embedder.embed_text([query])
        text_results: Dict[str, Any] = {"documents": [[]], "metadatas": [[]], "ids": [[]]}
        if len(text_embeddings):
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [text_embeddings[0].tolist()],
                "n_results": max(n_results, rerank_top_k if rerank else n_results),
            }
            if where:
                query_kwargs["where"] = where
            try:
                raw = self.text_collection.query(**query_kwargs)
                text_results = self._rerank(query, raw, n_results) if rerank else raw
            except Exception:
                pass

        image_embeddings = self.embedder.embed_text_for_images([query])
        image_results: Dict[str, Any] = {"documents": [[]], "metadatas": [[]], "ids": [[]]}
        if len(image_embeddings):
            image_kwargs: Dict[str, Any] = {
                "query_embeddings": [image_embeddings[0].tolist()],
                "n_results": n_results,
            }
            if where:
                image_kwargs["where"] = where
            try:
                image_results = self.image_collection.query(**image_kwargs)
            except Exception:
                pass

        return {"text": text_results, "image": image_results}
