from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.cache import (
    get_query_embeddings,
    set_query_embeddings,
    get_retrieval_results,
    set_retrieval_results,
)
from app.ml.embeddings import embed_query_for_images, embed_text_batch
from app.ml.index_build import get_index_version
from app.settings import settings
from app.storage.lancedb_store import LanceDBStore
from app.storage.schema import MetadataStore, Chunk

_CROSS_ENCODER = None
_LANCEDB_STORE = LanceDBStore(settings.paths.lancedb_dir)
_METADATA_STORE = MetadataStore(os.path.join(settings.paths.lancedb_dir, "metadata.sqlite3"))


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder

            _CROSS_ENCODER = CrossEncoder(settings.models.reranker)
        except Exception:
            _CROSS_ENCODER = False
    return _CROSS_ENCODER


def retrieve_text(user_id: str, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.retrieval.index_topk_text
    version = get_index_version(user_id)

    cached = get_retrieval_results(user_id, f"text::{query}", version)
    if cached is not None:
        return cached

    text_vec, _ = _get_embeddings(query)
    if text_vec.size == 0:
        return []

    raw = _LANCEDB_STORE.search_text(user_id, text_vec.tolist(), top_k)
    results: List[Dict[str, Any]] = []
    for entry in raw:
        chunk = _METADATA_STORE.get_chunk(entry["chunk_id"])
        if not chunk or not chunk.text:
            continue
        results.append(
            {
                "chunk_id": chunk.id,
                "modality": "text",
                "score": float(entry["score"]),
                "metadata": _prepare_metadata(chunk),
                "text": chunk.text,
            }
        )
    set_retrieval_results(user_id, f"text::{query}", version, results)
    return results


def retrieve_images(user_id: str, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.retrieval.index_topk_image
    version = get_index_version(user_id)

    cached = get_retrieval_results(user_id, f"image::{query}", version)
    if cached is not None:
        return cached

    _, image_vec = _get_embeddings(query)
    if image_vec.size == 0:
        return []

    raw = _LANCEDB_STORE.search_image(user_id, image_vec.tolist(), top_k)
    results: List[Dict[str, Any]] = []
    for entry in raw:
        chunk = _METADATA_STORE.get_chunk(entry["chunk_id"])
        if not chunk:
            continue
        results.append(
            {
                "chunk_id": chunk.id,
                "modality": "image",
                "score": float(entry["score"]),
                "metadata": _prepare_metadata(chunk),
                "text": None,
            }
        )
    set_retrieval_results(user_id, f"image::{query}", version, results)
    return results


def retrieve(user_id: str, query: str) -> List[Dict[str, Any]]:
    version = get_index_version(user_id)
    normalized = _normalize_query(query)
    cached = get_retrieval_results(user_id, normalized, version)
    if cached is not None:
        return cached

    text_results = retrieve_text(user_id, query)
    image_results = retrieve_images(user_id, query)

    reranked = _rerank_text(query, text_results)
    fused = _fuse_results(reranked, image_results)

    set_retrieval_results(user_id, normalized, version, fused)
    return fused


def _get_embeddings(query: str) -> Tuple[np.ndarray, np.ndarray]:
    cached = get_query_embeddings(query)
    if cached:
        return cached
    text_vec = embed_text_batch([query])
    image_vec = embed_query_for_images(query)
    set_query_embeddings(query, text_vec[0] if text_vec.size else np.zeros(384, dtype=np.float32), image_vec)
    if text_vec.size == 0:
        text_vec = np.zeros((1, 384), dtype=np.float32)
    return text_vec[0], image_vec


def _rerank_text(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results or not settings.retrieval.use_rerank:
        return results

    cross_encoder = _get_cross_encoder()
    if not cross_encoder:
        return results

    top_candidates = results[: settings.retrieval.rerank_topk]
    if not top_candidates:
        return results

    pairs = [(query, item["text"]) for item in top_candidates if item.get("text")]
    if not pairs:
        return results

    scores = cross_encoder.predict(pairs)
    for item, score in zip(top_candidates, scores):
        item["rerank_score"] = float(score)

    # Merge rerank scores back while preserving existing ordering for the rest.
    reranked = top_candidates + results[len(top_candidates) :]
    reranked.sort(key=lambda item: item.get("rerank_score", item["score"]), reverse=True)
    return reranked


def _fuse_results(text_results: List[Dict[str, Any]], image_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    text_cos_scores = [item["score"] for item in text_results]
    text_rerank_scores = [item.get("rerank_score") for item in text_results if "rerank_score" in item]
    image_cos_scores = [item["score"] for item in image_results]

    text_cos_z = _z_scores(text_cos_scores)
    text_rerank_z = _z_scores(text_rerank_scores) if text_rerank_scores else []
    image_cos_z = _z_scores(image_cos_scores)

    for idx, item in enumerate(text_results):
        z_vals: List[float] = []
        if text_cos_z:
            z_vals.append(text_cos_z[idx])
        if text_rerank_z and idx < len(text_rerank_z):
            z_vals.append(text_rerank_z[idx])
        combined = float(np.mean(z_vals)) if z_vals else item["score"]
        items.append({**item, "combined_score": combined})

    for idx, item in enumerate(image_results):
        z_val = image_cos_z[idx] if image_cos_z else item["score"]
        items.append({**item, "combined_score": float(z_val)})

    items.sort(key=lambda entry: entry["combined_score"], reverse=True)
    return items[: settings.retrieval.final_n]


def _z_scores(values: Sequence[Optional[float]]) -> List[float]:
    numeric = [v for v in values if v is not None]
    if not numeric:
        return []
    arr = np.array(numeric, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0:
        return [0.0 for _ in values]
    return [float((v - mean) / std) if v is not None else 0.0 for v in values]


def _prepare_metadata(chunk: Chunk) -> Dict[str, Any]:
    meta = dict(chunk.meta or {})
    meta.setdefault("doc_id", chunk.document_id)
    meta.setdefault("modality", chunk.modality)
    meta.setdefault("page_no", chunk.page_no)
    meta.setdefault("start_ts", chunk.start_ts)
    meta.setdefault("end_ts", chunk.end_ts)
    meta.setdefault("file_path", chunk.file_path)
    return meta


__all__ = ["retrieve_text", "retrieve_images", "retrieve"]

