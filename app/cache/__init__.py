"""
Lightweight in-memory caches for query embeddings, retrieval results, and chat responses.

# [MIGRATE] Introduced to reduce duplicate embedding work and reuse retrieval outputs.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from app.ml.index_build import get_index_version

EMBED_TTL_SEC = 300
RETRIEVAL_TTL_SEC = 120
CHAT_TTL_SEC = 60

_EMBED_CACHE: Dict[str, Tuple[float, Tuple[np.ndarray, np.ndarray]]] = {}
_RETRIEVAL_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}
_CHAT_CACHE: Dict[Tuple[str, str, int], Tuple[float, Any]] = {}


def _now() -> float:
    return time.time()


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def clear_all_caches() -> None:
    """Utility for tests to clear cache state."""
    _EMBED_CACHE.clear()
    _RETRIEVAL_CACHE.clear()
    _CHAT_CACHE.clear()


# ---------------------------------------------------------------------------
# Query embedding cache
# ---------------------------------------------------------------------------
def get_query_embeddings(query: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    normalized = _normalize_query(query)
    entry = _EMBED_CACHE.get(normalized)
    if not entry:
        return None
    expires_at, payload = entry
    if expires_at < _now():
        _EMBED_CACHE.pop(normalized, None)
        return None
    return payload


def set_query_embeddings(query: str, text_vec: np.ndarray, image_vec: np.ndarray, ttl: int = EMBED_TTL_SEC) -> None:
    normalized = _normalize_query(query)
    _EMBED_CACHE[normalized] = (_now() + ttl, (text_vec, image_vec))


# ---------------------------------------------------------------------------
# Retrieval cache
# ---------------------------------------------------------------------------
def get_retrieval_results(user_id: str, query: str, index_version: int) -> Optional[Any]:
    normalized = _normalize_query(query)
    key = (user_id, normalized, index_version)
    entry = _RETRIEVAL_CACHE.get(key)
    if not entry:
        return None
    expires_at, payload = entry
    if expires_at < _now():
        _RETRIEVAL_CACHE.pop(key, None)
        return None
    return payload


def set_retrieval_results(user_id: str, query: str, index_version: int, results: Any, ttl: int = RETRIEVAL_TTL_SEC) -> None:
    normalized = _normalize_query(query)
    key = (user_id, normalized, index_version)
    _RETRIEVAL_CACHE[key] = (_now() + ttl, results)


# ---------------------------------------------------------------------------
# Chat response cache decorator
# ---------------------------------------------------------------------------
def chat_cache(ttl: int = CHAT_TTL_SEC) -> Callable:
    """
    Decorator caching chat responses keyed by user/query/index_version.

    The wrapped function must accept ``user_id`` and ``query`` as its first two arguments.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(user_id: str, query: str, *args, **kwargs):
            version = get_index_version(user_id)
            normalized = _normalize_query(query)
            extra = tuple(sorted(kwargs.items())) if kwargs else ()
            key = (user_id, normalized, version, extra)
            entry = _CHAT_CACHE.get(key)
            if entry:
                expires_at, payload = entry
                if expires_at >= _now():
                    return payload
            result = func(user_id, query, *args, **kwargs)
            _CHAT_CACHE[key] = (_now() + ttl, result)
            return result

        return wrapper

    return decorator


__all__ = [
    "get_query_embeddings",
    "set_query_embeddings",
    "get_retrieval_results",
    "set_retrieval_results",
    "chat_cache",
    "clear_all_caches",
]
