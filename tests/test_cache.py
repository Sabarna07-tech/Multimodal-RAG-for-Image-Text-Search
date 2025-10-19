from __future__ import annotations

import numpy as np

from app.cache import (
    clear_all_caches,
    get_query_embeddings,
    set_query_embeddings,
    get_retrieval_results,
    set_retrieval_results,
)


def test_query_embedding_cache():
    clear_all_caches()
    vec_t = np.ones(384, dtype=np.float32)
    vec_i = np.ones(512, dtype=np.float32)
    set_query_embeddings(" test Query ", vec_t, vec_i, ttl=1)
    cached = get_query_embeddings("test query")
    assert cached is not None


def test_retrieval_cache_version_invalidation():
    clear_all_caches()
    set_retrieval_results("user", "q", 1, [1])
    assert get_retrieval_results("user", "Q", 1) == [1]
    assert get_retrieval_results("user", "Q", 2) is None

