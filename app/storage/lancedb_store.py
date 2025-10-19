from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import lancedb
import numpy as np
import pyarrow as pa


@dataclass
class VectorRow:
    """Payload used when writing vectors to LanceDB."""

    chunk_id: str
    user_id: str
    document_id: str
    modality: str
    embedding: Sequence[float]
    meta: Dict[str, Any]


class LanceDBStore:
    """Utility around two LanceDB tables (text/image) with cosine similarity search."""

    def __init__(self, db_path: str) -> None:
        # [MIGRATE] Centralized LanceDB access with shared text/image collections.
        self._db = lancedb.connect(db_path)
        self._text_table = self._ensure_table("text_collection")
        self._image_table = self._ensure_table("image_collection")

    @staticmethod
    def _schema() -> pa.Schema:
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("modality", pa.string()),
                pa.field("embedding", pa.list_(pa.float32())),
                pa.field("meta", pa.string(), nullable=True),
            ]
        )

    def _ensure_table(self, name: str):
        if name in self._db.table_names():
            table = self._db.open_table(name)
        else:
            table = self._db.create_table(name, schema=self._schema())
        try:  # optional vector index build
            table.create_index(
                metric="cosine",
                column="embedding",
                index_type="IVF_PQ",
                num_partitions=32,
                num_sub_vectors=16,
            )
        except Exception:
            pass
        return table

    @staticmethod
    def _normalize(vector: Sequence[float]) -> List[float]:
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm <= 0:
            return arr.tolist()
        return (arr / norm).tolist()

    @staticmethod
    def _prepare_rows(rows: Iterable[VectorRow]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for row in rows:
            prepared.append(
                {
                    "chunk_id": row.chunk_id,
                    "user_id": row.user_id,
                    "document_id": row.document_id,
                    "modality": row.modality,
                    "embedding": LanceDBStore._normalize(row.embedding),
                    "meta": json.dumps(row.meta or {}),
                }
            )
        return prepared

    def upsert_text_vectors(self, rows: Iterable[VectorRow]) -> None:
        payloads = self._prepare_rows(rows)
        if not payloads:
            return
        for chunk in payloads:
            self._text_table.delete(self._where_clause("chunk_id", chunk["chunk_id"]))
        self._text_table.add(payloads)

    def upsert_image_vectors(self, rows: Iterable[VectorRow]) -> None:
        payloads = self._prepare_rows(rows)
        if not payloads:
            return
        for chunk in payloads:
            self._image_table.delete(self._where_clause("chunk_id", chunk["chunk_id"]))
        self._image_table.add(payloads)

    def search_text(self, user_id: str, query_vec: Sequence[float], top_k: int) -> List[Dict[str, Any]]:
        vector = self._normalize(query_vec)
        results = (
            self._text_table.search(vector)
            .where(self._where_clause("user_id", user_id))
            .metric("cosine")
            .limit(max(top_k, 1))
            .to_list()
        )
        return self._format_results(results)

    def search_image(self, user_id: str, query_vec: Sequence[float], top_k: int) -> List[Dict[str, Any]]:
        vector = self._normalize(query_vec)
        results = (
            self._image_table.search(vector)
            .where(self._where_clause("user_id", user_id))
            .metric("cosine")
            .limit(max(top_k, 1))
            .to_list()
        )
        return self._format_results(results)

    @staticmethod
    def _format_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for row in rows:
            distance = float(row.get("_distance", 0.0))
            similarity = 1.0 - distance
            formatted.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "score": similarity,
                    "meta": json.loads(row.get("meta") or "{}"),
                }
            )
        formatted.sort(key=lambda item: item["score"], reverse=True)
        return formatted

    @staticmethod
    def _where_clause(column: str, value: str) -> str:
        safe = str(value).replace("'", "''")
        return f"{column} == '{safe}'"
