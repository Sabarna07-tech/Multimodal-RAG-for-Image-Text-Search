from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, validator


class Document(BaseModel):
    """Metadata describing an ingested document-level asset."""

    id: str
    user_id: str
    source_type: str  # Expected values: pdf | youtube
    source_uri: str
    title: Optional[str] = None
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("source_type")
    def _validate_source_type(cls, value: str) -> str:
        allowed = {"pdf", "youtube"}
        if value not in allowed:
            raise ValueError(f"source_type must be one of {allowed}, got {value!r}")
        return value


class Chunk(BaseModel):
    """Represents a retrievable chunk associated with a document."""

    id: str
    document_id: str
    modality: str  # Expected values: text | image
    text: Optional[str] = None
    page_no: Optional[int] = None
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    file_path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("modality")
    def _validate_modality(cls, value: str) -> str:
        allowed = {"text", "image"}
        if value not in allowed:
            raise ValueError(f"modality must be one of {allowed}, got {value!r}")
        return value


class Metadata(BaseModel):
    """Envelope containing a document and its associated chunks."""

    document: Document
    chunks: List[Chunk] = Field(default_factory=list)


class MetadataStore:
    """SQLite-backed metadata catalog for LanceDB vectors."""

    def __init__(self, db_path: str) -> None:
        # [MIGRATE] Centralized metadata store for document/chunk records.
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _cursor(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    def _ensure_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    text TEXT,
                    page_no INTEGER,
                    start_ts REAL,
                    end_ts REAL,
                    file_path TEXT,
                    meta TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                """
            )

    # ---------------------
    # Document CRUD
    # ---------------------
    def upsert_document(self, document: Document) -> Document:
        payload = document.dict()
        payload["created_at"] = document.created_at.isoformat()
        payload["updated_at"] = datetime.utcnow().isoformat()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, user_id, source_type, source_uri, title, status, created_at, updated_at)
                VALUES (:id, :user_id, :source_type, :source_uri, :title, :status, :created_at, :updated_at)
                ON CONFLICT(id) DO UPDATE SET
                    user_id=excluded.user_id,
                    source_type=excluded.source_type,
                    source_uri=excluded.source_uri,
                    title=excluded.title,
                    status=excluded.status,
                    updated_at=excluded.updated_at;
                """,
                payload,
            )
        return self.get_document(document.id)

    def get_document(self, document_id: str) -> Optional[Document]:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cur.fetchone()
        finally:
            cur.close()
        if not row:
            return None
        return Document(**dict(row))

    def list_documents(self, user_id: Optional[str] = None) -> List[Document]:
        cur = self._conn.cursor()
        try:
            if user_id:
                cur.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
            else:
                cur.execute("SELECT * FROM documents ORDER BY updated_at DESC")
            rows = cur.fetchall()
        finally:
            cur.close()
        return [Document(**dict(row)) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    # ---------------------
    # Chunk CRUD
    # ---------------------
    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        now = datetime.utcnow().isoformat()
        with self._cursor() as cur:
            for chunk in chunks:
                payload = chunk.dict()
                payload["created_at"] = chunk.created_at.isoformat()
                payload["updated_at"] = now
                payload["meta"] = json.dumps(payload.get("meta", {}))
                cur.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, modality, text, page_no, start_ts, end_ts, file_path, meta, created_at, updated_at
                    ) VALUES (
                        :id, :document_id, :modality, :text, :page_no, :start_ts, :end_ts, :file_path, :meta, :created_at, :updated_at
                    )
                    ON CONFLICT(id) DO UPDATE SET
                        document_id=excluded.document_id,
                        modality=excluded.modality,
                        text=excluded.text,
                        page_no=excluded.page_no,
                        start_ts=excluded.start_ts,
                        end_ts=excluded.end_ts,
                        file_path=excluded.file_path,
                        meta=excluded.meta,
                        updated_at=excluded.updated_at;
                    """,
                    payload,
                )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = cur.fetchone()
        finally:
            cur.close()
        if not row:
            return None
        data = dict(row)
        data["meta"] = json.loads(data.get("meta") or "{}")
        return Chunk(**data)

    def list_chunks(self, document_id: str) -> List[Chunk]:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY created_at", (document_id,))
            rows = cur.fetchall()
        finally:
            cur.close()
        results: List[Chunk] = []
        for row in rows:
            data = dict(row)
            data["meta"] = json.loads(data.get("meta") or "{}")
            results.append(Chunk(**data))
        return results

    def delete_chunk(self, chunk_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

    def delete_chunks_for_document(self, document_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    # ---------------------
    # Aggregations
    # ---------------------
    def get_metadata(self, document_id: str) -> Optional[Metadata]:
        document = self.get_document(document_id)
        if not document:
            return None
        chunks = self.list_chunks(document_id)
        return Metadata(document=document, chunks=chunks)
