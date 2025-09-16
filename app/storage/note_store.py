from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS video_notes (
    user_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    title TEXT,
    video_url TEXT,
    duration INTEGER,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, video_id)
);
"""


class NoteStore:
    """SQLite-backed storage for generated notes and quizzes."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def upsert(self, user_id: str, video_id: str, data: Dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        payload = json.dumps(data)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO video_notes (user_id, video_id, title, video_url, duration, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, video_id) DO UPDATE SET
                    title=excluded.title,
                    video_url=excluded.video_url,
                    duration=excluded.duration,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    video_id,
                    data.get("title"),
                    data.get("video_url"),
                    data.get("duration"),
                    payload,
                    now,
                    now,
                ),
            )
            conn.commit()

    def list_videos(self, user_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT video_id, title, duration, updated_at FROM video_notes WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                "video_id": video_id,
                "title": title,
                "duration": duration,
                "updated_at": updated_at,
            }
            for video_id, title, duration, updated_at in rows
        ]

    def get_video(self, user_id: str, video_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT title, video_url, duration, payload, created_at, updated_at FROM video_notes WHERE user_id = ? AND video_id = ?",
                (user_id, video_id),
            ).fetchone()
        if not row:
            return None
        title, video_url, duration, payload, created_at, updated_at = row
        data = json.loads(payload)
        data.update(
            {
                "video_id": video_id,
                "title": title,
                "video_url": video_url,
                "duration": duration,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        return data

    def get_quiz(self, user_id: str, video_id: str) -> Optional[Dict[str, Any]]:
        record = self.get_video(user_id, video_id)
        if not record:
            return None
        return {
            "video_id": video_id,
            "title": record.get("title"),
            "quiz": record.get("quiz", []),
        }
