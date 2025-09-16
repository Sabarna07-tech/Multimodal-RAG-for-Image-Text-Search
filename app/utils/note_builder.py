from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.generation.generator import Generator


def _limit_transcript(transcript: List[Dict[str, Any]], limit_chars: int) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    total = 0
    for segment in transcript:
        text = segment.get("text", "").strip()
        if not text:
            continue
        piece = {"start": int(segment.get("start", 0)), "text": text}
        total += len(text)
        if total > limit_chars:
            break
        collected.append(piece)
    return collected


def _fallback_notes(transcript: List[Dict[str, Any]], question_count: int) -> Dict[str, Any]:
    if not transcript:
        return {
            "summary": "No transcript available for this video.",
            "key_points": [],
            "timeline": [],
            "quiz": [],
        }
    key_segments = transcript[: min(5, len(transcript))]
    summary_text = " ".join(seg.get("text", "") for seg in key_segments)
    timeline = [
        {"timestamp": int(seg.get("start", 0)), "note": seg.get("text", "")}
        for seg in transcript[: min(10, len(transcript))]
    ]
    quiz = [
        {
            "question": f"What was discussed around {int(seg.get('start', 0))} seconds?",
            "answer": seg.get("text", ""),
        }
        for seg in key_segments[:question_count]
    ]
    return {
        "summary": summary_text,
        "key_points": [seg.get("text", "") for seg in key_segments],
        "timeline": timeline,
        "quiz": quiz,
    }


def build_notes_payload(
    transcript: List[Dict[str, Any]],
    video_info: Optional[Dict[str, Any]],
    video_id: str,
    video_url: str,
    generator: Optional[Generator],
) -> Dict[str, Any]:
    limited = _limit_transcript(transcript, settings.NOTE_CONTEXT_CHARS)
    base = _fallback_notes(limited, settings.QUIZ_QUESTION_COUNT)

    if generator and getattr(generator, "model", None) and limited:
        context_lines = [f"[{seg['start']}s] {seg['text']}" for seg in limited]
        prompt = (
            "You are creating study notes for a learner. Given the transcript excerpts below, "
            "produce JSON with fields: summary (string), key_points (list of 5 short bullet strings), "
            "timeline (list of objects with timestamp (int seconds) and note (string)), and quiz (list of "
            f"{settings.QUIZ_QUESTION_COUNT} question objects with question and answer fields). If you cannot comply, "
            "return an empty JSON object.\n\nTranscript:\n" + "\n".join(context_lines)
        )
        try:
            response = generator.model.generate_content(prompt)
            candidate = json.loads(response.text or "{}")
            if isinstance(candidate, dict) and candidate:
                for key in ("summary", "key_points", "timeline", "quiz"):
                    if key not in candidate:
                        raise ValueError("missing keys")
                base = candidate
        except Exception:
            pass

    return {
        "video_id": video_id,
        "video_url": video_url,
        "title": video_info.get("title") if video_info else None,
        "duration": video_info.get("duration") if video_info else None,
        "summary": base.get("summary", ""),
        "key_points": base.get("key_points", []),
        "timeline": base.get("timeline", []),
        "quiz": base.get("quiz", []),
    }
