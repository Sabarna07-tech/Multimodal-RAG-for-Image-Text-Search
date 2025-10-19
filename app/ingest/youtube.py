from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yt_dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

from app.settings import settings

_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_\-]{6,})")


class YouTubeIngestError(RuntimeError):
    """Raised when YouTube ingestion fails after retries."""


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float


def _extract_video_id(url: str) -> Optional[str]:
    match = _YT_ID_RE.search(url)
    return match.group(1) if match else None


def resolve_video_id(url: str) -> Optional[str]:
    """Public helper to extract a YouTube video ID from a URL."""
    return _extract_video_id(url)


def download_video(
    url: str,
    output_dir: Optional[Path] = None,
    *,
    max_retries: int = 3,
    backoff_sec: float = 2.0,
) -> Path:
    """
    Download a YouTube video via yt-dlp and return the local path.

    # [MIGRATE] Transcript-first ingestion begins with a resilient yt-dlp download.
    """
    target_dir = Path(output_dir or Path(settings.paths.ingest_cache_dir) / "youtube_media")
    target_dir.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            ydl_opts: Dict[str, object] = {
                "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
                "format": settings.youtube.download_format,
                "quiet": True,
                "no_warnings": True,
                "retries": settings.youtube.retries,
                "socket_timeout": settings.youtube.timeout_sec,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = Path(ydl.prepare_filename(info))
                if not file_path.exists():
                    raise FileNotFoundError(f"yt-dlp did not produce output for {url}")
                return file_path
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(backoff_sec * attempt)
    raise YouTubeIngestError(f"Failed to download YouTube video after {max_retries} attempts: {last_error}")


def get_transcript(video_id: str, languages: Sequence[str] = ("en", "en-US")) -> Optional[List[TranscriptSegment]]:
    """
    Fetch a transcript for the supplied video ID.

    Returns None when transcripts are disabled or unavailable.
    """
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        preferred = None
        for lang in languages:
            try:
                preferred = transcripts.find_transcript([lang])
                if preferred:
                    break
            except Exception:
                continue
        if not preferred:
            preferred = next(iter(transcripts), None)
        if not preferred:
            return None
        entries = preferred.fetch()
        segments: List[TranscriptSegment] = []
        for entry in entries:
            text = entry.get("text", "").strip()
            if not text:
                continue
            start = float(entry.get("start", 0.0))
            duration = float(entry.get("duration", 0.0))
            segments.append(TranscriptSegment(text=text, start=start, end=start + duration))
        return segments or None
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:  # pragma: no cover - API fallback path
        return None


def _extract_audio(video_path: Path) -> Path:
    audio_dir = video_path.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{video_path.stem}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=settings.youtube.timeout_sec)
    except subprocess.SubprocessError as exc:
        raise YouTubeIngestError(f"Failed to extract audio from {video_path}") from exc
    return audio_path


def _load_whisper_model() -> Tuple[Optional[object], Optional[object]]:
    model = None
    align_model = None
    try:
        import whisperx  # type: ignore

        model = whisperx.load_model("base", device="cuda" if _cuda_available() else "cpu")
        align_model = whisperx.load_align_model(language_code="en", device=model.device)
        return model, align_model
    except Exception:
        try:
            import whisper  # type: ignore

            model = whisper.load_model("base", device="cuda" if _cuda_available() else "cpu")
            return model, None
        except Exception:
            return None, None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _transcribe_with_whisper(audio_path: Path) -> List[TranscriptSegment]:
    model, align_model = _load_whisper_model()
    if model is None:
        raise YouTubeIngestError(
            "Transcript unavailable and Whisper/WhisperX not installed. Install whisper or whisperx to enable fallback."
        )

    try:
        if align_model:
            import whisperx  # type: ignore

            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=8)
            aligned = whisperx.align(result["segments"], align_model, audio, model.device)
            segments = aligned["segments"]
        else:
            import whisper  # type: ignore

            result = model.transcribe(str(audio_path))
            segments = result.get("segments", [])
    except Exception as exc:  # pragma: no cover - inference failure path
        raise YouTubeIngestError(f"Whisper transcription failed for {audio_path}") from exc

    transcript_segments: List[TranscriptSegment] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        transcript_segments.append(TranscriptSegment(text=text, start=start, end=end))
    return transcript_segments


def transcript_or_fallback(url: str) -> Tuple[Optional[List[TranscriptSegment]], Optional[Path], Dict[str, Any]]:
    """
    Attempt transcript-first ingestion and fall back to Whisper when unavailable.

    Returns (transcript_segments, audio_path, info_dict)
    """
    info: Dict[str, Any] = {}
    video_id = _extract_video_id(url)
    transcript = None
    if video_id:
        transcript = get_transcript(video_id)

    video_path: Optional[Path] = None
    if transcript is None:
        video_path = download_video(url)
        transcript = get_transcript(video_id) if video_id else None

    if transcript is not None:
        if video_path is not None:
            info["video_path"] = str(video_path)
        return transcript, None, info

    if video_path is None:
        video_path = download_video(url)

    audio_path = _extract_audio(video_path)
    transcript = _transcribe_with_whisper(audio_path)
    info["audio_path"] = str(audio_path)
    info["video_path"] = str(video_path)
    return transcript, audio_path, info


__all__ = [
    "TranscriptSegment",
    "YouTubeIngestError",
    "download_video",
    "get_transcript",
    "resolve_video_id",
    "transcript_or_fallback",
]
