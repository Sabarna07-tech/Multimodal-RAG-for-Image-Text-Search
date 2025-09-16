import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Tuple, Optional

import cv2
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from app.core.config import settings

YTCACHE = settings.INGEST_CACHE_DIR
os.makedirs(YTCACHE, exist_ok=True)

_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_\-]{6,})")


def _video_id(url: str) -> Optional[str]:
    match = _YT_ID_RE.search(url)
    return match.group(1) if match else None


def try_transcript(url: str) -> List[dict]:
    if not settings.YT_PREFER_TRANSCRIPT:
        return []
    vid = _video_id(url)
    if not vid:
        return []
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
        try:
            transcript = transcripts.find_transcript(["en"])
        except Exception:
            transcript = next(iter(transcripts), None)
        return transcript.fetch() if transcript else []
    except (TranscriptsDisabled, NoTranscriptFound):
        return []
    except Exception:
        return []


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") and shutil.which("ffprobe")


def download_lightweight(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": settings.YT_DOWNLOAD_FORMAT,
        "quiet": True,
        "no_warnings": True,
        "retries": settings.YT_RETRIES,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
        if not video_path.lower().endswith(".mp4") and _ffmpeg_available():
            mp4_path = os.path.splitext(video_path)[0] + ".mp4"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-c", "copy", mp4_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                video_path = mp4_path
            except Exception:
                pass
        return video_path


def _duration_sec(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / max(fps, 1e-5))


def sample_frames_uniform(video_path: str, out_dir: str, stride_sec: int, max_frames: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return paths
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if total <= 0:
        cap.release()
        return paths
    step = int(max(1, fps * stride_sec))
    index = 0
    while index < total and len(paths) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = int(index / max(fps, 1e-5))
        output_path = os.path.join(out_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(output_path, frame)
        paths.append(output_path)
        index += step
    cap.release()
    return paths


def _persist_frame_paths(temp_paths: List[str], user_id: Optional[str], video_id: Optional[str]) -> List[str]:
    if not temp_paths:
        return []
    target_dir = os.path.join(YTCACHE, "frames", user_id or "anon", video_id or "unknown")
    os.makedirs(target_dir, exist_ok=True)
    persisted: List[str] = []
    for path in temp_paths:
        dest = os.path.join(target_dir, os.path.basename(path))
        shutil.move(path, dest)
        persisted.append(dest)
    return persisted


def extract_youtube_data(url: str, user_id: Optional[str] = None) -> Tuple[Optional[str], List[dict], List[str]]:
    transcript = try_transcript(url)
    need_download = (not transcript) or (not settings.YT_LAZY_FRAMES)
    video_path: Optional[str] = None
    frame_paths: List[str] = []
    vid = _video_id(url)

    with tempfile.TemporaryDirectory(dir=YTCACHE) as tmp_dir:
        if need_download:
            video_path = download_lightweight(url, tmp_dir)
            duration = _duration_sec(video_path) or 0
            if settings.YT_MAX_DURATION_MIN > 0 and duration / 60.0 > settings.YT_MAX_DURATION_MIN:
                raise RuntimeError(f"Video too long ({duration/60:.1f} min)")

            if not settings.YT_LAZY_FRAMES:
                frames_dir = os.path.join(tmp_dir, "frames")
                temp_frames = sample_frames_uniform(
                    video_path,
                    frames_dir,
                    stride_sec=settings.YT_FRAME_STRIDE_SEC,
                    max_frames=settings.YT_MAX_FRAMES,
                )
                frame_paths = _persist_frame_paths(temp_frames, user_id, vid)

    return video_path, transcript, frame_paths