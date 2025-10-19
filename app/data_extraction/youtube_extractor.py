import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple, Optional

import cv2
import imagehash
import yt_dlp
from PIL import Image
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from app.settings import settings

YTCACHE = settings.paths.ingest_cache_dir
os.makedirs(YTCACHE, exist_ok=True)

_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_\-]{6,})")


def _video_id(url: str) -> Optional[str]:
    match = _YT_ID_RE.search(url)
    return match.group(1) if match else None


def try_transcript(url: str) -> List[dict]:
    if not settings.youtube.prefer_transcript:
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


def download_lightweight(url: str, out_dir: str, download: bool) -> Tuple[Optional[str], Dict[str, any]]:
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": settings.youtube.download_format,
        "quiet": True,
        "no_warnings": True,
        "retries": settings.youtube.retries,
        "socket_timeout": 30,
        "skip_download": not download,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=download)
        video_path: Optional[str] = None
        if download:
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
        return video_path, info


def _duration_sec(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / max(fps, 1e-5))


def sample_frames_scene_aware(
    video_path: str,
    out_dir: str,
    stride_sec: int,
    max_frames: int,
    scene_threshold: float,
    dedup_hamming: int,
) -> List[Dict[str, Any]]:
    """Sample frames using scene detection and perceptual deduplication."""
    # [MIGRATE] Scene-aware sampling with perceptual deduplication.
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    grab_stride = max(float(stride_sec), 1.0)
    hashes: List[imagehash.ImageHash] = []
    results: List[Dict[str, Any]] = []

    success, frame = cap.read()
    last_hist = None
    last_capture_ts = -float(grab_stride)
    frame_index = 0

    while success and len(results) < max_frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        scene_score = 1.0
        if last_hist is not None:
            scene_score = cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

        timestamp = frame_index / max(fps, 1e-5)
        capture_due_to_scene = scene_score > scene_threshold
        capture_due_to_stride = (timestamp - last_capture_ts) >= grab_stride

        if capture_due_to_scene or capture_due_to_stride or not results:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            frame_hash = imagehash.phash(pil_image)
            is_duplicate = any((frame_hash - existing) <= dedup_hamming for existing in hashes)

            if not is_duplicate:
                file_name = f"frame_{int(timestamp * 1000)}.jpg"
                output_path = os.path.join(out_dir, file_name)
                pil_image.save(output_path, format="JPEG", quality=90)
                hashes.append(frame_hash)
                results.append(
                    {
                        "path": output_path,
                        "timestamp": float(timestamp),
                        "scene_score": float(scene_score),
                    }
                )
                last_capture_ts = timestamp

        last_hist = hist
        frame_index += 1
        success, frame = cap.read()

    cap.release()
    return results


def _persist_frame_paths(
    temp_frames: List[Dict[str, Any]], user_id: Optional[str], video_id: Optional[str]
) -> List[Dict[str, Any]]:
    if not temp_frames:
        return []
    target_dir = os.path.join(YTCACHE, "frames", user_id or "anon", video_id or "unknown")
    os.makedirs(target_dir, exist_ok=True)
    persisted: List[Dict[str, Any]] = []
    for frame in temp_frames:
        src_path = frame["path"]
        filename = os.path.basename(src_path)
        dest = os.path.join(target_dir, filename)
        shutil.move(src_path, dest)
        frame["path"] = dest
        persisted.append(frame)
    return persisted


def extract_youtube_data(url: str, user_id: Optional[str] = None) -> Tuple[Optional[str], List[dict], List[Dict[str, Any]], Dict[str, Any]]:
    transcript = try_transcript(url)
    need_download = (not transcript) or (not settings.youtube.lazy_frames)
    video_path: Optional[str] = None
    frames: List[Dict[str, Any]] = []
    vid = _video_id(url)

    with tempfile.TemporaryDirectory(dir=YTCACHE) as tmp_dir:
        video_path, info = download_lightweight(url, tmp_dir, download=need_download)
        if need_download and video_path:
            duration = _duration_sec(video_path) or 0
            if settings.youtube.max_duration_min > 0 and duration / 60.0 > settings.youtube.max_duration_min:
                raise RuntimeError(f"Video too long ({duration/60:.1f} min)")

            if not settings.youtube.lazy_frames:
                frames_dir = os.path.join(tmp_dir, "frames")
                temp_frames = sample_frames_scene_aware(
                    video_path,
                    frames_dir,
                    stride_sec=settings.youtube.frame_stride_sec,
                    max_frames=settings.youtube.max_frames,
                    scene_threshold=settings.youtube.frame_scene_threshold,
                    dedup_hamming=settings.youtube.frame_dedup_delta,
                )
                frames = _persist_frame_paths(temp_frames, user_id, vid)
        else:
            info = info or {}

    info.setdefault("id", vid)
    info.setdefault("webpage_url", url)

    return video_path, transcript, frames, info
