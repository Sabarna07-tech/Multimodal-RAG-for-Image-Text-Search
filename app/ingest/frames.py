from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from PIL import Image
import imagehash

from app.settings import settings

_PTS_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg fails to produce candidate frames."""


def _require_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise FrameExtractionError("ffmpeg binary not found in PATH.")
    return binary


def _parse_showinfo(log_text: str) -> List[float]:
    timestamps: List[float] = []
    for match in _PTS_RE.finditer(log_text):
        timestamps.append(float(match.group(1)))
    return timestamps


def extract_scene_frames(
    video_path: Path,
    user_id: str,
    doc_id: str,
    scene_threshold: float,
    max_frames: int,
    dedup_hamming: int = 4,
) -> List[Dict[str, float]]:
    """
    Extract frames at scene boundaries and apply perceptual deduplication.

    # [MIGRATE] ffmpeg-driven scene-aware frame sampler with phash deduplication.
    """
    ffmpeg_bin = _require_ffmpeg()
    target_dir = Path(settings.paths.ingest_cache_dir) / "frames" / user_id / doc_id
    target_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = target_dir / "tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_template = temp_dir / "frame_%06d.jpg"
    filter_expr = f"select='gt(scene,{scene_threshold})',showinfo"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(video_path),
        "-vf",
        filter_expr,
        "-vsync",
        "vfr",
        str(output_template),
    ]
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(settings.youtube.timeout_sec, 30),
    )
    if process.returncode != 0:
        raise FrameExtractionError(f"ffmpeg scene extraction failed: {process.stderr}")

    timestamps = _parse_showinfo(process.stderr)
    candidates = sorted(temp_dir.glob("frame_*.jpg"))
    if not candidates:
        return []

    deduped: List[Dict[str, float]] = []
    hashes: List[imagehash.ImageHash] = []
    for index, frame_path in enumerate(candidates):
        with Image.open(frame_path) as img:
            img_rgb = img.convert("RGB")
            phash = imagehash.phash(img_rgb)
        if any(existing - phash <= dedup_hamming for existing in hashes):
            continue
        hashes.append(phash)

        timestamp = timestamps[index] if index < len(timestamps) else 0.0
        dest_name = f"{doc_id}_frame_{len(deduped):06d}.jpg"
        dest_path = target_dir / dest_name
        shutil.move(str(frame_path), dest_path)
        deduped.append(
            {
                "file_path": str(dest_path),
                "start_ts": float(timestamp),
                "end_ts": float(timestamp),
            }
        )
        if len(deduped) >= max_frames:
            break

    shutil.rmtree(temp_dir, ignore_errors=True)
    return deduped


__all__ = ["extract_scene_frames", "FrameExtractionError"]

