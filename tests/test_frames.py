from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ingest.frames import extract_scene_frames
from app.settings import settings


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg binary not available")
def test_extract_scene_frames_dedup(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (64, 64))

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]
    for color in colors:
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        for _ in range(10):
            writer.write(frame)
    writer.release()

    frames = extract_scene_frames(
        video_path=video_path,
        user_id="test-user",
        doc_id="doc-123",
        scene_threshold=0.1,
        max_frames=5,
        dedup_hamming=4,
    )

    assert frames, "Expected at least one frame to be extracted"
    assert len(frames) <= settings.youtube.max_frames
    for frame in frames:
        assert Path(frame["file_path"]).exists()
        assert frame["start_ts"] >= 0
