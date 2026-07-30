"""Smoke tests for LiveCC 2 FPS clip sampling under bursty capture."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miis_broadcast.workers.livecc import FrameItem, build_clip_from_buffer


def test_build_clip_samples_evenly_at_target_fps() -> None:
    # Simulate ~30 FPS capture into a 1.0s window; clip should keep ~2 FPS spacing.
    buf: deque[FrameItem] = deque()
    for i in range(31):
        t = i / 30.0
        frame = np.full((48, 64, 3), i % 255, dtype=np.uint8)
        buf.append(FrameItem(frame=frame, t=t))

    clip = build_clip_from_buffer(buf, window_sec=1.0, target_fps=2.0)
    assert clip is not None
    assert clip.frames.shape[0] >= 2
    assert abs(clip.fps - 2.0) < 1e-6
    # Even grid for 1s @ 2 FPS → about 3 samples (t0, mid, t1).
    assert 2 <= clip.frames.shape[0] <= 4


def test_build_clip_survives_bursty_then_gap() -> None:
    buf: deque[FrameItem] = deque()
    # Burst at start, then a later frame — sampling must still return ≥2 frames.
    for i in range(10):
        buf.append(FrameItem(frame=np.zeros((32, 32, 3), np.uint8), t=i * 0.01))
    buf.append(FrameItem(frame=np.ones((32, 32, 3), np.uint8) * 9, t=1.0))
    clip = build_clip_from_buffer(buf, window_sec=1.0, target_fps=2.0)
    assert clip is not None
    assert clip.frames.shape[0] >= 2
