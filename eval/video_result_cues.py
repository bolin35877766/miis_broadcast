"""Detect persistent basketball result banners directly from video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoResultCue:
    start: float
    end: float
    kind: str


def detect_result_cues(
    video: Path,
    *,
    sample_hz: float = 10.0,
    merge_gap_sec: float = 0.8,
    min_duration_sec: float = 0.8,
) -> list[VideoResultCue]:
    """Return persistent cyan score and orange out-of-bounds banners.

    The game renders both result labels across the horizontal center of the
    right-hand gameplay view. Requiring persistence rejects court lines,
    avatars, and brief UI highlights that share either hue.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / sample_hz))
    hits: list[tuple[float, str, int]] = []

    try:
        for frame_index in range(0, frame_count, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            gameplay = frame[:, frame.shape[1] // 2 :]
            height = gameplay.shape[0]
            center = gameplay[int(height * 0.35) : int(height * 0.60)]
            hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            orange = cv2.countNonZero(cv2.inRange(hsv, (5, 180, 180), (22, 255, 255)))
            cyan = cv2.countNonZero(cv2.inRange(hsv, (38, 160, 160), (95, 255, 255)))
            kind = "out_of_bounds" if orange > 5000 else ("score" if cyan > 3000 else "")
            if kind:
                hits.append((frame_index / fps, kind, orange if kind == "out_of_bounds" else cyan))
    finally:
        capture.release()

    groups: list[list[tuple[float, str, int]]] = []
    for hit in hits:
        if (
            not groups
            or hit[1] != groups[-1][-1][1]
            or hit[0] - groups[-1][-1][0] > merge_gap_sec
        ):
            groups.append([hit])
        else:
            groups[-1].append(hit)

    sample_period = step / fps
    cues = []
    for group in groups:
        start = group[0][0]
        end = group[-1][0] + sample_period
        # A large solid cyan game element can occupy the center briefly.  Score
        # banner lettering is persistent but sparse (about 3k-5k pixels).
        sparse_score = group[0][1] != "score" or max(hit[2] for hit in group) < 10000
        if end - start >= min_duration_sec and sparse_score:
            cues.append(VideoResultCue(start=start, end=end, kind=group[0][1]))
    return cues
