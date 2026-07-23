"""Low-latency detector for persistent basketball result banners."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ResultBannerCue:
    start: float
    end: float
    kind: str
    side: str | None = None


class ResultBannerTracker:
    """Consume RGB/BGR split-screen frames and emit each persistent result once."""

    def __init__(self, min_duration_sec: float = 0.8, merge_gap_sec: float = 0.8) -> None:
        self.min_duration_sec = min_duration_sec
        self.merge_gap_sec = merge_gap_sec
        self.reset()

    def reset(self) -> None:
        self._kind: str | None = None
        self._start = 0.0
        self._last_seen = 0.0
        self._emitted = False
        self._side_votes: dict[str, int] = {"home": 0, "away": 0}

    @staticmethod
    def _detect_side(gameplay: np.ndarray, *, is_rgb: bool) -> str | None:
        """Classify the large Home/Away word printed below a result banner."""
        height, width = gameplay.shape[:2]
        roi = gameplay[
            int(height * 0.54) : int(height * 0.75),
            int(width * 0.18) : int(width * 0.82),
        ]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV if is_rgb else cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 185), (180, 95, 255))
        count, _, stats, _ = cv2.connectedComponentsWithStats(white)
        scale = max(0.25, roi.shape[0] * roi.shape[1] / (101 * 410))
        glyphs: list[tuple[int, int, int, int, int]] = []
        for x, y, glyph_w, glyph_h, area in stats[1:count]:
            if (
                area >= 450 * scale
                and glyph_h >= 25 * np.sqrt(scale)
                and glyph_w <= roi.shape[1] * 0.18
                and roi.shape[1] * 0.12 <= x <= roi.shape[1] * 0.78
            ):
                glyphs.append((int(x), int(y), int(glyph_w), int(glyph_h), int(area)))
        if len(glyphs) < 4:
            return None
        # Home has an aligned baseline; Away's final y extends below it.
        # Multi-frame voting rejects occasional collisions with background text.
        glyphs = sorted(sorted(glyphs, key=lambda item: item[4], reverse=True)[:4])
        if glyphs[-1][0] - glyphs[0][0] > roi.shape[1] * 0.55:
            return None
        first_three_bottom = int(np.median([y + glyph_h for _, y, _, glyph_h, _ in glyphs[:3]]))
        last_bottom = glyphs[-1][1] + glyphs[-1][3]
        return "away" if last_bottom - first_three_bottom >= 7 * np.sqrt(scale) else "home"

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        *,
        is_rgb: bool = True,
        splitscreen: bool = True,
    ) -> ResultBannerCue | None:
        """Detect persistent Scored! / Out of Bounds! banners.

        splitscreen=True  → use the RIGHT half (file / dual LEFT|RIGHT layout).
        splitscreen=False → treat the whole frame as gameplay (pure VR / OBS).
        """
        gameplay = frame[:, frame.shape[1] // 2 :] if splitscreen else frame
        height = gameplay.shape[0]
        center = gameplay[int(height * 0.35) : int(height * 0.60)]
        hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV if is_rgb else cv2.COLOR_BGR2HSV)
        orange = cv2.countNonZero(cv2.inRange(hsv, (5, 180, 180), (22, 255, 255)))
        cyan = cv2.countNonZero(cv2.inRange(hsv, (38, 160, 160), (95, 255, 255)))
        scale = center.shape[0] * center.shape[1] / (120 * 640)
        kind = None
        if orange > 5000 * scale:
            kind = "out_of_bounds"
        elif 3000 * scale < cyan < 10000 * scale:
            kind = "score"

        if kind is None and self._kind is not None:
            if timestamp - self._last_seen <= self.merge_gap_sec:
                return None
            self.reset()
            return None
        if kind != self._kind:
            self._kind = kind
            self._start = timestamp
            self._last_seen = timestamp
            self._emitted = False
            self._side_votes = {"home": 0, "away": 0}
            return None
        if kind:
            self._last_seen = timestamp
            side = self._detect_side(gameplay, is_rgb=is_rgb)
            if side:
                self._side_votes[side] += 1
        if kind and not self._emitted and timestamp - self._start >= self.min_duration_sec:
            self._emitted = True
            home_votes = self._side_votes["home"]
            away_votes = self._side_votes["away"]
            side = "home" if home_votes > away_votes else "away" if away_votes > home_votes else None
            return ResultBannerCue(self._start, timestamp, kind, side)
        return None
