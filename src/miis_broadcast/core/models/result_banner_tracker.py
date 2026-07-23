"""Low-latency detector for persistent basketball result banners."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_DIGIT_TEMPLATES = {
    digit: np.array([[char == "#" for char in row] for row in rows], dtype=np.uint8)
    for digit, rows in {
        0: ("..######..", ".########.", ".########.", "####..###.", "####..####", "###...####", "###...####", "###...####", "###...####", "###...####", "###...####", "####..###.", "#########.", ".########.", "..######..", "...###...."),
        1: (".....####.", "....#####.", "..#######.", "#########.", "#########.", ".########.", ".....####.", "......###.", "......###.", "......###.", "......###.", "......###.", "......####", "......####", "......####", "......####"),
        2: (".#######..", "#########.", "#########.", "......###.", "......###.", "......###.", "......###.", ".....###..", "....####..", "...###....", "..####....", ".####.....", ".###......", "#########.", "##########", "#########."),
        3: (".#######..", "#########.", "#########.", ".##..####.", "......###.", "......###.", "...######.", "..#######.", "...#######", "......####", "......####", "......####", ".###.#####", ".#########", ".#########", "..######.."),
        4: (".....###..", "....####..", "....####..", "...#####..", "...#####..", "..######..", "..##..##..", ".###..##..", ".##...##..", "##...###..", "##########", "##########", "##########", ".....###..", "......##..", "......##.."),
        5: ("...####...", "#########.", "#########.", "###.......", "###.......", "###.......", "########..", "#########.", "#########.", "......####", ".......###", ".......###", "......####", "#########.", "#########.", "..###....."),
        6: (".....###..", "..#######.", ".#######..", ".###......", "###.......", "###.......", "########..", "#########.", "##########", "###....###", "###....###", "###....###", ".###...###", ".#########", "..########", "...#####.."),
        7: ("##########", "##########", ".########.", "......###.", "......###.", "......###.", ".....###..", ".....###..", ".....###..", "....###...", "....###...", "....##....", "....##....", "...###....", "...###....", "...###...."),
        8: (".#######..", "#########.", "###...###.", "###...###.", "###...###.", ".###..###.", ".#######..", "..######..", ".########.", "####..####", "###....###", "###....###", "###....###", "####...###", ".#########", "..#######."),
        9: (".#######..", "########..", "#########.", "###...####", "###....###", "###....###", "###....###", "####..####", ".#########", "..########", "....#..###", ".......###", "......###.", "..#######.", ".#######..", "..###....."),
    }.items()
}


@dataclass(frozen=True)
class ResultBannerCue:
    start: float
    end: float
    kind: str
    side: str | None = None
    home_score: int | None = None
    away_score: int | None = None


class ResultBannerTracker:
    """Consume RGB/BGR split-screen frames and emit each persistent result once."""

    def __init__(self, min_duration_sec: float = 0.8, merge_gap_sec: float = 0.8) -> None:
        self.min_duration_sec = min_duration_sec
        self.merge_gap_sec = merge_gap_sec
        self.home_score = 0
        self.away_score = 0
        self._last_scoreboard_score: tuple[int, int] | None = None
        self._pending_score_cue: ResultBannerCue | None = None
        self._scoreboard_votes: dict[tuple[int, int], int] = {}
        self.reset()

    def reset(self) -> None:
        """Reset banner debounce and top-scoreboard tracking state."""
        self.home_score = 0
        self.away_score = 0
        self._last_scoreboard_score = None
        self._pending_score_cue = None
        self._scoreboard_votes = {}
        self._reset_banner_state()

    def _reset_banner_state(self) -> None:
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

    @staticmethod
    def _read_score_number(roi: np.ndarray, *, is_rgb: bool) -> int | None:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 170, 255)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        scale = max(0.35, roi.shape[0] / 43.0)
        glyphs: list[tuple[int, int, int, int, int, int]] = []
        for label, (x, y, width, height, area) in enumerate(stats[1:], 1):
            if (
                area >= 35 * scale * scale
                and height >= 15 * scale
                and 3 * scale <= width <= 18 * scale
            ):
                glyphs.append((int(x), int(y), int(width), int(height), int(area), label))
        if not glyphs or len(glyphs) > 2:
            return None
        digits: list[str] = []
        for x, y, width, height, _area, label in sorted(glyphs):
            glyph = (labels[y : y + height, x : x + width] == label).astype(np.uint8)
            normalized = cv2.resize(glyph, (10, 16), interpolation=cv2.INTER_AREA)
            normalized = (normalized >= 0.32).astype(np.uint8)
            distances = {
                digit: float(np.mean(np.abs(normalized.astype(float) - template)))
                for digit, template in _DIGIT_TEMPLATES.items()
            }
            digit, distance = min(distances.items(), key=lambda item: item[1])
            if distance > 0.38:
                return None
            digits.append(str(digit))
        return int("".join(digits))

    @classmethod
    def _detect_scoreboard(
        cls, gameplay: np.ndarray, *, is_rgb: bool
    ) -> tuple[int, int] | None:
        """Read Home/Away scores from the persistent top UI, never the timer."""
        height, width = gameplay.shape[:2]
        y0, y1 = int(height * 0.07), int(height * 0.17)
        home = cls._read_score_number(
            gameplay[y0:y1, int(width * 0.185) : int(width * 0.275)],
            is_rgb=is_rgb,
        )
        away = cls._read_score_number(
            gameplay[y0:y1, int(width * 0.515) : int(width * 0.615)],
            is_rgb=is_rgb,
        )
        if home is None or away is None or home > 99 or away > 99:
            return None
        return home, away

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
        scoreboard = self._detect_scoreboard(gameplay, is_rgb=is_rgb)
        if self._pending_score_cue is not None:
            if scoreboard is not None:
                baseline = self._last_scoreboard_score
                score_changed = baseline is None
                if baseline is not None:
                    old_home, old_away = baseline
                    new_home, new_away = scoreboard
                    score_changed = (
                        (new_home == old_home + 1 and new_away == old_away)
                        or (new_away == old_away + 1 and new_home == old_home)
                    )
                # The scoreboard can reappear with its old value for several
                # seconds after the Scored animation. Do not vote for that
                # stale value; wait until one side's top-UI score advances.
                if score_changed:
                    self._scoreboard_votes[scoreboard] = (
                        self._scoreboard_votes.get(scoreboard, 0) + 1
                    )
                if score_changed and self._scoreboard_votes[scoreboard] >= 3:
                    pending = self._pending_score_cue
                    side = pending.side
                    if self._last_scoreboard_score is not None:
                        old_home, old_away = self._last_scoreboard_score
                        new_home, new_away = scoreboard
                        if new_home > old_home and new_away == old_away:
                            side = "home"
                        elif new_away > old_away and new_home == old_home:
                            side = "away"
                    self._last_scoreboard_score = scoreboard
                    self.home_score, self.away_score = scoreboard
                    self._pending_score_cue = None
                    self._scoreboard_votes = {}
                    return ResultBannerCue(
                        pending.start,
                        pending.end,
                        pending.kind,
                        side,
                        scoreboard[0],
                        scoreboard[1],
                    )
            if timestamp - self._pending_score_cue.end > 7.0:
                pending = self._pending_score_cue
                self._pending_score_cue = None
                self._scoreboard_votes = {}
                return pending
        elif scoreboard is not None:
            self._last_scoreboard_score = scoreboard
            self.home_score, self.away_score = scoreboard

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
            self._reset_banner_state()
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
            cue = ResultBannerCue(self._start, timestamp, kind, side)
            if kind == "score":
                self._pending_score_cue = cue
                self._scoreboard_votes = {}
                return None
            return cue
        return None
