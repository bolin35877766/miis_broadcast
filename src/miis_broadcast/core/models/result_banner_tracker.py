"""Low-latency detector for persistent basketball result banners."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .banner_template_matcher import BannerTemplateMatcher, default_template_dir

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
    """Detect ``Scored!`` / ``Out of Bounds!`` / ``Shot Clock Violation!``.

    Primary path: multi-scale template match on the gameplay view (the whole
    frame, or its right half when the feed is a LEFT|RIGHT composite).
    Scoreboard digit templates optionally attach Home/Away totals after a make.
    """

    def __init__(
        self,
        min_duration_sec: float = 0.35,
        merge_gap_sec: float = 0.9,
        template_dir: Optional[Path | str] = None,
        use_templates: bool = True,
    ) -> None:
        self.min_duration_sec = min_duration_sec
        self.merge_gap_sec = merge_gap_sec
        self.home_score = 0
        self.away_score = 0
        self._last_scoreboard_score: tuple[int, int] | None = None
        self._pending_score_cue: ResultBannerCue | None = None
        self._scoreboard_votes: dict[tuple[int, int], int] = {}
        self._matcher: Optional[BannerTemplateMatcher] = None
        if use_templates:
            self._matcher = BannerTemplateMatcher(
                template_dir or default_template_dir()
            )
            if not self._matcher.available:
                self._matcher = None
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
    def _as_bgr(frame: np.ndarray, *, is_rgb: bool) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if is_rgb else frame

    @staticmethod
    def _detect_side(gameplay: np.ndarray, *, is_rgb: bool) -> str | None:
        """Classify the large Home/Away word printed below a result banner.

        Measured on 640×480 gameplay captures: the word sits at y 0.61-0.70 and
        x 0.36-0.59, so the ROI below brackets it with margin for FOV drift.
        "Away" is told from "Home" by its descender dropping below the baseline.
        """
        height, width = gameplay.shape[:2]
        roi = gameplay[
            int(height * 0.56) : int(height * 0.80),
            int(width * 0.20) : int(width * 0.80),
        ]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV if is_rgb else cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 185), (180, 95, 255))
        count, _, stats, _ = cv2.connectedComponentsWithStats(white)
        # Reference ROI area at 640×480: ~115×384.
        scale = max(0.25, roi.shape[0] * roi.shape[1] / (115 * 384))
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
        """Read Home/Away scores from the persistent top UI, never the timer.

        Measured on 640×480 captures: digits sit at y 0.05-0.13. Home is just
        left of the clock (x ~0.34-0.36); Away is just right of it
        (x ~0.515-0.555). The clock itself occupies ~0.39-0.46 and must stay out
        of both windows. Enrichment only — a missed digit never blocks a banner.
        """
        height, width = gameplay.shape[:2]
        y0, y1 = int(height * 0.050), int(height * 0.130)
        home = cls._read_score_number(
            gameplay[y0:y1, int(width * 0.340) : int(width * 0.360)],
            is_rgb=is_rgb,
        )
        away = cls._read_score_number(
            gameplay[y0:y1, int(width * 0.515) : int(width * 0.555)],
            is_rgb=is_rgb,
        )
        if home is None or away is None or home > 99 or away > 99:
            return None
        return home, away

    def _resolve_side(
        self, gameplay: np.ndarray, gameplay_bgr: np.ndarray, *, is_rgb: bool
    ) -> Optional[str]:
        if self._matcher is not None:
            side, conf = self._matcher.match_side(gameplay_bgr)
            if side is not None:
                return side
        return self._detect_side(gameplay, is_rgb=is_rgb)

    def _finish_score_with_board(
        self, pending: ResultBannerCue, scoreboard: tuple[int, int], side: Optional[str]
    ) -> ResultBannerCue:
        self._last_scoreboard_score = scoreboard
        self.home_score, self.away_score = scoreboard
        self._pending_score_cue = None
        self._scoreboard_votes = {}
        logging.info(
            "[ResultBanner] confirmed Scored! %s (%d-%d)",
            side, scoreboard[0], scoreboard[1],
        )
        return ResultBannerCue(
            pending.start,
            pending.end,
            pending.kind,
            side,
            scoreboard[0],
            scoreboard[1],
        )

    def debug_probe(self, frame: np.ndarray, *, is_rgb: bool = True) -> str:
        """Return a one-line report of best template confidences per region.

        Helps locate the banner in a live composite: scans the full frame plus
        left / right halves so we can see which region (if any) the templates
        actually respond to, and at what confidence vs the score_threshold.
        """
        if self._matcher is None:
            return "[BannerProbe] templates unavailable"
        regions = {
            "full": frame,
            "left": frame[:, : frame.shape[1] // 2],
            "right": frame[:, frame.shape[1] // 2 :],
        }
        parts: list[str] = []
        for name, region in regions.items():
            region_bgr = self._as_bgr(region, is_rgb=is_rgb)
            _kind, confs = self._matcher.match_kind(region_bgr)
            side, side_c = self._matcher.match_side(region_bgr)
            parts.append(
                f"{name}: scored={confs.get('score', -1):.2f} "
                f"oob={confs.get('out_of_bounds', -1):.2f} "
                f"scv={confs.get('shot_clock_violation', -1):.2f} "
                f"side={side or '-'}({side_c:.2f})"
            )
        return (
            f"[BannerProbe] thr={self._matcher.score_threshold:.2f} | "
            + " | ".join(parts)
        )

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
        gameplay_bgr = self._as_bgr(gameplay, is_rgb=is_rgb)
        scoreboard = self._detect_scoreboard(gameplay, is_rgb=is_rgb)

        # ── Pending score waiting for scoreboard +1 (enrichment) ──────────
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
                if score_changed:
                    self._scoreboard_votes[scoreboard] = (
                        self._scoreboard_votes.get(scoreboard, 0) + 1
                    )
                if score_changed and self._scoreboard_votes[scoreboard] >= 2:
                    pending = self._pending_score_cue
                    side = pending.side
                    if self._last_scoreboard_score is not None:
                        old_home, old_away = self._last_scoreboard_score
                        new_home, new_away = scoreboard
                        if new_home > old_home and new_away == old_away:
                            side = "home"
                        elif new_away > old_away and new_home == old_home:
                            side = "away"
                    return self._finish_score_with_board(pending, scoreboard, side)
            # Template already named the side — don't wait forever for OCR digits.
            if (
                self._pending_score_cue.side is not None
                and timestamp - self._pending_score_cue.end > 1.2
            ):
                pending = self._pending_score_cue
                self._pending_score_cue = None
                self._scoreboard_votes = {}
                logging.info(
                    "[ResultBanner] emit Scored! %s (template; scoreboard not ready)",
                    pending.side,
                )
                return pending
            if timestamp - self._pending_score_cue.end > 7.0:
                logging.info(
                    "[ResultBanner] discard pending Scored! (no confirm, side=%s)",
                    self._pending_score_cue.side,
                )
                self._pending_score_cue = None
                self._scoreboard_votes = {}
        elif scoreboard is not None:
            self._last_scoreboard_score = scoreboard
            self.home_score, self.away_score = scoreboard

        # ── Kind detection: template match only ───────────────────────────
        kind: Optional[str] = None
        confs: dict[str, float] = {
            "score": -1.0,
            "out_of_bounds": -1.0,
            "shot_clock_violation": -1.0,
        }
        if self._matcher is not None:
            kind, confs = self._matcher.match_kind(gameplay_bgr)

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
            side = self._resolve_side(gameplay, gameplay_bgr, is_rgb=is_rgb)
            if side:
                self._side_votes[side] += 1
        if kind and not self._emitted and timestamp - self._start >= self.min_duration_sec:
            self._emitted = True
            home_votes = self._side_votes["home"]
            away_votes = self._side_votes["away"]
            side = (
                "home"
                if home_votes > away_votes
                else "away"
                if away_votes > home_votes
                else None
            )
            cue = ResultBannerCue(self._start, timestamp, kind, side)
            scored_c = confs.get("score", -1.0)
            oob_c = confs.get("out_of_bounds", -1.0)
            scv_c = confs.get("shot_clock_violation", -1.0)
            if kind == "score":
                # A template-confirmed Scored! is trustworthy on its own → emit
                # immediately so the broadcast actually cuts in. Side (Home/Away)
                # and scoreboard digits are best-effort enrichment only.
                home_s = away_s = None
                if scoreboard is not None and self._last_scoreboard_score is not None:
                    old_home, old_away = self._last_scoreboard_score
                    new_home, new_away = scoreboard
                    if (
                        (new_home == old_home + 1 and new_away == old_away)
                        or (new_away == old_away + 1 and new_home == old_home)
                    ):
                        home_s, away_s = new_home, new_away
                        self._last_scoreboard_score = scoreboard
                        self.home_score, self.away_score = scoreboard
                logging.info(
                    "[ResultBanner] Scored! %s via template "
                    "(scored=%.2f oob=%.2f scv=%.2f)",
                    side, scored_c, oob_c, scv_c,
                )
                return ResultBannerCue(
                    self._start, timestamp, "score", side, home_s, away_s
                )
            if kind == "shot_clock_violation":
                logging.info(
                    "[ResultBanner] Shot Clock Violation! %s "
                    "(tm scored=%.2f oob=%.2f scv=%.2f)",
                    side, scored_c, oob_c, scv_c,
                )
                return cue
            logging.info(
                "[ResultBanner] Out of Bounds! %s "
                "(tm scored=%.2f oob=%.2f scv=%.2f)",
                side, scored_c, oob_c, scv_c,
            )
            return cue
        return None
