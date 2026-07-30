"""Heuristic dunk cue: player hand contacting the orange rim.

LiveCC rarely names dunks under lag / first-person FOV. In this VR title the
player's gloves are black+yellow and the rim is bright orange, so a cheap
OpenCV overlap check can mark "possible dunk" without waiting on the VLM.

This is soft evidence only — it does not interrupt TTS by itself. When a
``Scored!`` banner arrives within a short window after contact, the score
grounding path may prefer a dunk finish. Contact cues must NOT be written into
the general LiveCC recent-action deque: that would sticky-label every later
basket as a dunk via ``pick_result_action``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Used only when Scored! arrives inside the fresh contact window.
DUNK_CONTACT_EVIDENCE = (
    "The player reaches rim height with the hand on the rim for a dunk."
)


@dataclass(frozen=True)
class RimHandContact:
    """One confirmed hand-on-rim observation."""

    timestamp: float
    overlap_pixels: int
    rim_pixels: int
    hand_pixels: int


class RimHandContactDetector:
    """Detect black+yellow player gloves overlapping the orange rim.

    Tuned conservative for ~640×480 first-person / free-switch gameplay.
    False positives (layups / jumpers near the hoop) are worse than misses, so
    thresholds prefer silence over spam. Emitting a "possible dunk" UI cue is
    cheaper than *attaching* dunk evidence to a Scored! banner — attach uses a
    higher overlap / hand bar so a marginal brush near the iron cannot force
    「灌籃」 over the real finish.
    """

    def __init__(
        self,
        *,
        # Session obs_20260728_234332: Home Scored! @109.12 attached dunk from a
        # contact with overlap=22 / hand=68 (exact old floor of 22) — user
        # confirmed no dunk. Emit floor is 30 so real flicker samples (~30)
        # still fire; attach uses a higher bar so weak brushes cannot force
        # 「灌籃」.
        min_overlap: int = 30,
        min_rim_pixels: int = 60,
        # Real dunks often show only a sliver of glove yellow (partial occlusion /
        # motion blur); 20 was rejecting confirmed black+overlap hand-on-rim
        # moments (e.g. yellow peaked at 16 across a multi-frame dunk).
        min_yellow_near: int = 10,
        min_black_near: int = 18,
        # Only attach dunk wording to Scored! when contact was clearly a hand
        # on the iron, not a 1-frame brush / UI flicker at the emit floor.
        min_attach_overlap: int = 55,
        min_attach_hand_pixels: int = 120,
        confirm_frames: int = 2,
        cooldown_sec: float = 2.0,
        # Scored! banners frequently land 3–7s after the hand-on-rim moment
        # (celebration camera / OpenCV confirm lag). Real session: dunk @5.91,
        # Scored! Home @11.97 (Δ=6.06s) — 4s was still too tight.
        attach_window_sec: float = 7.5,
        # Contact that arrives during the Scored! hold (after banner start)
        # must still attach — dunk frames often confirm a beat after the
        # banner flashes. Strength (for_attach), not a shorter post-window,
        # is what blocks post-score celebration brushes.
        attach_post_window_sec: float = 3.0,
        rim_y_max_frac: float = 0.55,
        color_memory_frames: int = 4,
    ) -> None:
        self.min_overlap = int(min_overlap)
        self.min_rim_pixels = int(min_rim_pixels)
        self.min_yellow_near = int(min_yellow_near)
        self.min_black_near = int(min_black_near)
        self.min_attach_overlap = int(min_attach_overlap)
        self.min_attach_hand_pixels = int(min_attach_hand_pixels)
        self.confirm_frames = int(confirm_frames)
        self.cooldown_sec = float(cooldown_sec)
        self.attach_window_sec = float(attach_window_sec)
        self.attach_post_window_sec = float(attach_post_window_sec)
        self.rim_y_max_frac = float(rim_y_max_frac)
        # Real dunks flicker between which glove colour is clearly visible
        # from one ~0.14s sample to the next (fast motion, partial occlusion),
        # so requiring both yellow AND black to independently clear their
        # thresholds in the exact same frame was silently rejecting genuine
        # contact. Remember the last few frames' best yellow/black-near-rim
        # readings and require both to have shown up recently instead of
        # simultaneously — this keeps the "not just a yellow UI blob" guard
        # from ``test_rim_alone_ball_or_yellow_only_is_not_a_dunk`` while
        # tolerating the flicker.
        self._color_memory: deque[tuple[int, int]] = deque(maxlen=int(color_memory_frames))
        self._hit_streak = 0
        self._last_emit_t = -1e9
        self._last_contact_t = -1e9
        self._last_contact_overlap = 0
        self._last_contact_hand = 0
        self.last_probe: dict[str, int] = {}

    def reset(self) -> None:
        self._color_memory.clear()
        self._hit_streak = 0
        self._last_emit_t = -1e9
        self._last_contact_t = -1e9
        self._last_contact_overlap = 0
        self._last_contact_hand = 0
        self.last_probe = {}

    @property
    def last_contact_t(self) -> float:
        return self._last_contact_t

    def recently_contacted(
        self,
        timestamp: float,
        *,
        window_sec: float | None = None,
        post_window_sec: float | None = None,
        for_attach: bool = False,
    ) -> bool:
        """True if a dunk-contact cue is near ``timestamp``.

        Accepts contact shortly *before* the score banner (pre-window) **or**
        shortly *after* it (post-window). The post-window matters because
        ``confirm_frames`` + scan cadence often finalize the dunk cue during
        the Scored! hold, after ``cue.start`` was already recorded.

        When ``for_attach`` is True, also require the last contact to clear the
        stronger attach thresholds — UI "possible dunk" spam is tolerable;
        forcing 「灌籃」 on a weak brush is not.
        """
        window = self.attach_window_sec if window_sec is None else float(window_sec)
        post = (
            self.attach_post_window_sec
            if post_window_sec is None
            else float(post_window_sec)
        )
        delta = timestamp - self._last_contact_t
        if delta >= 0.0:
            in_window = delta <= window
        else:
            in_window = -delta <= post
        if not in_window:
            return False
        if for_attach:
            return (
                self._last_contact_overlap >= self.min_attach_overlap
                and self._last_contact_hand >= self.min_attach_hand_pixels
            )
        return True

    @staticmethod
    def _as_bgr(frame: np.ndarray, *, is_rgb: bool) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if is_rgb else frame

    @staticmethod
    def _orange_mask(hsv: np.ndarray) -> np.ndarray:
        # Basketball rim orange (OpenCV H is 0-179). Keep this clear of glove yellow.
        return cv2.inRange(hsv, (5, 110, 110), (22, 255, 255))

    @staticmethod
    def _yellow_mask(hsv: np.ndarray) -> np.ndarray:
        # Glove yellow only — H starts at 28 so rim orange cannot leak in.
        return cv2.inRange(hsv, (28, 100, 110), (40, 255, 255))

    @staticmethod
    def _black_mask(hsv: np.ndarray) -> np.ndarray:
        return cv2.inRange(hsv, (0, 0, 0), (180, 80, 55))

    def _rim_mask(self, orange: np.ndarray) -> np.ndarray:
        """Keep thin / elongated orange structure; drop the filled ball blob."""
        height, width = orange.shape[:2]
        y_max = max(1, int(height * self.rim_y_max_frac))
        band = orange.copy()
        band[y_max:, :] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(band)
        rim = np.zeros_like(band)
        frame_area = float(height * width)
        for label in range(1, count):
            _x, _y, w, h, area = stats[label]
            if area < self.min_rim_pixels:
                continue
            if area > 0.035 * frame_area:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            fill = area / max(w * h, 1)
            # Prefer elongated / hollow rim arcs; reject compact filled blobs.
            if fill > 0.65 and aspect < 1.8:
                continue
            if aspect < 1.35 and fill > 0.55:
                continue
            rim[labels == label] = 255
        return rim

    def probe(self, frame: np.ndarray, *, is_rgb: bool = True) -> dict[str, int]:
        """Return mask sizes for debugging (no debounce)."""
        bgr = self._as_bgr(frame, is_rgb=is_rgb)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        height = hsv.shape[0]
        rim = self._rim_mask(self._orange_mask(hsv))
        yellow = self._yellow_mask(hsv)
        black = self._black_mask(hsv)
        # Build the glove in the upper band first (yellow + adjacent black),
        # THEN test overlap with the rim. Requiring yellow∈rim_roi first was
        # chopping the glove in half and leaving yellow/black 30px apart.
        y_max = max(1, int(height * self.rim_y_max_frac))
        yellow_band = yellow.copy()
        yellow_band[y_max:, :] = 0
        black_band = black.copy()
        black_band[y_max:, :] = 0
        inv_yellow = cv2.bitwise_not(yellow_band)
        dist = cv2.distanceTransform(inv_yellow, cv2.DIST_L2, 3)
        near_yellow = np.where(dist <= 14.0, 255, 0).astype(np.uint8)
        black_candidates = cv2.bitwise_and(black_band, near_yellow)
        black_glove = np.zeros_like(black_candidates)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(black_candidates)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if not (15 <= area <= 2200):
                continue
            fill = area / max(w * h, 1)
            if fill < 0.28:
                continue
            black_glove[labels == label] = 255
        hand = cv2.bitwise_or(yellow_band, black_glove)
        rim_touch = cv2.dilate(
            rim, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1
        )
        yellow_on_rim = cv2.bitwise_and(yellow_band, rim_touch)
        black_on_rim = cv2.bitwise_and(black_glove, rim_touch)
        overlap = cv2.bitwise_and(hand, rim_touch)
        return {
            "rim_pixels": int(cv2.countNonZero(rim)),
            "hand_pixels": int(cv2.countNonZero(hand)),
            "yellow_near_rim": int(cv2.countNonZero(yellow_on_rim)),
            "black_near_rim": int(cv2.countNonZero(black_on_rim)),
            "overlap_pixels": int(cv2.countNonZero(overlap)),
        }

    def _is_hit(self, stats: dict[str, int]) -> bool:
        """Decide whether one probe reading counts as hand-on-rim contact.

        Overlap/rim must be strong RIGHT NOW (an actual hand-on-rim moment),
        but the "this is really a two-tone glove, not a yellow UI blob"
        colour check is allowed to look back over the recent memory window
        (real dunk footage alternates which glove colour is clearly visible
        from one ~0.14s sample to the next).

        When overlap + black are already very strong, accept a weaker yellow
        reading — partial occlusion routinely clips yellow below the normal
        threshold while the black glove body is clearly on the iron.
        """
        self._color_memory.append((stats["yellow_near_rim"], stats["black_near_rim"]))
        recent_yellow = max((y for y, _b in self._color_memory), default=0)
        recent_black = max((b for _y, b in self._color_memory), default=0)
        yellow_ok = recent_yellow >= self.min_yellow_near
        if (
            not yellow_ok
            and stats["overlap_pixels"] >= max(self.min_overlap * 3, 60)
            and recent_black >= max(self.min_black_near * 2, 40)
            and recent_yellow >= max(4, self.min_yellow_near // 2)
        ):
            yellow_ok = True
        return (
            stats["rim_pixels"] >= self.min_rim_pixels
            and stats["overlap_pixels"] >= self.min_overlap
            and yellow_ok
            and recent_black >= self.min_black_near
        )

    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        *,
        is_rgb: bool = True,
        splitscreen: bool = False,
    ) -> Optional[RimHandContact]:
        """Return a contact cue when hand-on-rim is confirmed, else None."""
        gameplay = frame[:, frame.shape[1] // 2 :] if splitscreen else frame
        stats = self.probe(gameplay, is_rgb=is_rgb)
        self.last_probe = stats
        if self._is_hit(stats):
            self._hit_streak += 1
        else:
            self._hit_streak = 0
            return None
        if self._hit_streak < self.confirm_frames:
            return None
        if timestamp - self._last_emit_t < self.cooldown_sec:
            return None
        self._hit_streak = 0
        self._last_emit_t = timestamp
        self._last_contact_t = timestamp
        self._last_contact_overlap = int(stats["overlap_pixels"])
        self._last_contact_hand = int(stats["hand_pixels"])
        return RimHandContact(
            timestamp=timestamp,
            overlap_pixels=stats["overlap_pixels"],
            rim_pixels=stats["rim_pixels"],
            hand_pixels=stats["hand_pixels"],
        )
