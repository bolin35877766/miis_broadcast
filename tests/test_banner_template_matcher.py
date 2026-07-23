"""Template-matching result banner smoke tests (uses committed assets).

Banners are painted at realistic on-screen fractions of the right (gameplay)
half, mirroring how the width-fraction matcher searches — live "Scored!" spans
~1/3 of the half, "Out of Bounds!" ~2/3.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from miis_broadcast.core.models.banner_template_matcher import BannerTemplateMatcher
from miis_broadcast.core.models.result_banner_tracker import ResultBannerTracker

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "assets" / "result_banners"


def _paint_gameplay(
    banner_name: str,
    side_name: str | None,
    *,
    height: int = 1080,
    width: int = 960,
    banner_frac: float = 0.33,
) -> np.ndarray:
    """Paint a banner (and optional side word) into a single gameplay view."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    banner = cv2.imread(str(TEMPLATES / banner_name))
    assert banner is not None, banner_name
    bw = int(width * banner_frac)
    bh = int(bw * banner.shape[0] / banner.shape[1])
    banner = cv2.resize(banner, (bw, bh))
    y0 = int(height * 0.46)
    x0 = (width - bw) // 2
    frame[y0 : y0 + bh, x0 : x0 + bw] = banner
    if side_name is not None:
        side = cv2.imread(str(TEMPLATES / side_name))
        assert side is not None, side_name
        sw = int(width * 0.28)
        sh = int(sw * side.shape[0] / side.shape[1])
        side = cv2.resize(side, (sw, sh))
        sy = y0 + bh + int(height * 0.02)
        sx = (width - sw) // 2
        frame[sy : sy + sh, sx : sx + sw] = side
    return frame


def test_matcher_loads_and_self_matches() -> None:
    assert (TEMPLATES / "scored.png").is_file()
    matcher = BannerTemplateMatcher(TEMPLATES)
    assert matcher.available
    kind, scored_c, oob_c = matcher.match_kind(_paint_gameplay("scored.png", None))
    assert kind == "score"
    assert scored_c >= matcher.score_threshold
    assert scored_c > oob_c


def test_tracker_emits_score_and_oob_on_synthetic_canvas() -> None:
    def full_frame(banner_name: str, side_name: str, frac: float) -> np.ndarray:
        frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
        frame[:, 960:] = _paint_gameplay(banner_name, side_name, banner_frac=frac)
        return frame

    tracker = ResultBannerTracker(min_duration_sec=0.15, template_dir=TEMPLATES)
    cue = None
    frame = full_frame("scored.png", "away.png", 0.33)
    for i in range(10):
        cue = tracker.update(frame, i * 0.08, is_rgb=False, splitscreen=True)
        if cue is not None:
            break
    assert cue is not None
    assert cue.kind == "score"
    assert cue.side == "away"

    tracker.reset()
    cue = None
    frame = full_frame("out_of_bounds.png", "home.png", 0.66)
    for i in range(10):
        cue = tracker.update(frame, i * 0.08, is_rgb=False, splitscreen=True)
        if cue is not None:
            break
    assert cue is not None
    assert cue.kind == "out_of_bounds"
    assert cue.side == "home"


def test_blank_frame_never_emits() -> None:
    tracker = ResultBannerTracker(min_duration_sec=0.15, template_dir=TEMPLATES)
    rng = np.random.default_rng(0)
    frame = (rng.random((1080, 1920, 3)) * 255).astype(np.uint8)
    assert all(
        tracker.update(frame, i * 0.08, is_rgb=False, splitscreen=True) is None
        for i in range(15)
    )


if __name__ == "__main__":
    test_matcher_loads_and_self_matches()
    test_tracker_emits_score_and_oob_on_synthetic_canvas()
    test_blank_frame_never_emits()
    print("ok")
