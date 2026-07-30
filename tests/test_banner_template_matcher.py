"""Template-matching result banner smoke tests (uses committed assets).

Banners are painted at the fractions measured on real 640×480 captures, which
is also how the width-fraction matcher searches:

    banner text   y 0.42,  width 0.31 (Scored!) / 0.60 (Out of Bounds!) / 0.70 (SCV)
    Home / Away   y 0.61,  width 0.22, centred

The default feed is one 640×480 view; a LEFT|RIGHT composite puts the same
fractions inside its right half.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from miis_broadcast.core.models.banner_template_matcher import BannerTemplateMatcher
from miis_broadcast.core.models.result_banner_tracker import ResultBannerTracker

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "assets" / "result_banners"

# LiveCC gameplay resolution.
FRAME_W, FRAME_H = 640, 480

BANNER_Y_FRAC = 0.4226
SIDE_Y_FRAC = 0.6067
SIDE_WIDTH_FRAC = 0.2202
BANNER_WIDTH_FRACS = {
    "scored.png": 0.3091,
    "out_of_bounds.png": 0.6048,
    "shot_clock_violation.png": 0.7027,
}


def _paint_gameplay(
    banner_name: str,
    side_name: str | None,
    *,
    height: int = FRAME_H,
    width: int = FRAME_W,
) -> np.ndarray:
    """Paint a banner (and optional side word) into a single gameplay view."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)

    def place(name: str, width_frac: float, y_frac: float) -> None:
        image = cv2.imread(str(TEMPLATES / name))
        assert image is not None, name
        target_w = int(width * width_frac)
        target_h = int(round(target_w * image.shape[0] / image.shape[1]))
        image = cv2.resize(image, (target_w, target_h))
        y0 = int(height * y_frac)
        x0 = (width - target_w) // 2
        frame[y0 : y0 + target_h, x0 : x0 + target_w] = image

    place(banner_name, BANNER_WIDTH_FRACS[banner_name], BANNER_Y_FRAC)
    if side_name is not None:
        place(side_name, SIDE_WIDTH_FRAC, SIDE_Y_FRAC)
    return frame


def test_matcher_loads_and_self_matches() -> None:
    assert (TEMPLATES / "scored.png").is_file()
    assert (TEMPLATES / "shot_clock_violation.png").is_file()
    matcher = BannerTemplateMatcher(TEMPLATES)
    assert matcher.available
    kind, confs = matcher.match_kind(_paint_gameplay("scored.png", None))
    assert kind == "score"
    assert confs["score"] >= matcher.score_threshold
    assert confs["score"] > confs["out_of_bounds"]
    assert confs["score"] > confs["shot_clock_violation"]


def test_templates_keep_the_rendered_width_to_height_ratio() -> None:
    """Crops from a different aspect ratio stretch glyphs and kill confidence."""
    expected = {
        "scored.png": 0.261,
        "out_of_bounds.png": 0.134,
        "shot_clock_violation.png": 0.091,
        "home.png": 0.327,
        "away.png": 0.300,
    }
    for name, aspect in expected.items():
        image = cv2.imread(str(TEMPLATES / name))
        assert image is not None, name
        actual = image.shape[0] / image.shape[1]
        assert abs(actual - aspect) < 0.03, (name, actual, aspect)


def _first_cue(frame: np.ndarray, *, splitscreen: bool):
    tracker = ResultBannerTracker(min_duration_sec=0.15, template_dir=TEMPLATES)
    for step in range(10):
        cue = tracker.update(frame, step * 0.08, is_rgb=False, splitscreen=splitscreen)
        if cue is not None:
            return cue
    return None


CASES = (
    ("scored.png", "away.png", "score", "away"),
    ("out_of_bounds.png", "home.png", "out_of_bounds", "home"),
    ("shot_clock_violation.png", "home.png", "shot_clock_violation", "home"),
)


def test_tracker_emits_on_a_single_640x480_view() -> None:
    for banner, side_image, expect_kind, expect_side in CASES:
        cue = _first_cue(_paint_gameplay(banner, side_image), splitscreen=False)
        assert cue is not None, banner
        assert cue.kind == expect_kind, (banner, cue)
        assert cue.side == expect_side, (banner, cue)


def test_tracker_emits_from_the_right_half_of_a_composite() -> None:
    for banner, side_image, expect_kind, expect_side in CASES:
        frame = np.full((FRAME_H, FRAME_W * 2, 3), 30, dtype=np.uint8)
        frame[:, FRAME_W:] = _paint_gameplay(banner, side_image)
        cue = _first_cue(frame, splitscreen=True)
        assert cue is not None, banner
        assert cue.kind == expect_kind, (banner, cue)
        assert cue.side == expect_side, (banner, cue)


def test_blank_frame_never_emits() -> None:
    tracker = ResultBannerTracker(min_duration_sec=0.15, template_dir=TEMPLATES)
    rng = np.random.default_rng(0)
    frame = (rng.random((FRAME_H, FRAME_W, 3)) * 255).astype(np.uint8)
    assert all(
        tracker.update(frame, i * 0.08, is_rgb=False, splitscreen=False) is None
        for i in range(15)
    )


def test_scenery_without_a_banner_never_emits() -> None:
    """Smooth gradients + blobs stand in for court/sky clutter."""
    tracker = ResultBannerTracker(min_duration_sec=0.15, template_dir=TEMPLATES)
    rng = np.random.default_rng(7)
    frame = (rng.random((FRAME_H, FRAME_W, 3)) * 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (61, 61), 0)
    frame[:, :, 2] = np.clip(frame[:, :, 2].astype(int) + 60, 0, 255).astype(np.uint8)
    assert all(
        tracker.update(frame, i * 0.08, is_rgb=False, splitscreen=False) is None
        for i in range(15)
    )


if __name__ == "__main__":
    test_matcher_loads_and_self_matches()
    test_templates_keep_the_rendered_width_to_height_ratio()
    test_tracker_emits_on_a_single_640x480_view()
    test_tracker_emits_from_the_right_half_of_a_composite()
    test_blank_frame_never_emits()
    test_scenery_without_a_banner_never_emits()
    print("ok")
