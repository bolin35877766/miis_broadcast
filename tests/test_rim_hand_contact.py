"""Synthetic-frame tests for the rim × player-hand dunk cue."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miis_broadcast.core.models.broadcast_grounding import (
    compose_result_evidence,
    ground_broadcast_text,
    pick_result_action,
    reset_style_line_memory,
)
from miis_broadcast.core.models.rim_hand_contact import (
    DUNK_CONTACT_EVIDENCE,
    RimHandContactDetector,
)


def _blank(h: int = 480, w: int = 640) -> np.ndarray:
    # Mid-gray court so the black glove is a separate component (not merged
    # into a solid dark background).
    return np.full((h, w, 3), 110, dtype=np.uint8)


def _paint_rim(frame: np.ndarray) -> None:
    # Thin orange ellipse near the top — rim-like, not a filled ball.
    cv2.ellipse(frame, (320, 140), (90, 28), 0, 0, 360, (0, 100, 255), 5)


def _paint_ball(frame: np.ndarray) -> None:
    cv2.circle(frame, (320, 220), 42, (0, 90, 255), -1)


def _paint_hand_on_rim(frame: np.ndarray) -> None:
    # Small black+yellow glove whose bottom edge sits on the rim arc without
    # completely erasing the orange iron underneath.
    cv2.circle(frame, (320, 108), 12, (12, 12, 12), -1)
    cv2.circle(frame, (328, 104), 7, (0, 255, 255), -1)


def _paint_yellow_near_rim_only(frame: np.ndarray) -> None:
    # Yellow highlight without black glove body — must NOT count as a dunk.
    cv2.circle(frame, (328, 108), 8, (0, 255, 255), -1)


def test_hand_on_rim_emits_dunk_contact() -> None:
    det = RimHandContactDetector(confirm_frames=3, cooldown_sec=0.0)
    frame = _blank()
    _paint_rim(frame)
    _paint_hand_on_rim(frame)
    assert det.update(frame, 1.0, is_rgb=False) is None
    assert det.update(frame, 1.1, is_rgb=False) is None
    contact = det.update(frame, 1.2, is_rgb=False)
    assert contact is not None
    assert contact.overlap_pixels >= det.min_overlap
    assert det.recently_contacted(1.5)
    assert det.recently_contacted(8.0)  # inside attach_window_sec (~7.5s)
    assert not det.recently_contacted(9.5)  # outside attach_window_sec
    # Contact during the Scored! hold (after banner start) must still attach.
    assert det.recently_contacted(0.5)  # contact @1.2 is 0.7s after banner @0.5


def test_dunk_contact_six_seconds_before_score_still_attaches() -> None:
    # Regression: dunk @5.91 / Scored! Home @11.97 (Δ=6.06s). LiveCC said
    # "layup" and the old 4s window dropped the dunk, so TTS said 上籃.
    det = RimHandContactDetector()
    det._last_contact_t = 5.91
    assert det.recently_contacted(11.97)


def test_strong_black_overlap_with_weak_yellow_still_counts_as_dunk() -> None:
    # Regression from obs session: Home Scored! @ 63.48 had black/overlap
    # clearly on the rim for ~0.7s, but yellow never cleared the old floor
    # of 20 (peaked at 16). That dunk was silently dropped.
    det = RimHandContactDetector(confirm_frames=2, cooldown_sec=0.0)
    samples = [
        (61.83, 4087, 2, 96, 98),
        (62.00, 4071, 3, 93, 96),
        (62.12, 3123, 7, 157, 164),
        (62.25, 6188, 16, 102, 118),
        (62.38, 6903, 4, 40, 44),
    ]
    contact = None
    for sec, rim, yellow, black, overlap in samples:
        stats = {
            "rim_pixels": rim,
            "hand_pixels": 0,
            "yellow_near_rim": yellow,
            "black_near_rim": black,
            "overlap_pixels": overlap,
        }
        det._hit_streak = det._hit_streak + 1 if det._is_hit(stats) else 0
        if det._hit_streak >= det.confirm_frames:
            contact = sec
            break
    assert contact == 62.25


def test_contact_during_score_hold_still_attaches() -> None:
    # Regression: Scored! Home @ 12.97, RimHand finalized @ 15.63 during the
    # hold. Old recently_contacted(start) required contact BEFORE start, so
    # the dunk evidence was dropped and the call became generic "得分".
    det = RimHandContactDetector(confirm_frames=1, cooldown_sec=0.0)
    det._last_contact_t = 15.63
    det._last_contact_overlap = 200
    det._last_contact_hand = 500
    assert det.recently_contacted(12.97)
    assert det.recently_contacted(12.97, for_attach=True)
    assert not det.recently_contacted(12.97, post_window_sec=2.0)  # 2.66s > 2.0


def test_marginal_rim_brush_must_not_attach_dunk_to_score() -> None:
    # Regression from obs_20260728_234332: Scored! Home @109.12 attached dunk
    # from possible dunk @109.53 with overlap=22 / hand=68 (old emit floor).
    # User confirmed there was no dunk — attach requires a stronger contact.
    det = RimHandContactDetector()
    det._last_contact_t = 109.53
    det._last_contact_overlap = 22
    det._last_contact_hand = 68
    assert det.recently_contacted(109.12)  # time window alone still matches
    assert not det.recently_contacted(109.12, for_attach=True)

    det._last_contact_overlap = 80
    det._last_contact_hand = 200
    assert det.recently_contacted(109.12, for_attach=True)


def test_flickering_glove_colour_across_frames_still_counts_as_dunk() -> None:
    # Regression: real dunk footage alternates which glove colour clears its
    # threshold from one ~0.14s sample to the next (fast motion / partial
    # occlusion), so requiring BOTH yellow and black in the exact same frame
    # silently missed genuine contact. Replayed pixel counts from an actual
    # missed-dunk session (obs_20260728_185742): the frame at 35.53 alone
    # only has yellow=8/black=22, but 35.39 just before it had yellow=37 —
    # the detector must remember that recent yellow instead of demanding it
    # again in the same frame as a strong black+overlap reading.
    det = RimHandContactDetector(confirm_frames=2, cooldown_sec=0.0)
    samples = [
        (34.98, 18087, 0, 16, 16),
        (35.11, 13803, 10, 0, 10),
        (35.25, 13752, 0, 5, 5),
        (35.39, 4727, 37, 209, 246),
        (35.53, 4032, 8, 22, 30),
    ]
    contact = None
    for sec, rim, yellow, black, overlap in samples:
        stats = {
            "rim_pixels": rim,
            "hand_pixels": 0,
            "yellow_near_rim": yellow,
            "black_near_rim": black,
            "overlap_pixels": overlap,
        }
        det._hit_streak = det._hit_streak + 1 if det._is_hit(stats) else 0
        if det._hit_streak >= det.confirm_frames:
            contact = sec
            break
    assert contact == 35.53


def test_rim_alone_ball_or_yellow_only_is_not_a_dunk() -> None:
    det = RimHandContactDetector(confirm_frames=1, cooldown_sec=0.0)
    rim_only = _blank()
    _paint_rim(rim_only)
    assert det.update(rim_only, 1.0, is_rgb=False) is None

    ball_only = _blank()
    _paint_ball(ball_only)
    assert det.update(ball_only, 2.0, is_rgb=False) is None

    yellow_only = _blank()
    _paint_rim(yellow_only)
    _paint_yellow_near_rim_only(yellow_only)
    assert det.update(yellow_only, 3.0, is_rgb=False) is None

    ball_and_hand = _blank()
    _paint_ball(ball_and_hand)
    _paint_hand_on_rim(ball_and_hand)
    assert det.update(ball_and_hand, 4.0, is_rgb=False) is None


def test_scored_home_after_rim_contact_grounds_as_dunk() -> None:
    reset_style_line_memory()
    raw = compose_result_evidence("Scored! Home", DUNK_CONTACT_EVIDENCE)
    assert ground_broadcast_text("Nice!", raw) == (
        "The player reaches rim height with the hand on the rim for a dunk and scores."
    )
    assert ground_broadcast_text("漂亮！", raw, language="zh") == "玩家灌籃得分。"


def test_stale_dunk_cue_must_not_steal_later_scores_via_pick_result_action() -> None:
    # Regression: dunk contact used to be appended into the LiveCC action deque,
    # so pick_result_action kept preferring it for every later Scored!.
    assert pick_result_action(
        [
            "The player pulls up for a mid-range jumper.",
            DUNK_CONTACT_EVIDENCE,
        ]
    ) == "The player pulls up for a mid-range jumper."


def test_dunk_evidence_constant_names_dunk_for_pickers() -> None:
    assert "dunk" in DUNK_CONTACT_EVIDENCE.lower()
