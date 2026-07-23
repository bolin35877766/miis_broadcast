from pathlib import Path
import sys

import cv2


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from video_result_cues import detect_result_cues
from run_commentary_pipeline import _attach_recent_action, _detect_result_banner_records
from miis_broadcast.core.models.result_banner_tracker import ResultBannerTracker


def test_test_merged_result_sequence() -> None:
    video = Path(__file__).resolve().parents[1] / "examples" / "test_merged.mp4"
    cues = detect_result_cues(video)
    assert [cue.kind for cue in cues] == [
        "out_of_bounds", "out_of_bounds", "out_of_bounds", "out_of_bounds",
        "out_of_bounds", "score", "out_of_bounds", "score", "out_of_bounds",
        "out_of_bounds", "out_of_bounds", "score", "out_of_bounds", "score",
        "out_of_bounds", "out_of_bounds", "out_of_bounds", "score", "score",
        "score", "out_of_bounds", "score", "out_of_bounds",
    ]


def test_tracker_reads_home_and_away_from_verified_video_frames() -> None:
    video = Path(__file__).resolve().parents[1] / "examples" / "test_merged.mp4"
    cap = cv2.VideoCapture(str(video))
    try:
        for timestamp, expected in (
            (109.8, "away"),
            (201.5, "home"),
            (293.8, "away"),
            (307.3, "home"),
        ):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()
            assert ok
            gameplay = frame[:, frame.shape[1] // 2 :]
            assert ResultBannerTracker._detect_side(gameplay, is_rgb=False) == expected
    finally:
        cap.release()


def test_eval_pipeline_uses_gui_tracker_with_fixed_score_identity() -> None:
    video = Path(__file__).resolve().parents[1] / "examples" / "test_merged.mp4"
    records = _detect_result_banner_records(video)
    scores = [(round(item["start"], 1), item["result_side"]) for item in records if item["result_kind"] == "score"]
    assert scores == [
        (69.2, "home"),
        (109.7, "away"),
        (166.9, "away"),
        (201.3, "home"),
        (260.2, "home"),
        (293.6, "away"),
        (306.9, "home"),
        (342.2, "home"),
    ]


def test_eval_banner_carries_only_recent_preceding_action() -> None:
    sources = [
        {"event": "livecc_segment", "index": 1, "end": 10.0, "livecc_text": "old action"},
        {
            "event": "livecc_segment",
            "index": 2,
            "end": 16.0,
            "livecc_text": "The player shoots under defensive pressure.",
        },
        {"event": "livecc_segment", "index": 3, "end": 21.0, "livecc_text": "future action"},
    ]
    banner = {
        "event": "result_banner_cue",
        "start": 18.0,
        "livecc_text": "Scored! Home",
        "livecc": {"metadata": {"raw": "Scored! Home"}},
    }
    _attach_recent_action(banner, sources)
    assert banner["livecc_text"].endswith("\nScored! Home")
    assert "defensive pressure" in banner["livecc_text"]
    assert "future action" not in banner["livecc_text"]
    assert banner["context_source_index"] == 2
