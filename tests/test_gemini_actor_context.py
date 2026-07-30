from pathlib import Path
import sys
from collections import deque
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miis_broadcast.workers.gemini import GeminiBackgroundWorker
from miis_broadcast.core.models.gemini_broadcaster import _ground_priority
from miis_broadcast.gui import MainWindow


def test_background_context_keeps_latest_actor_frames_with_caption_text() -> None:
    worker = GeminiBackgroundWorker(lambda: 0.0)
    worker.update_context({
        "event": "raw_description",
        "metadata": {"raw": "The player dribbles.", "actor_frames_jpeg": [b"a", b"b", b"c"]},
    })
    worker.update_context({
        "event": "raw_description",
        "metadata": {"raw": "The opponent reaches for the ball.", "actor_frames_jpeg": [b"d", b"e", b"f"]},
    })
    context = worker._build_context()
    assert context["metadata"]["raw"] == (
        "The player dribbles. The opponent reaches for the ball."
    )
    assert context["metadata"]["actor_frames_jpeg"] == [b"d", b"e", b"f"]


def test_gui_attaches_nearest_frames_only_to_routine_action() -> None:
    fake = SimpleNamespace(
        _actor_frame_cache=deque([(1.0, b"a"), (2.0, b"b"), (3.0, b"c"), (4.0, b"d")])
    )
    action = MainWindow._with_actor_frames(
        fake,
        {"event": "raw_description", "metadata": {"raw": "The player dribbles."}},
        1.0,
        3.0,
        "The player dribbles.",
    )
    assert action["metadata"]["actor_frames_jpeg"] == [b"a", b"b", b"c"]
    result = MainWindow._with_actor_frames(
        fake,
        {"event": "raw_description", "metadata": {"raw": "Scored! Home"}},
        1.0,
        3.0,
        "Scored! Home",
    )
    assert "actor_frames_jpeg" not in result["metadata"]


def test_priority_policy_speaks_p4_but_keeps_p5_silent() -> None:
    p4 = {"priority": 4, "broadcast_text": "The player resets.", "should_speak": True}
    p5 = {"priority": 5, "broadcast_text": "Both players wait.", "should_speak": False}
    assert MainWindow._broadcast_tts_allowed(None, p4)
    assert not MainWindow._broadcast_tts_allowed(None, p5)
    # A stale/incorrect should_speak flag must not bypass the hard P5 guard.
    assert not MainWindow._broadcast_tts_allowed(None, dict(p5, should_speak=True))


def test_p5_is_reserved_for_off_court_or_unrelated_content() -> None:
    assert _ground_priority(5, "The player and robot opponent stand in a standoff.") == 4
    assert _ground_priority(5, "The ballhandler waits at the perimeter.") == 4
    assert _ground_priority(5, "The crowd is cheering away from the court.") == 5
    assert _ground_priority(5, "A person adjusts equipment in the room.") == 5
    assert _ground_priority(5, "An unrelated indoor scene.") == 5


def test_fast_dedup_and_out_of_bounds_priority_are_configured() -> None:
    assert MainWindow._FAST_BLADE_DEDUP_WINDOW_S == 3.0
    assert MainWindow._scan_priority("Out of Bounds! Away") == 1
    root = Path(__file__).resolve().parents[1]
    assert "dedup_window_s: 3.0" in (root / "configs" / "app.yml").read_text(encoding="utf-8")
    prompts = (root / "configs" / "system_prompts.yml").read_text(encoding="utf-8")
    assert "scoring play, out-of-bounds ruling" in prompts
    assert "得分、出界判決" in prompts
    assert prompts.count("size-up fake / standoff") == 4
    assert prompts.count("假動作拉鋸") == 4
    assert prompts.count("off-court activity, crowd/cheering only") == 4
    assert prompts.count("場外活動、只有觀眾歡呼") == 4
    assert "假動作規則：" in prompts
    assert "FAKE RULE:" in prompts
    assert "單純持球觀察、對峙、尋找切入點（沒有晃動）不要說假動作" in prompts
    assert "Quiet holding with no rock or jab" in prompts
    assert "LiveCC shows rocking / size-up / jab / pump fake" in prompts


def test_referee_cue_wins_over_negative_scoring_lead_in() -> None:
    assert MainWindow._scan_priority(
        "Previous visible action: no points being scored.\nOut of Bounds! Home"
    ) == 1
    assert MainWindow._scan_priority("The shot does not go through.") == 2
    assert MainWindow._scan_priority("The layup rims out — no good.") == 2
    assert MainWindow._scan_priority("The player misses the jumper.") == 2
    assert MainWindow._scan_priority("No points are scored on the attempt.") == 3
    assert MainWindow._scan_priority("Scored! Away") == 1


def test_livecc_narrative_score_words_are_p2_not_hard_p1() -> None:
    """LiveCC prose must not hard-interrupt; only exact banners are P1."""
    assert MainWindow._scan_priority(
        "The player dribbles past the half-court line. The player ... scores."
    ) == 2
    assert MainWindow._scan_priority(
        "they make an easy dunk with no one around!"
    ) == 2
    assert MainWindow._scan_priority(
        "The player dunks on the robot opponent in front of the home bench."
    ) == 2
    assert MainWindow._scan_priority("Robot opponent makes a shot at half-court") == 2
    assert MainWindow._scan_priority(
        "The third -person player scores away on this shot attempt from the top ..."
    ) == 2
    assert MainWindow._scan_priority("Home scores!") == 2
    assert MainWindow._scan_priority("Scored! Home") == 1
    assert MainWindow._scan_priority("Out of Bounds! Home") == 1


def test_a_fake_is_not_treated_as_a_scoring_play() -> None:
    assert MainWindow._scan_priority(
        "The player fakes a dunk and resets at the perimeter."
    ) == 3
    assert MainWindow._scan_priority(
        "The player pump-fakes the shot while the robot opponent stays down."
    ) == 3
    assert MainWindow._scan_priority(
        "The player fakes the jumper, then dunks it home."
    ) == 2


class _CounterSignal:
    def __init__(self) -> None:
        self.calls = 0

    def emit(self, *_args) -> None:
        self.calls += 1


class _PauseCounter:
    def __init__(self) -> None:
        self.calls = 0

    def pause(self) -> None:
        self.calls += 1


def test_repeated_guarded_p1_does_not_pause_background_again() -> None:
    background = _PauseCounter()
    confirmed = _CounterSignal()
    routed = []
    fake = SimpleNamespace(
        _use_gemini=True,
        _ensure_log_dir=lambda: None,
        _recent_basketball_actions=deque(),
        _with_actor_frames=lambda data, *_args: data,
        _pending_livecc_fragment=None,
        _pending_score_banner=None,
        _last_banner_kind_side=None,
        _last_banner_video_t=-1e9,
        _BANNER_ECHO_COOLDOWN_S=4.0,
        _scan_priority=MainWindow._scan_priority,
        _write_log=lambda *_args: None,
        _fmt_time=lambda value: f"{value:.1f}",
        _FAST_BLADE_DEDUP_WINDOW_S=3.0,
        livecc_log_file=Path("/tmp/unused.log"),
        mode="camera",
        _is_duplicate_tts=lambda *_args, **_kwargs: False,
        _is_p1_audio_active=lambda: True,
        _p1_hard_interrupt=lambda already, **_kwargs: routed.append(("interrupt", already)),
        _fast_blade_enqueue_gemini=lambda *_args, **kwargs: routed.append(
            ("enqueue", kwargs["already_p1"])
        ),
        gemini_bg_worker=background,
        signal_p1_confirmed=confirmed,
    )
    MainWindow._route_segment(
        fake, 10.0, 11.0, {"event": "raw_description", "metadata": {"raw": "Scored! Home"}}
    )
    assert background.calls == 0
    assert confirmed.calls == 0
    assert routed == [("interrupt", True), ("enqueue", True)]


def test_livecc_banner_echo_without_side_is_suppressed() -> None:
    # Regression: after announcing Shot Clock Violation! Home, LiveCC often
    # re-says "Ahh! Shot clock violation!" with no Home/Away. Old echo logic
    # required an exact (kind, side) match, so None != "home" and the same
    # event was voiced twice.
    routed = []
    fake = SimpleNamespace(
        _use_gemini=True,
        _ensure_log_dir=lambda: None,
        _recent_basketball_actions=deque(),
        _with_actor_frames=lambda data, *_args: data,
        _pending_livecc_fragment=None,
        _pending_score_banner=None,
        _last_banner_kind_side=("shot_clock_violation", "home"),
        _last_banner_video_t=27.23,
        _BANNER_ECHO_COOLDOWN_S=4.0,
        _scan_priority=MainWindow._scan_priority,
        _write_log=lambda *_args: None,
        _fmt_time=lambda value: f"{value:.1f}",
        _FAST_BLADE_DEDUP_WINDOW_S=3.0,
        livecc_log_file=Path("/tmp/unused.log"),
        mode="camera",
        _is_duplicate_tts=lambda *_args, **_kwargs: False,
        _is_p1_audio_active=lambda: False,
        _p1_hard_interrupt=lambda *_args, **_kwargs: routed.append("interrupt"),
        _fast_blade_enqueue_gemini=lambda *_args, **_kwargs: routed.append("enqueue"),
        gemini_bg_worker=_PauseCounter(),
        signal_p1_confirmed=_CounterSignal(),
        _rim_hand_detector=SimpleNamespace(recently_contacted=lambda *_a, **_k: False),
    )
    MainWindow._route_segment(
        fake,
        27.45,
        28.45,
        {"event": "raw", "metadata": {"raw": "Ahh! Shot clock violation! ..."}},
    )
    assert routed == []
