from pathlib import Path
import sys
from collections import deque
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miis_broadcast.workers.gemini import GeminiBackgroundWorker
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
