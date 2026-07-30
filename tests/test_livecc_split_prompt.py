from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miis_broadcast.core.prompt.prompt_manager import PromptManager
from miis_broadcast.core.models.gemini_broadcaster import _VIEW_RELATIONSHIP_CONTEXT


def test_basketball_single_view_prompt_matches_640x480_feed() -> None:
    prompt = PromptManager(ROOT / "configs" / "livecc_prompts.yml", sport="basketball").livecc_query()
    lower = " ".join(prompt.lower().split())
    assert "one live one-on-one basketball gameplay view" in lower
    assert "640x480" in lower
    assert "left-right" not in lower
    assert "the player" in lower and "the robot opponent" in lower
    assert "referee's final ruling" in lower
    assert "scored! away means the robot opponent scored" in lower
    assert "shot clock violation! home means the player ran out of shot clock" in lower
    assert "a dunk carries the ball up to rim height" in lower
    assert "fake / size-up (required when motion is visible)" in lower
    assert 'must include "fake" or "size-up"' in lower
    assert "with no rocking / jab / sway is not a fake" in lower
    assert "the feed can lag or drop frames" in lower
    assert "never guess between dunk, layup and jumper" in lower
    assert "missed shots matter (required wording)" in lower
    assert 'must include the word "misses"' in lower
    assert "never mention vr, headset, controller" in lower


def test_basketball_split_prompt_links_both_views_and_bans_device_story() -> None:
    prompt = PromptManager(ROOT / "configs" / "livecc_prompts.yml", sport="basketball").livecc_query_splitscreen()
    lower = " ".join(prompt.lower().split())
    assert "same one-on-one basketball moment" in lower
    assert "third-person gameplay view" in lower
    assert "first-person view" in lower
    assert 'call the left person and right hands "the player"' in lower
    assert 'call the other right avatar "the robot opponent"' in lower
    assert "referee's final ruling" in lower
    assert "scored! away means the robot opponent scored" in lower
    assert "out of bounds! away means the robot opponent sent the ball out" in lower
    assert "the player gets possession" in lower
    assert "without mentioning text, a banner, a screen or a referee" in lower
    assert "a dribbling robot avatar is the robot opponent" in lower
    assert "third-person player's synchronized dribble" in lower
    assert "use the home/away possession mapping above" in lower
    assert "no teammates or passes" in lower
    assert "never mention views, vr, equipment" in lower
    assert "a dunk is pushed down through the hoop from rim height" in lower
    assert 'must include "fake" or "size-up"' in lower
    assert "quiet holding with no rocking / jab is not a fake" in lower
    assert "the feed can lag or drop frames" in lower


def test_boxing_single_and_split_prompts() -> None:
    pm = PromptManager(ROOT / "configs" / "livecc_prompts.yml", sport="boxing")
    single = pm.livecc_query().lower()
    assert "one live one-on-one boxing gameplay view" in single
    assert "left-right" not in single
    split = pm.livecc_query_splitscreen().lower()
    assert "same boxing action" in split
    assert "third-person gameplay view" in split
    assert "first-person in-game view" in split


def test_gemini_receives_single_view_guardrail() -> None:
    lower = _VIEW_RELATIONSHIP_CONTEXT.lower()
    assert "one live basketball gameplay view" in lower
    assert "640x480" in lower
    assert "observer error" in lower
    assert "explicitly name the player or the robot opponent" in lower
    assert "center is time only" in lower
    assert "scored! away means the robot opponent scored" in lower
    assert "referee's final ruling" in lower
    assert "out of bounds! away means the robot opponent sent the ball out" in lower
    assert "the player gets possession" in lower
    assert "without mentioning text, a banner, a screen, or a referee" in lower
    assert "dribbling robot avatar means the robot opponent has possession" in lower
    assert "use its home/away mapping above for the awarded possession" in lower
    assert "keep the finishing move the caption actually reports" in lower
    assert "never upgrade it to a shot, a make, or a miss" in lower
    assert "when the caption shows that motion, the broadcast must use fake or size-up wording" in lower
    assert "never invent a fake for quiet holding or staring with no rock / jab / sway" in lower


def test_camera_infer_cfg_matches_livecc_paper_2fps() -> None:
    from miis_broadcast.server.session import _camera_infer_cfg

    cfg = _camera_infer_cfg()
    assert cfg["target_fps"] == 2.0
    assert cfg["window_sec"] == 1.0
    assert cfg["infer_interval"] == 1.0
    assert cfg["memory_reset_every"] == 24.0
