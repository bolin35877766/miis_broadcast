from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miis_broadcast.core.prompt.prompt_manager import PromptManager
from miis_broadcast.core.models.gemini_broadcaster import _VIEW_RELATIONSHIP_CONTEXT


def test_basketball_split_prompt_links_both_views_and_bans_device_story() -> None:
    prompt = PromptManager(ROOT / "configs" / "livecc_prompts.yml", sport="basketball").livecc_query_splitscreen()
    lower = " ".join(prompt.lower().split())
    assert "same one-on-one basketball moment" in lower
    assert "third-person gameplay view" in lower
    assert "first-person view" in lower
    assert 'call the left person and right hands "the player"' in lower
    assert 'call the other right avatar "the robot opponent"' in lower
    assert "ignore names, numbers, the timer and the persistent scoreboard" in lower
    assert "referee's final ruling" in lower
    assert "scored! away means the robot opponent scored" in lower
    assert "out of bounds! away means the robot opponent sent the ball out" in lower
    assert "without mentioning text, a banner, a screen or a referee" in lower
    assert "a dribbling robot avatar is the robot opponent" in lower
    assert "third-person player's synchronized dribble" in lower
    assert "never derive next possession from home/away" in lower
    assert "no teammates or passes" in lower
    assert "never mention views, vr, equipment" in lower


def test_boxing_split_prompt_has_same_view_invariant() -> None:
    prompt = PromptManager(ROOT / "configs" / "livecc_prompts.yml", sport="boxing").livecc_query_splitscreen()
    lower = prompt.lower()
    assert "same boxing action" in lower
    assert "third-person gameplay view" in lower
    assert "first-person in-game view" in lower
    assert 'are "the player"' in lower
    assert 'is "the opponent"' in lower


def test_gemini_receives_the_same_relationship_guardrail() -> None:
    lower = _VIEW_RELATIONSHIP_CONTEXT.lower()
    assert "synchronized third-person gameplay view" in lower
    assert "observer error" in lower
    assert "right as the authoritative" in lower
    assert "explicitly name the player or the robot opponent" in lower
    assert "center is time only" in lower
    assert "scored! away means the robot opponent scored" in lower
    assert "referee's final ruling" in lower
    assert "out of bounds! away means the robot opponent sent the ball out" in lower
    assert "without mentioning text, a banner, a screen, or a referee" in lower
    assert "dribbling robot avatar means the robot opponent has possession" in lower
    assert "third-person player's synchronized dribble" in lower
    assert "never derive the next possession from home/away" in lower
