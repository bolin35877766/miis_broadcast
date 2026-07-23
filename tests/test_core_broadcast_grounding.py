from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miis_broadcast.core.models.broadcast_grounding import (
    compose_result_evidence,
    ground_broadcast_text,
    result_cue,
)
from miis_broadcast.core.models.gemini_broadcaster import (
    _build_gemini_contents,
    _override_actor_subject,
)


def test_grounding_uses_active_output_language() -> None:
    assert ground_broadcast_text("Away scores!", "Scored! Away", language="zh") == "機器人對手攻向籃框並成功得分。"
    assert ground_broadcast_text("Home scores!", "Scored! Home", language="zh") == "玩家攻向籃框並成功得分。"
    assert ground_broadcast_text("球進了！", "The player shoots.", language="zh") == "球員攻向籃框並出手。"


def test_result_banner_maps_home_to_player_and_away_to_robot() -> None:
    assert ground_broadcast_text("Score!", "Scored!\nHome") == "The player attacks the basket and scores."
    assert ground_broadcast_text("Score!", "Scored! Away") == "The robot opponent attacks the basket and scores."
    assert ground_broadcast_text("Out", "Out of Bounds! Home") == "The player sends the ball out of bounds."
    assert ground_broadcast_text("Out", "Out of Bounds! Away") == "The robot opponent sends the ball out of bounds."


def test_result_cues_require_the_exact_banner_words_and_exclamation() -> None:
    assert result_cue("Scored! Home") == "score"
    assert result_cue("Out of Bounds! Away") == "out_of_bounds"
    for text in ("scores!", "Scored Home", "Out of Bound!", "out of bounds", "the ball scored"):
        assert result_cue(text) is None


def test_result_commentary_uses_prior_visible_action_context() -> None:
    score = compose_result_evidence(
        "Scored! Home", "The player releases a contested shot under defensive pressure."
    )
    assert ground_broadcast_text("A basket.", score) == (
        "Under defensive pressure, the player releases the shot and scores."
    )
    oob = compose_result_evidence(
        "Out of Bounds! Away", "The robot opponent drives into the lane."
    )
    assert ground_broadcast_text("Play stops.", oob) == (
        "The robot opponent drives into the lane, and the ball goes out of bounds."
    )


def test_pressure_out_of_bounds_names_the_attacking_side_naturally() -> None:
    raw = compose_result_evidence(
        "Out of Bounds! Home", "The player attacks against close defensive pressure."
    )
    assert ground_broadcast_text("Play stops.", raw) == (
        "The player attacks under pressure, and the ball goes out of bounds."
    )
    assert ground_broadcast_text("停止比賽。", raw, language="zh") == (
        "玩家在壓力下進攻，球出了界。"
    )


def test_bare_result_cue_does_not_guess_an_actor() -> None:
    assert ground_broadcast_text("The player scores.", "Scored!") == "A basket is confirmed."
    assert ground_broadcast_text("The player gets it.", "Out of Bounds!") == "The ball goes out of bounds and possession resets."
    assert ground_broadcast_text(
        "The player says the ball went out of bounds.",
        "The player drives and the ball goes out of bounds.",
    ) == "The player protects the ball as the robot opponent applies pressure."


def test_grounding_preserves_non_outcome_action() -> None:
    text = "The ballhandler dribbles against close pressure."
    assert ground_broadcast_text(text, "The player dribbles.") == text


def test_grounding_preserves_drives_resets_and_possession_changes() -> None:
    actions = (
        "The player crosses over, cuts into the lane, then retreats to reset.",
        "The opponent steals the ball and carries it back toward the perimeter.",
        "The player blocks the attempt and secures the rebound.",
        "The turnover changes possession as the opponent moves into the backcourt.",
    )
    expected = (
        actions[0],
        "The robot opponent steals the ball and carries it back toward the perimeter.",
        actions[2],
        "The turnover changes possession as the robot opponent moves into the backcourt.",
    )
    for text, grounded in zip(actions, expected):
        assert ground_broadcast_text(text, text) == grounded


def test_one_on_one_wording_preserves_retreat_and_explicit_robot_role() -> None:
    assert ground_broadcast_text(
        "Player penetrates the paint, then kicks it back out.",
        "The player cuts in and retreats.",
    ) == "The player penetrates the paint, then retreats to the perimeter."
    assert ground_broadcast_text(
        "The robot swipes the ball and retreats.",
        "The opponent steals the ball.",
    ) == "The robot opponent swipes the ball and retreats."
    assert ground_broadcast_text(
        "The player sizes up test_bot1 while the opponent waits.",
        "The player faces test_bot1.",
    ) == "The player sizes up the robot opponent while the robot opponent waits."


def test_gemini_can_receive_authoritative_actor_frames() -> None:
    text_only = _build_gemini_contents({"metadata": {"raw": "play"}}, "prompt")
    assert text_only == ["prompt"]
    multimodal = _build_gemini_contents(
        {"metadata": {"raw": "play", "actor_frames_jpeg": [b"one", b"two", b"three", b"four"]}},
        "prompt",
    )
    assert len(multimodal) == 4
    assert "authoritative right-side" in multimodal[0].lower()


def test_authoritative_actor_overrides_only_the_sentence_subject() -> None:
    assert _override_actor_subject(
        "The player drives past the defender.", "robot_opponent"
    ) == "The robot opponent drives past the defender."
    assert _override_actor_subject(
        "The robot opponent retreats from the player.", "player"
    ) == "The player retreats from the robot opponent."
    assert _override_actor_subject(
        "The player sizes up the robot opponent.", "robot_opponent"
    ) == "The robot opponent sizes up the player."
    assert _override_actor_subject("The player dribbles.", "unclear") == "The player dribbles."


def test_ui_meta_commentary_keeps_confirmed_robot_actor_without_reading_clock() -> None:
    assert ground_broadcast_text(
        "The clock ticks down as the robot opponent holds possession.",
        "The robot opponent dribbles toward the player.",
    ) == "The robot opponent protects the ball as the player applies pressure."


def test_teamwork_hallucination_becomes_explicit_one_on_one_roles() -> None:
    assert ground_broadcast_text(
        "The player passes to an open teammate.", "The player raises the ball."
    ) == "The player protects the ball as the robot opponent applies pressure."
    assert ground_broadcast_text(
        "The opponent dishes it to a teammate.", "The opponent moves toward the player."
    ) == "The robot opponent protects the ball as the player applies pressure."
    assert ground_broadcast_text(
        "The opponent feeds it back into the player's hands.",
        "The opponent dishes the ball back to the player.",
    ) == "The robot opponent protects the ball as the player applies pressure."
