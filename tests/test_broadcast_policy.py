from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from broadcast_policy import (
    BroadcastPacer,
    compose_result_evidence,
    ground_broadcast_text,
    result_cue,
)


def test_result_cues_require_explicit_visual_text() -> None:
    assert result_cue('The screen displays "Scored! Home".') == "score"
    assert result_cue('Text appears saying "Out of Bounds! Away".') == "out_of_bounds"
    assert result_cue("The ball appears to go through the hoop.") is None
    assert result_cue("The model claims the ball goes out of bounds.") is None
    assert result_cue("Scored Home") is None
    assert result_cue("Out of Bound!") is None
    assert result_cue("The player scores!") is None


def test_result_commentary_includes_grounded_lead_in_action() -> None:
    raw = compose_result_evidence(
        "Scored! Away", "The robot opponent shoots while closely guarded."
    )
    assert ground_broadcast_text("It goes in.", raw) == (
        "Under defensive pressure, the robot opponent releases the shot and scores."
    )
    oob = compose_result_evidence(
        "Out of Bounds! Away", "The robot opponent attacks under close pressure."
    )
    assert ground_broadcast_text("Play stops.", oob) == (
        "The robot opponent attacks under pressure, but the ball goes out of bounds "
        "and the player takes possession."
    )


def test_score_cue_includes_current_session_score() -> None:
    assert ground_broadcast_text(
        "The player finishes.", "Scored! Home Score: Home 3, Away 2"
    ).endswith("The score is player 3, robot opponent 2.")


def test_unsupported_score_is_downgraded_to_attempt() -> None:
    assert "releases a shot" in ground_broadcast_text(
        "Nothing but net, a clinical finish!", "The player raises the ball."
    )
    assert ground_broadcast_text(
        "Away scores!", 'The result banner displays "Scored! Away".'
    ) == "The robot opponent attacks the basket and scores."


def test_explicit_result_overrides_hallucinated_commentary() -> None:
    assert ground_broadcast_text(
        "The shot swishes through the hoop.", 'Text appears saying "Out of Bounds! Away".'
    ) == "The robot opponent sends the ball out of bounds, giving possession to the player."
    text = "The defender blocks it and retrieves the rebound."
    assert ground_broadcast_text(text, "The player raises the ball.") == text


def test_possession_and_movement_details_are_preserved() -> None:
    actions = (
        "The player dribbles, cuts into the lane, then retreats to the perimeter.",
        "The opponent steals the ball and carries it back out to reset.",
        "The player secures the rebound as possession changes.",
    )
    expected = (
        actions[0],
        "The robot opponent steals the ball and carries it back out to reset.",
        actions[2],
    )
    for text, grounded in zip(actions, expected):
        assert ground_broadcast_text(text, text) == grounded


def test_latest_explicit_result_wins_inside_a_window() -> None:
    raw = 'First, "Scored! Home" appears. Later, text says "Out of Bounds! Away".'
    assert result_cue(raw) == "out_of_bounds"
    assert ground_broadcast_text("Wrong result", raw) == (
        "The robot opponent sends the ball out of bounds, giving possession to the player."
    )


def test_pacer_keeps_cues_and_throttles_routine_lines() -> None:
    pacer = BroadcastPacer(routine_interval_sec=6.0, cue_repeat_sec=3.0)
    assert pacer.should_keep(0.0, "The player dribbles.")
    assert not pacer.should_keep(2.0, "The defender closes out.")
    assert pacer.should_keep(3.0, 'Text: "Out of Bounds! Home"')
    assert not pacer.should_keep(4.0, 'Text: "Out of Bounds! Home"')
    assert pacer.should_keep(7.0, 'Text: "Out of Bounds! Home"')
