from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miis_broadcast.core.models.broadcast_grounding import (
    compose_result_evidence,
    ground_broadcast_text,
    pick_result_action,
    reset_style_line_memory,
    result_cue,
    strip_fake_moves,
)


def setup_function() -> None:
    """Each test starts with a clean spoken-line rotation memory."""
    reset_style_line_memory()
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
    assert ground_broadcast_text("Out", "Out of Bounds! Home") == (
        "The player sends the ball out of bounds, giving possession to the robot opponent."
    )
    assert ground_broadcast_text("Out", "Out of Bounds! Away") == (
        "The robot opponent sends the ball out of bounds, giving possession to the player."
    )


def test_confirmed_score_announces_the_updated_session_score() -> None:
    assert ground_broadcast_text(
        "Score!", "Scored! Home Score: Home 2, Away 1"
    ) == (
        "The player attacks the basket and scores. "
        "The score is player 2, robot opponent 1."
    )
    assert ground_broadcast_text(
        "得分！", "Scored! Away Score: Home 2, Away 2", language="zh"
    ) == "機器人對手攻向籃框並成功得分。目前比分：玩家 2，機器人對手 2。"


def test_result_cues_require_the_exact_banner_words_and_exclamation() -> None:
    assert result_cue("Scored! Home") == "score"
    assert result_cue("Out of Bounds! Away") == "out_of_bounds"
    for text in ("scores!", "Scored Home", "Out of Bound!", "out of bounds", "the ball scored"):
        assert result_cue(text) is None


def test_result_commentary_uses_prior_visible_action_context() -> None:
    # The confirmed sentence is built straight from what LiveCC actually
    # reported before the banner appeared - not a canned line - so the exact
    # pressure/drive detail LiveCC saw is preserved.
    score = compose_result_evidence(
        "Scored! Home", "The player releases a contested shot under defensive pressure."
    )
    assert ground_broadcast_text("A basket.", score) == (
        "The player releases a contested shot under defensive pressure and scores."
    )
    oob = compose_result_evidence(
        "Out of Bounds! Away", "The robot opponent drives into the lane."
    )
    assert ground_broadcast_text("Play stops.", oob) == (
        "The robot opponent drives into the lane, but the ball goes out of bounds "
        "and the player takes possession."
    )


def test_confirmed_score_reads_the_real_pre_shot_evidence_not_a_template() -> None:
    dunk = compose_result_evidence("Scored! Home", "The player rises for a slam dunk.")
    assert ground_broadcast_text("Nice finish!", dunk) == (
        "The player rises for a slam dunk and scores."
    )
    # zh has no reliable rule-based translation of the raw evidence, so it
    # falls back to the deterministic (non-random) finish template instead.
    assert ground_broadcast_text("漂亮！", dunk, language="zh") == "玩家灌籃得分。"

    layup = compose_result_evidence(
        "Scored! Away", "The robot opponent fingers the layup at the rim."
    )
    assert ground_broadcast_text("And in!", layup) == (
        "The robot opponent fingers the layup at the rim and scores."
    )
    assert ground_broadcast_text("進了！", layup, language="zh") == "機器人對手上籃得分。"

    jumper = compose_result_evidence(
        "Scored! Home", "The player pulls up for a mid-range jumper."
    )
    assert ground_broadcast_text("Good!", jumper) == (
        "The player pulls up for a mid-range jumper and scores."
    )
    assert ground_broadcast_text("好球！", jumper, language="zh") == "玩家跳投得分。"

    three = compose_result_evidence(
        "Scored! Away", "The robot opponent lets it fly from beyond the arc."
    )
    assert ground_broadcast_text("From deep!", three) == (
        "The robot opponent lets it fly from beyond the arc and scores."
    )
    assert ground_broadcast_text("三分！", three, language="zh") == "機器人對手三分命中。"

    # No prior evidence attached (bare banner): falls back to whatever finish
    # style Gemini's own draft names.
    assert ground_broadcast_text(
        "The player dunks it home!", "Scored! Home"
    ) == "The player throws down a dunk and scores."


def test_identical_evidence_yields_identical_score_text() -> None:
    # The output is a deterministic function of what LiveCC saw - not a dice
    # roll - so replaying the same evidence must always read the same way.
    dunk = compose_result_evidence("Scored! Home", "The player rises for a slam dunk.")
    first = ground_broadcast_text("Nice finish!", dunk)
    second = ground_broadcast_text("Nice finish!", dunk)
    assert first == second == "The player rises for a slam dunk and scores."


def test_different_evidence_yields_different_score_text() -> None:
    # Two different real plays should read differently because the wording
    # comes from the actual captured description, not a shared canned line.
    baseline_drive = compose_result_evidence(
        "Scored! Home", "The player drives down the baseline for the finish."
    )
    step_back = compose_result_evidence(
        "Scored! Home", "The player rises for a step-back jumper over the defender."
    )
    result_a = ground_broadcast_text("!", baseline_drive)
    result_b = ground_broadcast_text("!", step_back)
    assert result_a == "The player drives down the baseline for the finish and scores."
    assert result_b == "The player rises for a step-back jumper over the defender and scores."
    assert result_a != result_b


def test_a_faked_move_is_never_reported_as_the_finish() -> None:
    faked = compose_result_evidence(
        "Scored! Away", "The robot opponent fakes the jumper and attacks the rim."
    )
    assert ground_broadcast_text("In!", faked) == (
        "The robot opponent attacks the rim and scores."
    )
    # The move that actually goes in is the one named last. No prior evidence
    # is attached here, so this falls back to the finish read from Gemini's
    # own draft rather than the (unavailable) real pre-shot caption.
    assert ground_broadcast_text(
        "The player fakes the three, then dunks it home!", "Scored! Home"
    ) == "The player throws down a dunk and scores."
    fake_then_layup = compose_result_evidence(
        "Scored! Home", "The player pump fakes, then drives in for the layup."
    )
    assert ground_broadcast_text("Good!", fake_then_layup) == (
        "The player drives in for the layup and scores."
    )
    assert "jumper" not in strip_fake_moves("The player pump-fakes a jumper.").lower()
    assert "dunk" not in strip_fake_moves("The player fakes a dunk.").lower()
    stripped_zh = strip_fake_moves("玩家用假動作晃開防守者。")
    assert "假動作" not in stripped_zh and "晃開" not in stripped_zh


def test_result_action_prefers_the_attempt_over_the_freshest_size_up() -> None:
    # LiveCC lags the raw frames, so the newest caption at banner time is often
    # only the wind-up while the real attempt sits one caption earlier.
    assert pick_result_action(
        [
            "The player rocks the ball side to side against the defender.",
            "The player rises and throws down a dunk.",
        ]
    ) == "The player rises and throws down a dunk."
    assert pick_result_action(
        [
            "The player sizes up the robot opponent.",
            "The player releases a contested shot.",
        ]
    ) == "The player releases a contested shot."
    assert pick_result_action(
        ["The player dribbles at the perimeter.", "The robot opponent backpedals."]
    ) == "The player dribbles at the perimeter."
    assert pick_result_action([]) == ""


def test_score_pick_prefers_dunk_over_prior_missed_jumper() -> None:
    # Confirmed Scored! must not lock onto "jump shot but misses" when a dunk
    # caption is also in the recent window (common LiveCC lag pattern).
    assert pick_result_action(
        [
            "and dunks it home.",
            "jump shot but misses the rim",
        ],
        for_score=True,
    ) == "and dunks it home."
    assert pick_result_action(
        ["jump shot but misses the rim"],
        for_score=True,
    ) == ""
    assert "跳投" not in ground_broadcast_text(
        "進了！",
        compose_result_evidence("Scored! Home", "jump shot but misses the rim"),
        language="zh",
    )
    assert ground_broadcast_text(
        "進了！",
        compose_result_evidence("Scored! Home", "and dunks it home."),
        language="zh",
    ) == "玩家灌籃得分。"


def test_pressure_out_of_bounds_names_the_attacking_side_naturally() -> None:
    raw = compose_result_evidence(
        "Out of Bounds! Home", "The player attacks against close defensive pressure."
    )
    assert ground_broadcast_text("Play stops.", raw) == (
        "The player attacks under pressure, but the ball goes out of bounds "
        "and the robot opponent takes possession."
    )
    assert ground_broadcast_text("停止比賽。", raw, language="zh") == (
        "玩家在壓力下進攻，球出了界，球權轉交機器人對手。"
    )


def test_bare_result_cue_does_not_guess_an_actor() -> None:
    assert ground_broadcast_text("The player scores.", "Scored!") == "A basket is confirmed."
    assert ground_broadcast_text("The player gets it.", "Out of Bounds!") == "The ball goes out of bounds and possession resets."
    assert ground_broadcast_text(
        "The player says the ball went out of bounds.",
        "The player drives and the ball goes out of bounds.",
    ) == "The player protects the ball as the robot opponent applies pressure."


def test_missed_shot_is_reported_instead_of_neutral_filler() -> None:
    # No banner ever confirms a miss, but the broadcast draft is already the
    # real, actor-grounded LiveCC description of what happened - so we speak
    # it as-is instead of inventing or picking a canned line for it.
    assert ground_broadcast_text(
        "The player pulls up for a jumper but misses.",
        "The player pulls up for a jumper but misses the shot.",
    ) == "The player pulls up for a jumper but misses."
    assert ground_broadcast_text(
        "The robot opponent shoots but the shot rims out.",
        "The robot opponent shoots but the shot rims out.",
    ) == "The robot opponent shoots but the shot rims out."
    assert ground_broadcast_text(
        "玩家出手但沒進。", "玩家出手但沒進。", language="zh"
    ) == "玩家出手但沒進。"


def test_unconfirmed_make_claim_stays_neutral_until_banner_confirms() -> None:
    # No Scored! banner yet: an unconfirmed "makes it" claim must not be
    # announced as a made basket ahead of the referee.
    assert ground_broadcast_text(
        "The player drives and makes the shot.",
        "The player drives and makes the shot.",
    ) == "The player attacks and releases a shot toward the basket."


def test_quiet_possession_must_not_invent_a_fake() -> None:
    # Mere holding / staring is not a fake — only motion cues (rock / size-up /
    # jab / pump-fake) justify 假動作 wording.
    assert ground_broadcast_text(
        "玩家持球冷靜觀察，尋找進攻切入點。",
        "The player holds the ball at the perimeter.",
        language="zh",
    ) == "玩家持球冷靜觀察，尋找進攻切入點。"
    assert ground_broadcast_text(
        "雙方在場上對峙，正在尋找進攻節奏。",
        "The player and robot opponent face each other.",
        language="zh",
    ) == "雙方在場上對峙，正在尋找進攻節奏。"
    stripped = ground_broadcast_text(
        "玩家持球用假動作冷靜觀察。",
        "The player holds the ball looking for an opening.",
        language="zh",
    )
    assert "假動作" not in stripped
    en = ground_broadcast_text(
        "The player fakes while holding the ball.",
        "The player holds the ball at the perimeter.",
    )
    assert "fake" not in en.lower()
    # Scoring outcomes must not be rewritten into fake talk.
    assert "假動作" not in ground_broadcast_text(
        "玩家攻向籃框並成功得分。", "Scored! Home", language="zh"
    )


def test_livecc_fake_evidence_keeps_fake_wording() -> None:
    assert ground_broadcast_text(
        "玩家持球用假動作左右晃對手。",
        "The player rocks the ball side to side against the robot opponent.",
        language="zh",
    ) == "玩家持球用假動作左右晃對手。"
    assert "fake" in ground_broadcast_text(
        "The player fakes side to side.",
        "The player pump-fakes then hesitates.",
    ).lower()


def test_weak_size_up_evidence_does_not_force_fake_wording() -> None:
    # Regression: "sizes up" / rocks / sways alone used to FORCE-inject 假動作
    # into every draft, even ones Gemini deliberately wrote without it. Since
    # VLMs use these words loosely for routine ball-handling in front of a
    # defender (not just deliberate fakes), this made 假動作 show up on nearly
    # every background line during ordinary standoffs. Weak evidence alone
    # must leave the draft exactly as Gemini wrote it.
    out = ground_broadcast_text(
        "玩家持球冷靜觀察。",
        "The player sizes up the robot opponent.",
        language="zh",
    )
    assert out == "玩家持球冷靜觀察。"
    assert "假動作" not in out
    # But an existing fake mention in the draft is left alone, not stripped.
    kept = ground_broadcast_text(
        "玩家持球用假動作左右晃對手。",
        "The player sizes up against the robot opponent.",
        language="zh",
    )
    assert kept == "玩家持球用假動作左右晃對手。"
    # Gemini may see rocking in frames even when LiveCC only said "dribbles".
    assert "假動作" in ground_broadcast_text(
        "玩家持球用假動作左右晃對手。",
        "The player dribbles at the perimeter.",
        language="zh",
    )


def test_strong_fake_evidence_still_forces_injection() -> None:
    # Unlike bare "sizes up"/rock/sway, an explicit fake/jab-step/hesitation
    # caption is unambiguous, so it still forces 假動作 into a plain draft.
    out = ground_broadcast_text(
        "玩家持球冷靜觀察。",
        "The player hesitates with a jab step.",
        language="zh",
    )
    assert "假動作" in out or "試探步" in out
    en = ground_broadcast_text(
        "The player holds the ball at the perimeter.",
        "The player pump-fakes the jumper.",
    )
    assert "fake" in en.lower()


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


def test_confirmed_p1_banners_vary_lightly_by_broadcast_style() -> None:
    dunk = compose_result_evidence("Scored! Home", "The player rises for a slam dunk.")
    assert ground_broadcast_text("漂亮！", dunk, language="zh") == "玩家灌籃得分。"
    assert ground_broadcast_text(
        "漂亮！", dunk, language="zh", style="hype"
    ) == "哇！玩家灌籃得分，太炸裂啦！"
    assert ground_broadcast_text(
        "漂亮！", dunk, language="zh", style="calm"
    ) == "玩家完成灌籃得分。"
    assert ground_broadcast_text(
        "漂亮！", dunk, language="zh", style="trash_talk"
    ) == "玩家灌籃得分，這球沒得擋！"

    assert ground_broadcast_text(
        "Out", "Out of Bounds! Home", language="zh"
    ) == "玩家將球弄出界，球權轉交機器人對手。"
    assert ground_broadcast_text(
        "Out", "Out of Bounds! Home", language="zh", style="hype"
    ) == "出界！玩家把球弄出界，球權交出去了！"
    assert ground_broadcast_text(
        "Clock", "Shot Clock Violation! Away", language="zh", style="hype"
    ) == "時間到！機器人對手進攻時間到點，球權換邊！"
    assert ground_broadcast_text(
        "Score!", "Scored! Away", language="zh", style="hype"
    ) == "進了！機器人對手攻向籃框，得分進帳！"


def test_confirmed_p1_banner_lines_rotate_without_repeating() -> None:
    """Same event type must rotate wording and not speak an identical line twice."""
    lines = [
        ground_broadcast_text(
            "Out", "Out of Bounds! Home", language="zh", style="hype"
        )
        for _ in range(3)
    ]
    assert lines[0] != lines[1]
    assert lines[1] != lines[2]
    assert len(set(lines)) == 3
    for line in lines:
        assert "玩家" in line and "出界" in line and "球權" in line

    dunk = compose_result_evidence("Scored! Home", "The player rises for a slam dunk.")
    dunk_lines = [
        ground_broadcast_text("漂亮！", dunk, language="zh", style="hype")
        for _ in range(3)
    ]
    assert dunk_lines[0] != dunk_lines[1]
    assert len(set(dunk_lines)) == 3
    for line in dunk_lines:
        assert "玩家" in line and "灌籃" in line and "得分" in line
