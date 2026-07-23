"""Evidence-grounded outcome wording for basketball commentary."""

from __future__ import annotations

import re


PLAYER_SCORE_TEXT = "The player attacks the basket and scores."
OPPONENT_SCORE_TEXT = "The robot opponent attacks the basket and scores."
UNKNOWN_SCORE_TEXT = "A basket is confirmed."
PLAYER_OOB_CALL_TEXT = "The player sends the ball out of bounds, giving possession to the robot opponent."
OPPONENT_OOB_CALL_TEXT = "The robot opponent sends the ball out of bounds, giving possession to the player."
UNKNOWN_OUT_OF_BOUNDS_TEXT = "The ball goes out of bounds and possession resets."
NEUTRAL_SHOT_TEXT = "The player attacks and releases a shot toward the basket."
PLAYER_SCORE_TEXT_ZH = "玩家攻向籃框並成功得分。"
OPPONENT_SCORE_TEXT_ZH = "機器人對手攻向籃框並成功得分。"
UNKNOWN_SCORE_TEXT_ZH = "畫面確認這次進攻得分。"
PLAYER_OOB_CALL_TEXT_ZH = "玩家將球弄出界，球權轉交機器人對手。"
OPPONENT_OOB_CALL_TEXT_ZH = "機器人對手將球弄出界，球權轉交玩家。"
UNKNOWN_OUT_OF_BOUNDS_TEXT_ZH = "球出界，雙方重新準備球權。"
NEUTRAL_SHOT_TEXT_ZH = "球員攻向籃框並出手。"
PLAYER_CONTROL_TEXT = "The player protects the ball as the robot opponent applies pressure."
OPPONENT_CONTROL_TEXT = "The robot opponent protects the ball as the player applies pressure."
PLAYER_CONTROL_TEXT_ZH = "玩家保護球權，機器人對手持續施壓。"
OPPONENT_CONTROL_TEXT_ZH = "機器人對手保護球權，玩家持續施壓。"

_SCORE_CUE_RE = re.compile(r"\bscored\s*!", re.IGNORECASE)
_OOB_CUE_RE = re.compile(r"\bout\s+of\s+bounds\s*!", re.IGNORECASE)
_SCORE_OR_MISS_CLAIM_RE = re.compile(
    r"\b(?:score[sd]?|scoring|bucket|buckets|swish(?:es)?|nylon|"
    r"successful(?:ly)?|success|converts?|sinks?|sank|miss(?:es|ed)?|"
    r"makes? (?:the |a )?(?:shot|basket|layup)|made (?:the |a )?(?:shot|basket|layup)|"
    r"drops? (?:home|through|in)|bottom of (?:the )?(?:net|cup)|"
    r"banks? in|puts? it through|finds? the mark|hits? the mark|"
    r"goes? through|went through|through the hoop|for a basket|nothing but net)\b",
    re.IGNORECASE,
)
_OOB_CLAIM_RE = re.compile(r"\b(?:out of bounds|out of play|possession (?:is )?awarded)\b", re.IGNORECASE)
_ZH_SCORE_OR_MISS_CLAIM_RE = re.compile(r"得分|進球|進了|命中|投進|未進|沒進")
_ZH_OOB_CLAIM_RE = re.compile(r"出界|界外")
_TEAMWORK_CLAIM_RE = re.compile(
    r"\b(?:team-?mates?|passes?|passed|passing|dishes?|feeds?|fed|assists?)\b|隊友|傳球|助攻",
    re.IGNORECASE,
)
_UI_META_CLAIM_RE = re.compile(
    r"\b(?:clock|timer|scoreboard|interface|screen|camera|headset|controller|equipment|vr)\b",
    re.IGNORECASE,
)
_PRESSURE_CONTEXT_RE = re.compile(
    r"\b(?:pressure|pressured|defend(?:er|ing|ed)?|contest(?:ed|ing)?|"
    r"guard(?:ed|ing)?|challenge[sd]?|close[- ]out|traffic)\b|防守|壓力|干擾",
    re.IGNORECASE,
)
_DRIVE_CONTEXT_RE = re.compile(
    r"\b(?:drive[sd]?|driving|cut(?:s|ting)?|penetrat(?:e[sd]?|ing)|"
    r"attack(?:s|ing)?|lane|paint|rim|basket|layup)\b|切入|突破|禁區|籃下",
    re.IGNORECASE,
)
_SHOT_CONTEXT_RE = re.compile(
    r"\b(?:shoot(?:s|ing)?|shot|release[sd]?|jumper|layup|attempt)\b|出手|投籃",
    re.IGNORECASE,
)


def compose_result_evidence(result_text: str, recent_action: str = "") -> str:
    """Attach prior visible action without weakening the exact result cue."""
    action = " ".join(recent_action.split()).strip()
    return f"Previous visible action: {action}\n{result_text}" if action else result_text


def _result_context(broadcast_text: str, raw_visual_text: str) -> str:
    cue_matches = [*_SCORE_CUE_RE.finditer(raw_visual_text), *_OOB_CUE_RE.finditer(raw_visual_text)]
    evidence = raw_visual_text[: max((match.start() for match in cue_matches), default=0)]
    if _PRESSURE_CONTEXT_RE.search(evidence):
        return "pressure"
    if _DRIVE_CONTEXT_RE.search(evidence):
        return "drive"
    if _SHOT_CONTEXT_RE.search(evidence):
        return "shot"
    return ""


def _contextual_result_text(cue: str, side: str | None, context: str, zh: bool) -> str | None:
    actor_en = "the player" if side == "home" else "the robot opponent" if side == "away" else ""
    actor_zh = "玩家" if side == "home" else "機器人對手" if side == "away" else ""
    if cue == "score" and context and actor_en:
        if zh:
            if context == "pressure":
                return f"{actor_zh}在防守壓力下出手並成功得分。"
            if context == "drive":
                return f"{actor_zh}切入禁區並完成得分。"
            return f"{actor_zh}果斷出手並成功得分。"
        if context == "pressure":
            return f"Under defensive pressure, {actor_en} releases the shot and scores."
        if context == "drive":
            return f"{actor_en[0].upper() + actor_en[1:]} drives into the lane and finishes for the score."
        return f"{actor_en[0].upper() + actor_en[1:]} releases the shot and scores."
    if cue == "out_of_bounds" and context:
        receiver_en = "the robot opponent" if side == "home" else "the player" if side == "away" else ""
        receiver_zh = "機器人對手" if side == "home" else "玩家" if side == "away" else ""
        if actor_en:
            if zh:
                if context == "pressure":
                    return f"{actor_zh}在壓力下進攻，球出了界，球權轉交{receiver_zh}。"
                if context == "drive":
                    return f"{actor_zh}切入進攻，球出了界，球權轉交{receiver_zh}。"
                return f"{actor_zh}出手後球飛出界外，球權轉交{receiver_zh}。"
            if context == "pressure":
                return f"{actor_en[0].upper() + actor_en[1:]} attacks under pressure, but the ball goes out of bounds and {receiver_en} takes possession."
            if context == "drive":
                return f"{actor_en[0].upper() + actor_en[1:]} drives into the lane, but the ball goes out of bounds and {receiver_en} takes possession."
            return f"{actor_en[0].upper() + actor_en[1:]} releases the shot, the ball goes out of bounds, and {receiver_en} takes possession."
        if zh:
            lead = "在防守壓力下，球出了界" if context == "pressure" else (
                "切入過程中球出了界" if context == "drive" else "出手後球出了界"
            )
            return f"{lead}，雙方重新準備球權。"
        lead = "Under defensive pressure, the ball goes out of bounds" if context == "pressure" else (
            "The drive ends with the ball going out of bounds" if context == "drive"
            else "After the shot attempt, the ball goes out of bounds"
        )
        return f"{lead} and possession resets."
    return None


def _opponent_is_primary(raw_visual_text: str) -> bool:
    raw_lower = raw_visual_text.lower()
    opponent_positions = [
        raw_lower.find(label)
        for label in ("the robot opponent", "the opponent", "test_bot", "test bot")
        if label in raw_lower
    ]
    if not opponent_positions:
        return False
    player_position = raw_lower.find("the player")
    return player_position < 0 or min(opponent_positions) < player_position


def result_cue(raw_visual_text: str) -> str | None:
    cues = [
        *((match.start(), "score") for match in _SCORE_CUE_RE.finditer(raw_visual_text)),
        *((match.start(), "out_of_bounds") for match in _OOB_CUE_RE.finditer(raw_visual_text)),
    ]
    return max(cues)[1] if cues else None


def result_side(raw_visual_text: str, cue: str | None = None) -> str | None:
    """Return Home/Away for the latest explicit result banner, never the clock."""
    cue = cue or result_cue(raw_visual_text)
    pattern = _SCORE_CUE_RE if cue == "score" else _OOB_CUE_RE if cue == "out_of_bounds" else None
    if pattern is None:
        return None
    matches = list(pattern.finditer(raw_visual_text))
    if not matches:
        return None
    # Home/Away is printed immediately after the result phrase. Limiting the
    # search prevents the persistent top scoreboard from being mistaken for it.
    nearby = raw_visual_text[matches[-1].end() : matches[-1].end() + 48]
    side = re.search(r"\b(home|away)\b", nearby, re.IGNORECASE)
    return side.group(1).lower() if side else None


def result_score(raw_visual_text: str) -> tuple[int, int] | None:
    """Read the session score attached to the latest confirmed score cue."""
    matches = list(_SCORE_CUE_RE.finditer(raw_visual_text))
    if not matches:
        return None
    nearby = raw_visual_text[matches[-1].end() : matches[-1].end() + 120]
    score = re.search(
        r"\bscore\s*:\s*home\s+(\d+)\s*,?\s*away\s+(\d+)\b",
        nearby,
        re.IGNORECASE,
    )
    return (int(score.group(1)), int(score.group(2))) if score else None


def _append_score(text: str, score: tuple[int, int] | None, zh: bool) -> str:
    if score is None:
        return text
    home, away = score
    if zh:
        return f"{text.rstrip('。')}。目前比分：玩家 {home}，機器人對手 {away}。"
    return f"{text.rstrip('.')}. The score is player {home}, robot opponent {away}."


def ground_broadcast_text(
    broadcast_text: str, raw_visual_text: str, *, language: str = "en"
) -> str:
    zh = language == "zh"
    if not zh:
        # In one-on-one play, "kick it back out" falsely implies a pass. The
        # observed action is the same ballhandler retreating to the perimeter.
        broadcast_text = re.sub(
            r"\bkicks? (?:the ball|it) back out\b",
            "retreats to the perimeter",
            broadcast_text,
            flags=re.IGNORECASE,
        )
        broadcast_text = re.sub(r"^Player\b", "The player", broadcast_text)
        broadcast_text = re.sub(r"^The robot\b(?!\s+opponent)", "The robot opponent", broadcast_text, flags=re.IGNORECASE)
        broadcast_text = re.sub(r"\bthe robot\b(?!\s+opponent)", "the robot opponent", broadcast_text, flags=re.IGNORECASE)
        broadcast_text = re.sub(r"\btest_?bot\d*\b", "the robot opponent", broadcast_text, flags=re.IGNORECASE)
        broadcast_text = re.sub(
            r"^The opponent\b", "The robot opponent", broadcast_text, flags=re.IGNORECASE
        )
        broadcast_text = re.sub(
            r"(?<!robot )\bthe opponent\b",
            "the robot opponent",
            broadcast_text,
            flags=re.IGNORECASE,
        )
        broadcast_text = re.sub(
            r"^Players\b",
            "The player and robot opponent",
            broadcast_text,
            flags=re.IGNORECASE,
        )
    cue = result_cue(raw_visual_text)
    if cue == "score":
        side = result_side(raw_visual_text, cue)
        score = result_score(raw_visual_text)
        contextual = _contextual_result_text(
            cue, side, _result_context(broadcast_text, raw_visual_text), zh
        )
        if contextual:
            return _append_score(contextual, score, zh)
        if side == "home":
            return _append_score(PLAYER_SCORE_TEXT_ZH if zh else PLAYER_SCORE_TEXT, score, zh)
        if side == "away":
            return _append_score(OPPONENT_SCORE_TEXT_ZH if zh else OPPONENT_SCORE_TEXT, score, zh)
        return _append_score(UNKNOWN_SCORE_TEXT_ZH if zh else UNKNOWN_SCORE_TEXT, score, zh)
    if cue == "out_of_bounds":
        side = result_side(raw_visual_text, cue)
        contextual = _contextual_result_text(
            cue, side, _result_context(broadcast_text, raw_visual_text), zh
        )
        if contextual:
            return contextual
        if side == "home":
            return PLAYER_OOB_CALL_TEXT_ZH if zh else PLAYER_OOB_CALL_TEXT
        if side == "away":
            return OPPONENT_OOB_CALL_TEXT_ZH if zh else OPPONENT_OOB_CALL_TEXT
        return UNKNOWN_OUT_OF_BOUNDS_TEXT_ZH if zh else UNKNOWN_OUT_OF_BOUNDS_TEXT
    if _UI_META_CLAIM_RE.search(broadcast_text):
        if _opponent_is_primary(raw_visual_text):
            return OPPONENT_CONTROL_TEXT_ZH if zh else OPPONENT_CONTROL_TEXT
        return PLAYER_CONTROL_TEXT_ZH if zh else PLAYER_CONTROL_TEXT
    if _TEAMWORK_CLAIM_RE.search(broadcast_text):
        if _opponent_is_primary(raw_visual_text):
            return OPPONENT_CONTROL_TEXT_ZH if zh else OPPONENT_CONTROL_TEXT
        return PLAYER_CONTROL_TEXT_ZH if zh else PLAYER_CONTROL_TEXT
    if _SCORE_OR_MISS_CLAIM_RE.search(broadcast_text) or _ZH_SCORE_OR_MISS_CLAIM_RE.search(broadcast_text):
        return NEUTRAL_SHOT_TEXT_ZH if zh else NEUTRAL_SHOT_TEXT
    if _OOB_CLAIM_RE.search(broadcast_text) or _ZH_OOB_CLAIM_RE.search(broadcast_text):
        if _opponent_is_primary(raw_visual_text):
            return OPPONENT_CONTROL_TEXT_ZH if zh else OPPONENT_CONTROL_TEXT
        return PLAYER_CONTROL_TEXT_ZH if zh else PLAYER_CONTROL_TEXT
    return broadcast_text.strip()
