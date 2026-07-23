"""Deterministic grounding and pacing for Frame-based broadcast candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass


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

NEUTRAL_SHOT_TEXT = "The player attacks and releases a shot toward the basket."
PLAYER_SCORE_TEXT = "The player attacks the basket and scores."
OPPONENT_SCORE_TEXT = "The robot opponent attacks the basket and scores."
UNKNOWN_SCORE_TEXT = "A basket is confirmed."
PLAYER_OOB_CALL_TEXT = "The player sends the ball out of bounds."
OPPONENT_OOB_CALL_TEXT = "The robot opponent sends the ball out of bounds."
UNKNOWN_OUT_OF_BOUNDS_TEXT = "The ball goes out of bounds and possession resets."
NEUTRAL_CONTROL_TEXT = "The player controls the ball under pressure."
_PRESSURE_CONTEXT_RE = re.compile(
    r"\b(?:pressure|pressured|defend(?:er|ing|ed)?|contest(?:ed|ing)?|guard(?:ed|ing)?|"
    r"challenge[sd]?|close[- ]out|traffic)\b",
    re.IGNORECASE,
)
_DRIVE_CONTEXT_RE = re.compile(
    r"\b(?:drive[sd]?|driving|cut(?:s|ting)?|penetrat(?:e[sd]?|ing)|attack(?:s|ing)?|"
    r"lane|paint|rim|basket|layup)\b",
    re.IGNORECASE,
)
_SHOT_CONTEXT_RE = re.compile(
    r"\b(?:shoot(?:s|ing)?|shot|release[sd]?|jumper|layup|attempt)\b",
    re.IGNORECASE,
)


def compose_result_evidence(result_text: str, recent_action: str = "") -> str:
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


def _contextual_result_text(cue: str, side: str | None, context: str) -> str | None:
    actor = "the player" if side == "home" else "the robot opponent" if side == "away" else ""
    if cue == "score" and context and actor:
        if context == "pressure":
            return f"Under defensive pressure, {actor} releases the shot and scores."
        if context == "drive":
            return f"{actor[0].upper() + actor[1:]} drives into the lane and finishes for the score."
        return f"{actor[0].upper() + actor[1:]} releases the shot and scores."
    if cue == "out_of_bounds" and context:
        if actor:
            if context == "pressure":
                return f"{actor[0].upper() + actor[1:]} attacks under pressure, and the ball goes out of bounds."
            if context == "drive":
                return f"{actor[0].upper() + actor[1:]} drives into the lane, and the ball goes out of bounds."
            return f"{actor[0].upper() + actor[1:]} releases the shot, and the ball goes out of bounds."
        lead = "Under defensive pressure, the ball goes out of bounds" if context == "pressure" else (
            "The drive ends with the ball going out of bounds" if context == "drive"
            else "After the shot attempt, the ball goes out of bounds"
        )
        return f"{lead} and possession resets."
    return None


def result_cue(raw_visual_text: str) -> str | None:
    """Return the explicit result cue carried by the visual caption, if any."""
    # A window can contain the end of one play and the result of the next.  Use
    # the last explicit banner in temporal caption order, rather than giving a
    # made basket unconditional precedence.
    cues = [
        *((match.start(), "score") for match in _SCORE_CUE_RE.finditer(raw_visual_text)),
        *((match.start(), "out_of_bounds") for match in _OOB_CUE_RE.finditer(raw_visual_text)),
    ]
    return max(cues)[1] if cues else None


def result_side(raw_visual_text: str, cue: str | None = None) -> str | None:
    """Resolve the side printed beside the latest result cue."""
    cue = cue or result_cue(raw_visual_text)
    pattern = _SCORE_CUE_RE if cue == "score" else _OOB_CUE_RE if cue == "out_of_bounds" else None
    if pattern is None:
        return None
    matches = list(pattern.finditer(raw_visual_text))
    if not matches:
        return None
    nearby = raw_visual_text[matches[-1].end() : matches[-1].end() + 48]
    side = re.search(r"\b(home|away)\b", nearby, re.IGNORECASE)
    return side.group(1).lower() if side else None


def ground_broadcast_text(broadcast_text: str, raw_visual_text: str) -> str:
    """Make outcome wording agree with the latest explicit visual result banner."""
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
    broadcast_text = re.sub(r"^The opponent\b", "The robot opponent", broadcast_text, flags=re.IGNORECASE)
    broadcast_text = re.sub(r"(?<!robot )\bthe opponent\b", "the robot opponent", broadcast_text, flags=re.IGNORECASE)
    broadcast_text = re.sub(r"^Players\b", "The player and robot opponent", broadcast_text, flags=re.IGNORECASE)
    cue = result_cue(raw_visual_text)
    if cue == "score":
        side = result_side(raw_visual_text, cue)
        contextual = _contextual_result_text(cue, side, _result_context(broadcast_text, raw_visual_text))
        if contextual:
            return contextual
        return PLAYER_SCORE_TEXT if side == "home" else OPPONENT_SCORE_TEXT if side == "away" else UNKNOWN_SCORE_TEXT
    if cue == "out_of_bounds":
        side = result_side(raw_visual_text, cue)
        contextual = _contextual_result_text(cue, side, _result_context(broadcast_text, raw_visual_text))
        if contextual:
            return contextual
        return PLAYER_OOB_CALL_TEXT if side == "home" else OPPONENT_OOB_CALL_TEXT if side == "away" else UNKNOWN_OUT_OF_BOUNDS_TEXT
    if _SCORE_OR_MISS_CLAIM_RE.search(broadcast_text):
        return NEUTRAL_SHOT_TEXT
    if _OOB_CLAIM_RE.search(broadcast_text):
        return NEUTRAL_CONTROL_TEXT
    return broadcast_text.strip()


@dataclass
class BroadcastPacer:
    """Keep decisive cues and throttle routine commentary without consulting GT."""

    routine_interval_sec: float = 6.0
    cue_repeat_sec: float = 3.0
    _last_kept_start: float = -1e9
    _last_cue_start: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.routine_interval_sec < 0 or self.cue_repeat_sec < 0:
            raise ValueError("pacing intervals must be non-negative")
        self._last_cue_start = {}

    def should_keep(self, start: float, raw_visual_text: str) -> bool:
        cue = result_cue(raw_visual_text)
        if cue:
            assert self._last_cue_start is not None
            if start - self._last_cue_start.get(cue, -1e9) < self.cue_repeat_sec:
                return False
            self._last_cue_start[cue] = start
            self._last_kept_start = start
            return True
        if start - self._last_kept_start < self.routine_interval_sec:
            return False
        self._last_kept_start = start
        return True
