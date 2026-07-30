"""Evidence-grounded outcome wording for basketball commentary."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence


PLAYER_SCORE_TEXT = "The player attacks the basket and scores."
OPPONENT_SCORE_TEXT = "The robot opponent attacks the basket and scores."
UNKNOWN_SCORE_TEXT = "A basket is confirmed."
PLAYER_OOB_CALL_TEXT = "The player sends the ball out of bounds, giving possession to the robot opponent."
OPPONENT_OOB_CALL_TEXT = "The robot opponent sends the ball out of bounds, giving possession to the player."
UNKNOWN_OUT_OF_BOUNDS_TEXT = "The ball goes out of bounds and possession resets."
PLAYER_SHOT_CLOCK_TEXT = (
    "The player runs out of shot clock; possession goes to the robot opponent."
)
OPPONENT_SHOT_CLOCK_TEXT = (
    "The robot opponent runs out of shot clock; possession goes to the player."
)
UNKNOWN_SHOT_CLOCK_TEXT = "Shot clock expires and possession changes."
NEUTRAL_SHOT_TEXT = "The player attacks and releases a shot toward the basket."
PLAYER_SCORE_TEXT_ZH = "玩家攻向籃框並成功得分。"
OPPONENT_SCORE_TEXT_ZH = "機器人對手攻向籃框並成功得分。"
UNKNOWN_SCORE_TEXT_ZH = "畫面確認這次進攻得分。"
PLAYER_OOB_CALL_TEXT_ZH = "玩家將球弄出界，球權轉交機器人對手。"
OPPONENT_OOB_CALL_TEXT_ZH = "機器人對手將球弄出界，球權轉交玩家。"
UNKNOWN_OUT_OF_BOUNDS_TEXT_ZH = "球出界，雙方重新準備球權。"
PLAYER_SHOT_CLOCK_TEXT_ZH = "玩家進攻時間到點，球權轉交機器人對手。"
OPPONENT_SHOT_CLOCK_TEXT_ZH = "機器人對手進攻時間到點，球權轉交玩家。"
UNKNOWN_SHOT_CLOCK_TEXT_ZH = "進攻時間到點，球權轉換。"
NEUTRAL_SHOT_TEXT_ZH = "球員攻向籃框並出手。"
# A prolonged possession with no clear drive/pass/shot — keep neutral; do NOT
# invent a fake unless LiveCC actually reported rocking / jab / pump-fake motion.
PLAYER_CONTROL_TEXT = "The player protects the ball as the robot opponent applies pressure."
OPPONENT_CONTROL_TEXT = "The robot opponent protects the ball as the player applies pressure."
PLAYER_CONTROL_TEXT_ZH = "玩家保護球權，機器人對手持續施壓。"
OPPONENT_CONTROL_TEXT_ZH = "機器人對手保護球權，玩家持續施壓。"

# Minimal, deterministic per-finish lines. These are a last-resort fallback
# only - used when there is no real, usable visual description to ground the
# sentence in (see _ground_score_evidence_text below, which is what actually
# renders most confirmed baskets from what LiveCC really saw on screen).
_DUNK_TEXT_EN = "{Actor} throws down a dunk and scores."
_LAYUP_TEXT_EN = "{Actor} finishes the layup and scores."
_THREE_TEXT_EN = "{Actor} drains the three and scores."
_JUMPER_TEXT_EN = "{Actor} knocks down the jumper and scores."
_PRESSURE_TEXT_EN = "Under defensive pressure, {actor} releases the shot and scores."
_DRIVE_TEXT_EN = "{Actor} drives into the lane and finishes for the score."
_SHOT_TEXT_EN = "{Actor} releases the shot and scores."
_DUNK_TEXT_ZH = "{actor}灌籃得分。"
_LAYUP_TEXT_ZH = "{actor}上籃得分。"
_THREE_TEXT_ZH = "{actor}三分命中。"
_JUMPER_TEXT_ZH = "{actor}跳投得分。"
_PRESSURE_TEXT_ZH = "{actor}在防守壓力下出手並成功得分。"
_DRIVE_TEXT_ZH = "{actor}切入禁區並完成得分。"
_SHOT_TEXT_ZH = "{actor}果斷出手並成功得分。"

_SCORE_FALLBACK_EN = {
    "dunk": _DUNK_TEXT_EN,
    "layup": _LAYUP_TEXT_EN,
    "three": _THREE_TEXT_EN,
    "jumper": _JUMPER_TEXT_EN,
    "pressure": _PRESSURE_TEXT_EN,
    "drive": _DRIVE_TEXT_EN,
    "shot": _SHOT_TEXT_EN,
}
_SCORE_FALLBACK_ZH = {
    "dunk": _DUNK_TEXT_ZH,
    "layup": _LAYUP_TEXT_ZH,
    "three": _THREE_TEXT_ZH,
    "jumper": _JUMPER_TEXT_ZH,
    "pressure": _PRESSURE_TEXT_ZH,
    "drive": _DRIVE_TEXT_ZH,
    "shot": _SHOT_TEXT_ZH,
}

# Style tone pools for confirmed P1 banners. Facts (actor / outcome / finish)
# stay identical across variants; only wording / punctuation change.
# Each entry is one or more templates — a short-term memory avoids repeating
# the exact same spoken line back-to-back.
_StyleLine = str | tuple[str, ...]

_STYLE_SCORE_FALLBACK_ZH: dict[str, dict[str, _StyleLine]] = {
    "objective": {
        "dunk": (
            "{actor}灌籃得分。",
            "{actor}完成灌籃。",
            "{actor}這記灌籃得分。",
        ),
        "layup": (
            "{actor}上籃得分。",
            "{actor}完成上籃。",
            "{actor}近距離上籃得分。",
        ),
        "three": (
            "{actor}三分命中。",
            "{actor}外線三分得分。",
            "{actor}三分出手命中。",
        ),
        "jumper": (
            "{actor}跳投得分。",
            "{actor}中距離跳投命中。",
            "{actor}完成跳投得分。",
        ),
        "pressure": (
            "{actor}在防守壓力下出手並成功得分。",
            "{actor}頂住防守壓力後得分。",
            "{actor}受干擾仍出手命中。",
        ),
        "drive": (
            "{actor}切入禁區並完成得分。",
            "{actor}切入禁區得分。",
            "{actor}突破後完成得分。",
        ),
        "shot": (
            "{actor}果斷出手並成功得分。",
            "{actor}出手後成功得分。",
            "{actor}出手命中。",
        ),
    },
    "hype": {
        "dunk": (
            "哇！{actor}灌籃得分，太炸裂啦！",
            "天啊！{actor}這記灌籃直接砸進得分！",
            "漂亮！{actor}灌籃得分！",
        ),
        "layup": (
            "進了！{actor}上籃得分！",
            "得手！{actor}上籃輕鬆放進！",
            "哇！{actor}上籃得分進帳！",
        ),
        "three": (
            "空心！{actor}三分命中！",
            "進了！{actor}外線三分得分！",
            "漂亮！{actor}三分出手命中！",
        ),
        "jumper": (
            "漂亮！{actor}跳投得分！",
            "進了！{actor}中距離跳投命中！",
            "好球！{actor}跳投得分進帳！",
        ),
        "pressure": (
            "頂住了！{actor}在防守壓力下照樣得分！",
            "進了！{actor}被貼還是投進！",
            "強！{actor}在壓力下出手得分！",
        ),
        "drive": (
            "殺進去！{actor}切入禁區強勢得分！",
            "進了！{actor}切入禁區完成得分！",
            "漂亮！{actor}切入得分！",
        ),
        "shot": (
            "進了！{actor}出手得分！",
            "得手！{actor}果斷出手得分！",
            "哇！{actor}出手命中！",
        ),
    },
    "calm": {
        "dunk": (
            "{actor}完成灌籃得分。",
            "{actor}灌籃得分。",
            "{actor}在籃下完成灌籃。",
        ),
        "layup": (
            "{actor}完成上籃得分。",
            "{actor}上籃得分。",
            "{actor}近距離上籃命中。",
        ),
        "three": (
            "{actor}三分出手命中。",
            "{actor}外線三分得分。",
            "{actor}三分命中。",
        ),
        "jumper": (
            "{actor}中距離跳投命中。",
            "{actor}跳投得分。",
            "{actor}完成跳投得分。",
        ),
        "pressure": (
            "{actor}在防守壓力下出手並得分。",
            "{actor}頂住防守壓力後得分。",
            "{actor}受干擾仍出手命中。",
        ),
        "drive": (
            "{actor}切入禁區後完成得分。",
            "{actor}切入禁區得分。",
            "{actor}突破後完成得分。",
        ),
        "shot": (
            "{actor}出手後成功得分。",
            "{actor}出手命中。",
            "{actor}完成這次得分。",
        ),
    },
    "trash_talk": {
        "dunk": (
            "{actor}灌籃得分，這球沒得擋！",
            "{actor}直接灌進，誰攔得住！",
            "{actor}灌籃得分，太乾淨了！",
        ),
        "layup": (
            "{actor}上籃得分，輕鬆拿下！",
            "{actor}上籃得分，送分題啊！",
            "{actor}放進上籃，太簡單了！",
        ),
        "three": (
            "{actor}三分命中，外線給他開！",
            "{actor}三分進了，誰放的空檔！",
            "{actor}外線三分，手感來了！",
        ),
        "jumper": (
            "{actor}跳投得分，手感來了！",
            "{actor}跳投進了，擋不住！",
            "{actor}中距離跳投，收下！",
        ),
        "pressure": (
            "{actor}被貼還是投進，照樣得分！",
            "{actor}有防守也進，沒轍！",
            "{actor}頂著壓力得分，給過！",
        ),
        "drive": (
            "{actor}切入禁區得分，防守跟不上！",
            "{actor}殺進去得分，太慢了！",
            "{actor}切入得手，防守遲到！",
        ),
        "shot": (
            "{actor}出手得分，進了就是進了！",
            "{actor}出手命中，沒話說！",
            "{actor}得分進帳，收下！",
        ),
    },
}
_STYLE_SCORE_FALLBACK_EN: dict[str, dict[str, _StyleLine]] = {
    "objective": {
        "dunk": (
            "{Actor} throws down a dunk and scores.",
            "{Actor} finishes with a dunk.",
            "{Actor} scores on the dunk.",
        ),
        "layup": (
            "{Actor} finishes the layup and scores.",
            "{Actor} converts the layup.",
            "{Actor} scores on the layup.",
        ),
        "three": (
            "{Actor} drains the three and scores.",
            "{Actor} hits from beyond the arc.",
            "{Actor} converts the three.",
        ),
        "jumper": (
            "{Actor} knocks down the jumper and scores.",
            "{Actor} hits the jumper.",
            "{Actor} scores on the jump shot.",
        ),
        "pressure": (
            "Under defensive pressure, {actor} releases the shot and scores.",
            "{Actor} scores through the contest.",
            "Despite pressure, {actor} scores.",
        ),
        "drive": (
            "{Actor} drives into the lane and finishes for the score.",
            "{Actor} finishes the drive.",
            "{Actor} scores after the drive.",
        ),
        "shot": (
            "{Actor} releases the shot and scores.",
            "{Actor} converts the shot.",
            "{Actor} scores on the attempt.",
        ),
    },
    "hype": {
        "dunk": (
            "{Actor} throws down a dunk — what a slam!",
            "Wow! {Actor} hammers home the dunk!",
            "Yes! {Actor} throws down a dunk and scores!",
        ),
        "layup": (
            "{Actor} finishes the layup and scores!",
            "And in! {Actor} lays it up for the score!",
            "Got it! {Actor} finishes the layup!",
        ),
        "three": (
            "{Actor} drains the three — from downtown!",
            "Bang! {Actor} hits the three!",
            "Yes! {Actor} drains it from beyond the arc!",
        ),
        "jumper": (
            "{Actor} knocks down the jumper!",
            "Yes! {Actor} hits the mid-range jumper!",
            "Got it! {Actor} knocks down the jumper!",
        ),
        "pressure": (
            "Under defensive pressure, {actor} still scores!",
            "Contested — and {actor} still scores!",
            "Pressure on, and {actor} converts anyway!",
        ),
        "drive": (
            "{Actor} drives into the lane and finishes hard!",
            "Attacking! {Actor} drives and scores!",
            "Yes! {Actor} finishes the drive!",
        ),
        "shot": (
            "{Actor} lets it fly and scores!",
            "And in! {Actor} shoots and scores!",
            "Got it! {Actor} scores on the shot!",
        ),
    },
    "calm": {
        "dunk": (
            "{Actor} completes the dunk and scores.",
            "{Actor} finishes with a dunk.",
            "{Actor} scores on the dunk.",
        ),
        "layup": (
            "{Actor} finishes the layup cleanly.",
            "{Actor} converts the layup.",
            "{Actor} scores on the layup.",
        ),
        "three": (
            "{Actor} makes the three-point shot.",
            "{Actor} hits from beyond the arc.",
            "{Actor} converts the three.",
        ),
        "jumper": (
            "{Actor} converts the mid-range jumper.",
            "{Actor} hits the jumper.",
            "{Actor} scores on the jump shot.",
        ),
        "pressure": (
            "Under defensive pressure, {actor} still converts.",
            "{Actor} scores through the contest.",
            "Despite pressure, {actor} scores.",
        ),
        "drive": (
            "{Actor} drives into the lane and scores.",
            "{Actor} finishes the drive.",
            "{Actor} scores after the drive.",
        ),
        "shot": (
            "{Actor} releases and scores.",
            "{Actor} converts the shot.",
            "{Actor} scores on the attempt.",
        ),
    },
    "trash_talk": {
        "dunk": (
            "{Actor} throws down a dunk — no contest!",
            "{Actor} dunks it — who is stopping that?",
            "{Actor} slams it home — too easy!",
        ),
        "layup": (
            "{Actor} lays it in easy — too simple!",
            "{Actor} finishes the layup — gift points!",
            "{Actor} lays it up — defense was late!",
        ),
        "three": (
            "{Actor} drains the three — leave them open!",
            "{Actor} hits the three — who left them?",
            "{Actor} from deep — that is cash!",
        ),
        "jumper": (
            "{Actor} knocks down the jumper — that is cash!",
            "{Actor} hits the jumper — keep shooting!",
            "{Actor} mid-range — money!",
        ),
        "pressure": (
            "Contested or not, {actor} still scores!",
            "Pressure means nothing — {actor} scores!",
            "You contested it and {actor} still scored!",
        ),
        "drive": (
            "{Actor} drives and finishes — defense was late!",
            "{Actor} attacks the rim and scores — too slow!",
            "{Actor} drives home — no help defense!",
        ),
        "shot": (
            "{Actor} shoots and scores — nothing you can do!",
            "{Actor} scores — take the L!",
            "{Actor} hits it — that one counts!",
        ),
    },
}

# Generic score / OOB / shot-clock lines when no finish context is available.
_STYLE_GENERIC_ZH: dict[str, dict[str, _StyleLine]] = {
    "objective": {
        "home_score": (
            "玩家攻向籃框並成功得分。",
            "玩家完成得分。",
            "玩家這次進攻得分。",
        ),
        "away_score": (
            "機器人對手攻向籃框並成功得分。",
            "機器人對手完成得分。",
            "機器人對手這次進攻得分。",
        ),
        "unknown_score": (
            "畫面確認這次進攻得分。",
            "這球確認得分。",
            "進攻得分成立。",
        ),
        "home_oob": (
            "玩家將球弄出界，球權轉交機器人對手。",
            "玩家球出界，球權交給機器人對手。",
            "玩家弄出界外，球權轉換。",
        ),
        "away_oob": (
            "機器人對手將球弄出界，球權轉交玩家。",
            "機器人對手球出界，球權交給玩家。",
            "機器人對手弄出界外，球權轉換。",
        ),
        "unknown_oob": (
            "球出界，雙方重新準備球權。",
            "球出界，球權重新開始。",
            "出界，雙方重新準備。",
        ),
        "home_scv": (
            "玩家進攻時間到點，球權轉交機器人對手。",
            "玩家進攻時間用盡，球權轉換。",
            "玩家進攻違例，球權交給機器人對手。",
        ),
        "away_scv": (
            "機器人對手進攻時間到點，球權轉交玩家。",
            "機器人對手進攻時間用盡，球權轉換。",
            "機器人對手進攻違例，球權交給玩家。",
        ),
        "unknown_scv": (
            "進攻時間到點，球權轉換。",
            "進攻時間用盡，球權轉換。",
            "進攻違例，球權重新分配。",
        ),
        "home_oob_pressure": (
            "玩家在壓力下進攻，球出了界，球權轉交機器人對手。",
            "玩家受壓後球出界，球權交給機器人對手。",
            "玩家壓力下弄出界，球權轉換。",
        ),
        "home_oob_drive": (
            "玩家切入進攻，球出了界，球權轉交機器人對手。",
            "玩家切入過程球出界，球權交給機器人對手。",
            "玩家切入後出界，球權轉換。",
        ),
        "home_oob_shot": (
            "玩家出手後球飛出界外，球權轉交機器人對手。",
            "玩家出手後出界，球權交給機器人對手。",
            "玩家投籃後球出界，球權轉換。",
        ),
        "away_oob_pressure": (
            "機器人對手在壓力下進攻，球出了界，球權轉交玩家。",
            "機器人對手受壓後球出界，球權交給玩家。",
            "機器人對手壓力下弄出界，球權轉換。",
        ),
        "away_oob_drive": (
            "機器人對手切入進攻，球出了界，球權轉交玩家。",
            "機器人對手切入過程球出界，球權交給玩家。",
            "機器人對手切入後出界，球權轉換。",
        ),
        "away_oob_shot": (
            "機器人對手出手後球飛出界外，球權轉交玩家。",
            "機器人對手出手後出界，球權交給玩家。",
            "機器人對手投籃後球出界，球權轉換。",
        ),
    },
    "hype": {
        "home_score": (
            "進了！玩家攻向籃框，得分進帳！",
            "哇！玩家得分成功！",
            "得手！玩家攻籃得分！",
        ),
        "away_score": (
            "進了！機器人對手攻向籃框，得分進帳！",
            "哇！機器人對手得分成功！",
            "得手！機器人對手攻籃得分！",
        ),
        "unknown_score": (
            "進了！這球確認得分！",
            "哇！確認得分！",
            "進球了！得分成立！",
        ),
        "home_oob": (
            "出界！玩家把球弄出界，球權交出去了！",
            "糟糕！玩家球出界，球權丟了！",
            "出界了！玩家弄出界外，球權轉換！",
        ),
        "away_oob": (
            "出界！機器人對手把球弄出界，球權回來了！",
            "好機會！對手出界，球權回來！",
            "出界了！機器人對手弄出界外，球權轉換！",
        ),
        "unknown_oob": (
            "出界了！球權重新開始！",
            "球出界！重新來過！",
            "出界！球權重置！",
        ),
        "home_scv": (
            "時間到！玩家進攻時間到點，球權被迫交出！",
            "啊！玩家進攻時間到點，球權沒了！",
            "時間到點！玩家違例，球權轉換！",
        ),
        "away_scv": (
            "時間到！機器人對手進攻時間到點，球權換邊！",
            "好！對手進攻時間到點，球權回來！",
            "時間到點！機器人對手違例，球權轉換！",
        ),
        "unknown_scv": (
            "時間到！進攻時間到點，球權轉換！",
            "時間到點！球權轉換！",
            "進攻違例！球權重新分配！",
        ),
        "home_oob_pressure": (
            "出界！玩家在壓力下進攻，球出了界，球權轉交機器人對手！",
            "壓力太大！玩家出界，球權交出！",
            "出界了！玩家受壓後弄出界外！",
        ),
        "home_oob_drive": (
            "出界！玩家切入進攻，球出了界，球權轉交機器人對手！",
            "切入失控！玩家出界，球權交出！",
            "出界了！玩家切入後球出界！",
        ),
        "home_oob_shot": (
            "出界！玩家出手後球飛出界外，球權轉交機器人對手！",
            "出手出界！玩家球權丟了！",
            "出界了！玩家投籃後球飛出界！",
        ),
        "away_oob_pressure": (
            "出界！機器人對手在壓力下進攻，球出了界，球權轉交玩家！",
            "壓出來了！對手出界，球權回來！",
            "出界了！對手受壓後弄出界外！",
        ),
        "away_oob_drive": (
            "出界！機器人對手切入進攻，球出了界，球權轉交玩家！",
            "切入失敗！對手出界，球權回來！",
            "出界了！對手切入後球出界！",
        ),
        "away_oob_shot": (
            "出界！機器人對手出手後球飛出界外，球權轉交玩家！",
            "出手出界！對手球權丟了！",
            "出界了！對手投籃後球飛出界！",
        ),
    },
    "calm": {
        "home_score": (
            "玩家攻向籃框並完成得分。",
            "玩家完成這次得分。",
            "玩家進攻得手。",
        ),
        "away_score": (
            "機器人對手攻向籃框並完成得分。",
            "機器人對手完成這次得分。",
            "機器人對手進攻得手。",
        ),
        "unknown_score": (
            "畫面確認這次進攻得分。",
            "這次進攻得分成立。",
            "確認得分。",
        ),
        "home_oob": (
            "玩家將球弄出界，球權轉交機器人對手。",
            "玩家球出界，球權轉換。",
            "玩家出界，球權交給機器人對手。",
        ),
        "away_oob": (
            "機器人對手將球弄出界，球權轉交玩家。",
            "機器人對手球出界，球權轉換。",
            "機器人對手出界，球權交給玩家。",
        ),
        "unknown_oob": (
            "球出界，雙方重新準備球權。",
            "球出界，球權重置。",
            "出界，重新開始。",
        ),
        "home_scv": (
            "玩家進攻時間用盡，球權轉交機器人對手。",
            "玩家進攻時間到點，球權轉換。",
            "玩家進攻違例，球權交給機器人對手。",
        ),
        "away_scv": (
            "機器人對手進攻時間用盡，球權轉交玩家。",
            "機器人對手進攻時間到點，球權轉換。",
            "機器人對手進攻違例，球權交給玩家。",
        ),
        "unknown_scv": (
            "進攻時間到期，球權轉換。",
            "進攻時間到點，球權轉換。",
            "進攻違例，球權重新分配。",
        ),
        "home_oob_pressure": (
            "玩家在壓力下進攻，球出了界，球權轉交機器人對手。",
            "玩家受壓出界，球權轉換。",
            "玩家壓力下球出界，球權交給機器人對手。",
        ),
        "home_oob_drive": (
            "玩家切入進攻，球出了界，球權轉交機器人對手。",
            "玩家切入出界，球權轉換。",
            "玩家切入後球出界，球權交給機器人對手。",
        ),
        "home_oob_shot": (
            "玩家出手後球飛出界外，球權轉交機器人對手。",
            "玩家出手出界，球權轉換。",
            "玩家投籃後球出界，球權交給機器人對手。",
        ),
        "away_oob_pressure": (
            "機器人對手在壓力下進攻，球出了界，球權轉交玩家。",
            "機器人對手受壓出界，球權轉換。",
            "機器人對手壓力下球出界，球權交給玩家。",
        ),
        "away_oob_drive": (
            "機器人對手切入進攻，球出了界，球權轉交玩家。",
            "機器人對手切入出界，球權轉換。",
            "機器人對手切入後球出界，球權交給玩家。",
        ),
        "away_oob_shot": (
            "機器人對手出手後球飛出界外，球權轉交玩家。",
            "機器人對手出手出界，球權轉換。",
            "機器人對手投籃後球出界，球權交給玩家。",
        ),
    },
    "trash_talk": {
        "home_score": (
            "玩家攻向籃框並得分，進了就是進了！",
            "玩家得分，收下！",
            "玩家攻籃得分，沒話說！",
        ),
        "away_score": (
            "機器人對手攻向籃框並得分，這球給他了！",
            "對手得分，給過！",
            "機器人對手攻籃得分，算他的！",
        ),
        "unknown_score": (
            "確認得分，球進籃框！",
            "進了，確認得分！",
            "球進了，得分成立！",
        ),
        "home_oob": (
            "玩家把球弄出界，這下球權送人了！",
            "玩家出界，白白送球權！",
            "玩家弄出界，球權飛了！",
        ),
        "away_oob": (
            "機器人對手把球弄出界，球權回來了！",
            "對手出界，球權白撿！",
            "機器人對手弄出界，謝謝贈送！",
        ),
        "unknown_oob": (
            "球出界，重新來過！",
            "出界了，重來！",
            "球出界，重置球權！",
        ),
        "home_scv": (
            "玩家進攻時間到點，自己耗掉球權！",
            "玩家時間到，白白浪費！",
            "玩家進攻違例，球權沒了！",
        ),
        "away_scv": (
            "機器人對手進攻時間到點，球權換邊！",
            "對手時間到，球權回來！",
            "機器人對手進攻違例，謝謝！",
        ),
        "unknown_scv": (
            "進攻時間到點，球權轉換！",
            "時間到，球權換邊！",
            "進攻違例，重新分配！",
        ),
        "home_oob_pressure": (
            "玩家在壓力下搞丟球權，球出界給機器人對手！",
            "壓力太大，玩家出界送分！",
            "玩家受壓出界，球權飛了！",
        ),
        "home_oob_drive": (
            "玩家切入卻把球弄出界，球權轉交機器人對手！",
            "切入失控，玩家出界送人！",
            "玩家切入出界，白忙一場！",
        ),
        "home_oob_shot": (
            "玩家出手後球飛出界，球權轉交機器人對手！",
            "出手出界，玩家送球權！",
            "玩家投出界，球權飛了！",
        ),
        "away_oob_pressure": (
            "機器人對手在壓力下搞丟球權，球出界給玩家！",
            "壓出來了，對手出界送分！",
            "對手受壓出界，球權白撿！",
        ),
        "away_oob_drive": (
            "機器人對手切入卻把球弄出界，球權轉交玩家！",
            "切入失敗，對手出界送分！",
            "對手切入出界，謝謝贈送！",
        ),
        "away_oob_shot": (
            "機器人對手出手後球飛出界，球權轉交玩家！",
            "出手出界，對手送球權！",
            "對手投出界，球權回來！",
        ),
    },
}
_STYLE_GENERIC_EN: dict[str, dict[str, _StyleLine]] = {
    "objective": {
        "home_score": (
            "The player attacks the basket and scores.",
            "The player completes the score.",
            "The player scores on the attack.",
        ),
        "away_score": (
            "The robot opponent attacks the basket and scores.",
            "The robot opponent completes the score.",
            "The robot opponent scores on the attack.",
        ),
        "unknown_score": (
            "A basket is confirmed.",
            "The score is confirmed.",
            "That basket counts.",
        ),
        "home_oob": (
            "The player sends the ball out of bounds, giving possession to the robot opponent.",
            "The player goes out of bounds; the robot opponent takes possession.",
            "Out of bounds on the player — possession flips.",
        ),
        "away_oob": (
            "The robot opponent sends the ball out of bounds, giving possession to the player.",
            "The robot opponent goes out of bounds; the player takes possession.",
            "Out of bounds on the robot opponent — possession flips.",
        ),
        "unknown_oob": (
            "The ball goes out of bounds and possession resets.",
            "Out of bounds — possession resets.",
            "The ball is out; possession resets.",
        ),
        "home_scv": (
            "The player runs out of shot clock; possession goes to the robot opponent.",
            "Shot clock expires on the player — possession flips.",
            "The player commits a shot-clock violation.",
        ),
        "away_scv": (
            "The robot opponent runs out of shot clock; possession goes to the player.",
            "Shot clock expires on the robot opponent — possession flips.",
            "The robot opponent commits a shot-clock violation.",
        ),
        "unknown_scv": (
            "Shot clock expires and possession changes.",
            "Shot-clock violation — possession changes.",
            "The shot clock expires; possession resets.",
        ),
        "home_oob_pressure": (
            "The player attacks under pressure, but the ball goes out of bounds and the robot opponent takes possession.",
            "Under pressure, the player sends it out — robot opponent ball.",
            "Pressure forces the player out of bounds.",
        ),
        "home_oob_drive": (
            "The player drives into the lane, but the ball goes out of bounds and the robot opponent takes possession.",
            "The player's drive ends out of bounds — robot opponent ball.",
            "Drive out of bounds on the player.",
        ),
        "home_oob_shot": (
            "The player releases the shot, the ball goes out of bounds, and the robot opponent takes possession.",
            "The player's shot goes out of bounds — robot opponent ball.",
            "Shot out of bounds on the player.",
        ),
        "away_oob_pressure": (
            "The robot opponent attacks under pressure, but the ball goes out of bounds and the player takes possession.",
            "Under pressure, the robot opponent sends it out — player ball.",
            "Pressure forces the robot opponent out of bounds.",
        ),
        "away_oob_drive": (
            "The robot opponent drives into the lane, but the ball goes out of bounds and the player takes possession.",
            "The robot opponent's drive ends out of bounds — player ball.",
            "Drive out of bounds on the robot opponent.",
        ),
        "away_oob_shot": (
            "The robot opponent releases the shot, the ball goes out of bounds, and the player takes possession.",
            "The robot opponent's shot goes out of bounds — player ball.",
            "Shot out of bounds on the robot opponent.",
        ),
    },
    "hype": {
        "home_score": (
            "The player attacks the basket and scores!",
            "Yes! The player scores!",
            "Got it! The player finishes and scores!",
        ),
        "away_score": (
            "The robot opponent attacks the basket and scores!",
            "Yes! The robot opponent scores!",
            "Got it! The robot opponent finishes and scores!",
        ),
        "unknown_score": (
            "It is good — basket confirmed!",
            "Yes — that basket counts!",
            "Confirmed — it is good!",
        ),
        "home_oob": (
            "The player sends it out of bounds — possession flips!",
            "Out! The player loses it out of bounds!",
            "Out of bounds on the player — flip the ball!",
        ),
        "away_oob": (
            "The robot opponent sends it out — player ball!",
            "Out! The robot opponent turns it over!",
            "Out of bounds on the opponent — player ball!",
        ),
        "unknown_oob": (
            "Out of bounds — possession resets!",
            "It is out — reset possession!",
            "Out of bounds — start again!",
        ),
        "home_scv": (
            "Shot clock expires on the player — possession flips!",
            "Time! The player burns the shot clock!",
            "Shot-clock violation on the player!",
        ),
        "away_scv": (
            "Shot clock expires on the robot opponent — player ball!",
            "Time! The robot opponent burns the shot clock!",
            "Shot-clock violation on the opponent!",
        ),
        "unknown_scv": (
            "Shot clock expires — possession changes!",
            "Time's up — possession changes!",
            "Shot-clock violation — flip it!",
        ),
        "home_oob_pressure": (
            "The player attacks under pressure, but the ball goes out of bounds and the robot opponent takes possession!",
            "Pressure gets them — player out of bounds!",
            "Forced out! Player loses it under pressure!",
        ),
        "home_oob_drive": (
            "The player drives into the lane, but the ball goes out of bounds and the robot opponent takes possession!",
            "Drive out of bounds — player turns it over!",
            "The drive spills out — possession flips!",
        ),
        "home_oob_shot": (
            "The player releases the shot, the ball goes out of bounds, and the robot opponent takes possession!",
            "Shot goes out — player loses possession!",
            "That shot sails out of bounds!",
        ),
        "away_oob_pressure": (
            "The robot opponent attacks under pressure, but the ball goes out of bounds and the player takes possession!",
            "Pressure gets them — opponent out of bounds!",
            "Forced out! Opponent loses it under pressure!",
        ),
        "away_oob_drive": (
            "The robot opponent drives into the lane, but the ball goes out of bounds and the player takes possession!",
            "Drive out of bounds — opponent turns it over!",
            "The drive spills out — player ball!",
        ),
        "away_oob_shot": (
            "The robot opponent releases the shot, the ball goes out of bounds, and the player takes possession!",
            "Shot goes out — opponent loses possession!",
            "That shot sails out of bounds — player ball!",
        ),
    },
    "calm": {
        "home_score": (
            "The player attacks the basket and completes the score.",
            "The player completes the score.",
            "The player converts the attack.",
        ),
        "away_score": (
            "The robot opponent attacks the basket and completes the score.",
            "The robot opponent completes the score.",
            "The robot opponent converts the attack.",
        ),
        "unknown_score": (
            "A basket is confirmed.",
            "The basket is confirmed.",
            "Score confirmed.",
        ),
        "home_oob": (
            "The player sends the ball out of bounds; possession goes to the robot opponent.",
            "Out of bounds on the player; possession changes.",
            "The player goes out of bounds.",
        ),
        "away_oob": (
            "The robot opponent sends the ball out of bounds; possession goes to the player.",
            "Out of bounds on the robot opponent; possession changes.",
            "The robot opponent goes out of bounds.",
        ),
        "unknown_oob": (
            "The ball goes out of bounds and possession resets.",
            "Out of bounds; possession resets.",
            "The ball is out of bounds.",
        ),
        "home_scv": (
            "The player exhausts the shot clock; possession goes to the robot opponent.",
            "Shot clock expires on the player.",
            "The player commits a shot-clock violation.",
        ),
        "away_scv": (
            "The robot opponent exhausts the shot clock; possession goes to the player.",
            "Shot clock expires on the robot opponent.",
            "The robot opponent commits a shot-clock violation.",
        ),
        "unknown_scv": (
            "The shot clock expires and possession changes.",
            "Shot clock expires; possession changes.",
            "Shot-clock violation.",
        ),
        "home_oob_pressure": (
            "The player attacks under pressure, but the ball goes out of bounds and the robot opponent takes possession.",
            "Pressure forces the player out of bounds.",
            "Under pressure, the player goes out of bounds.",
        ),
        "home_oob_drive": (
            "The player drives into the lane, but the ball goes out of bounds and the robot opponent takes possession.",
            "The player's drive ends out of bounds.",
            "Drive out of bounds on the player.",
        ),
        "home_oob_shot": (
            "The player releases the shot, the ball goes out of bounds, and the robot opponent takes possession.",
            "The player's shot goes out of bounds.",
            "Shot out of bounds on the player.",
        ),
        "away_oob_pressure": (
            "The robot opponent attacks under pressure, but the ball goes out of bounds and the player takes possession.",
            "Pressure forces the robot opponent out of bounds.",
            "Under pressure, the robot opponent goes out of bounds.",
        ),
        "away_oob_drive": (
            "The robot opponent drives into the lane, but the ball goes out of bounds and the player takes possession.",
            "The robot opponent's drive ends out of bounds.",
            "Drive out of bounds on the robot opponent.",
        ),
        "away_oob_shot": (
            "The robot opponent releases the shot, the ball goes out of bounds, and the player takes possession.",
            "The robot opponent's shot goes out of bounds.",
            "Shot out of bounds on the robot opponent.",
        ),
    },
    "trash_talk": {
        "home_score": (
            "The player attacks the basket and scores — that one counts!",
            "The player scores — take it!",
            "Player scores — nothing you can do!",
        ),
        "away_score": (
            "The robot opponent attacks the basket and scores — they got that one!",
            "Opponent scores — fine, they earned it!",
            "Robot opponent scores — give them that!",
        ),
        "unknown_score": (
            "Basket confirmed — it is good!",
            "It counts — basket good!",
            "Confirmed score — yes!",
        ),
        "home_oob": (
            "The player boots it out of bounds — gift possession!",
            "Player out of bounds — free ball!",
            "Out of bounds — player just gave it away!",
        ),
        "away_oob": (
            "The robot opponent boots it out — player ball!",
            "Opponent out of bounds — thank you!",
            "Out of bounds — free possession!",
        ),
        "unknown_oob": (
            "Out of bounds — reset and go again!",
            "It is out — start over!",
            "Out of bounds — reset!",
        ),
        "home_scv": (
            "Shot clock burns the player — possession flips!",
            "Player wastes the clock — turnover!",
            "Shot-clock violation — player gift!",
        ),
        "away_scv": (
            "Shot clock burns the robot opponent — player ball!",
            "Opponent wastes the clock — thank you!",
            "Shot-clock violation — free ball!",
        ),
        "unknown_scv": (
            "Shot clock expires — possession changes!",
            "Clock's gone — flip possession!",
            "Shot-clock violation — reset!",
        ),
        "home_oob_pressure": (
            "The player attacks under pressure, but the ball goes out of bounds and the robot opponent takes possession!",
            "Pressure wins — player out of bounds!",
            "Forced out — player turns it over!",
        ),
        "home_oob_drive": (
            "The player drives into the lane, but the ball goes out of bounds and the robot opponent takes possession!",
            "Drive fails out of bounds — gift!",
            "Player drive spills out — turnover!",
        ),
        "home_oob_shot": (
            "The player releases the shot, the ball goes out of bounds, and the robot opponent takes possession!",
            "Shot sails out — player turnover!",
            "That shot is out — free ball!",
        ),
        "away_oob_pressure": (
            "The robot opponent attacks under pressure, but the ball goes out of bounds and the player takes possession!",
            "Pressure wins — opponent out of bounds!",
            "Forced out — opponent turns it over!",
        ),
        "away_oob_drive": (
            "The robot opponent drives into the lane, but the ball goes out of bounds and the player takes possession!",
            "Drive fails out of bounds — thank you!",
            "Opponent drive spills out — player ball!",
        ),
        "away_oob_shot": (
            "The robot opponent releases the shot, the ball goes out of bounds, and the player takes possession!",
            "Shot sails out — free possession!",
            "That shot is out — player ball!",
        ),
    },
}

# Recent fully-formatted P1 lines — used to avoid speaking the exact same
# sentence twice in a short window. Tests can clear this via reset_style_line_memory().
_RECENT_STYLE_LINES: deque[str] = deque(maxlen=12)
_STYLE_ROTATION: dict[str, int] = {}


def reset_style_line_memory() -> None:
    """Clear spoken-line memory (for tests / new inference sessions)."""
    _RECENT_STYLE_LINES.clear()
    _STYLE_ROTATION.clear()


def _as_variants(entry: _StyleLine | None) -> tuple[str, ...]:
    if entry is None:
        return ()
    if isinstance(entry, str):
        return (entry,)
    return tuple(entry)


def _pick_formatted_line(pool_key: str, templates: tuple[str, ...], **fmt: str) -> str | None:
    """Pick a formatted line, preferring ones not spoken recently."""
    if not templates:
        return None
    options = [t.format(**fmt) for t in templates]
    recent = set(_RECENT_STYLE_LINES)
    fresh = [line for line in options if line not in recent]
    pool = fresh if fresh else options
    idx = _STYLE_ROTATION.get(pool_key, 0) % len(pool)
    _STYLE_ROTATION[pool_key] = _STYLE_ROTATION.get(pool_key, 0) + 1
    chosen = pool[idx]
    _RECENT_STYLE_LINES.append(chosen)
    return chosen

_SCORE_CUE_RE = re.compile(r"\bscored\s*!", re.IGNORECASE)
_OOB_CUE_RE = re.compile(r"\bout\s+of\s+bounds\s*!", re.IGNORECASE)
_SHOT_CLOCK_CUE_RE = re.compile(r"\bshot\s*[- ]?\s*clock\s+violation\s*!", re.IGNORECASE)
_MAKE_CLAIM_RE = re.compile(
    r"\b(?:score[sd]?|scoring|bucket|buckets|swish(?:es)?|nylon|"
    r"successful(?:ly)?|success|converts?|sinks?|sank|"
    r"makes? (?:the |a )?(?:shot|basket|layup)|made (?:the |a )?(?:shot|basket|layup)|"
    r"drops? (?:home|through|in)|bottom of (?:the )?(?:net|cup)|"
    r"banks? in|puts? it through|finds? the mark|hits? the mark|"
    r"goes? through|went through|through the hoop|for a basket|nothing but net)\b",
    re.IGNORECASE,
)
# Unlike a make claim (risky to invent before the referee banner confirms it),
# a miss claim carries little downside: a wrong miss call is silently
# corrected a moment later if a Scored! banner actually appears.
_MISS_CLAIM_RE = re.compile(
    r"\b(?:miss(?:es|ed|ing)?|no good|off (?:the )?(?:rim|iron|back[- ]?board)|"
    r"rims? out|rattles? out|falls? short|comes? up short|airball|air ball|"
    r"clangs? off|doesn'?t (?:go in|fall)|bounces? (?:out|away))\b",
    re.IGNORECASE,
)
_OOB_CLAIM_RE = re.compile(r"\b(?:out of bounds|out of play|possession (?:is )?awarded)\b", re.IGNORECASE)
_ZH_MAKE_CLAIM_RE = re.compile(r"得分|進球|進了|命中|投進")
_ZH_MISS_CLAIM_RE = re.compile(r"沒進|未進|不進|沒中|沒有進|打鐵|彈框")
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
_DUNK_FINISH_RE = re.compile(
    r"\b(?:dunk|dunks|dunked|slam(?:s|med)?|jam(?:s|med)?)\b|灌籃|暴扣|猛扣",
    re.IGNORECASE,
)
_LAYUP_FINISH_RE = re.compile(
    r"\b(?:lay[- ]?ups?|finger\s*roll)\b|上籃",
    re.IGNORECASE,
)
# A bare "3" is deliberately excluded: scoreboard digits leaking into the draft
# would otherwise be read as a three-pointer.
_THREE_FINISH_RE = re.compile(
    r"\b(?:three(?:[- ]point(?:er)?)?|3[- ]point(?:er)?s?|"
    r"from (?:deep|downtown)|beyond the arc)\b|三分",
    re.IGNORECASE,
)
_JUMPER_FINISH_RE = re.compile(
    r"\b(?:jump(?:er| shot)|pull[- ]up|fadeaway)\b|跳投",
    re.IGNORECASE,
)
# A fake keeps the ball in the ballhandler's hands, so "fakes the jumper" must
# not be read as a jumper. The optional trailing noun is consumed together with
# the fake verb so the faked move disappears from the evidence blob as well.
_FAKE_MOVE_RE = re.compile(
    r"\b(?:pump|shot|ball|jab|hesitation|up[- ]and[- ]under)?[- ]?"
    r"(?:fake[sd]?|faking|hesitat(?:e[sd]?|ing|ion)|jab[- ]steps?|shimm(?:y|ies|ied))"
    r"(?:\s+(?:at|on|out|into)?\s*(?:a|an|the|his|its)?\s*"
    r"(?:jump(?:er| shot)?|shot|dunk|slam|lay[- ]?up|three|pass|handoff))?"
    r"|假動作|試探步|晃(?:動|開)?",
    re.IGNORECASE,
)
_FINISH_PATTERNS = (
    ("dunk", _DUNK_FINISH_RE),
    ("layup", _LAYUP_FINISH_RE),
    ("three", _THREE_FINISH_RE),
    ("jumper", _JUMPER_FINISH_RE),
)
_SUBJECT_RE = re.compile(r"^(the player|the robot opponent)\b", re.IGNORECASE)
_LEADING_COMMA_RE = re.compile(r"^(the player|the robot opponent)\s*,\s*", re.IGNORECASE)
_LEADING_CONNECTOR_RE = re.compile(
    r"^(the player|the robot opponent)\s*,?\s*(?:and|then)\s+", re.IGNORECASE
)
# LiveCC / caption cues that unambiguously name a fake / hesitation move.
# These alone justify FORCING 假動作 wording into Gemini's own draft even when
# it chose different words, because the source caption is explicit.
_FAKE_STRONG_EVIDENCE_RE = re.compile(
    r"\b(?:pump[- ]?fake[sd]?|shot[- ]?fake[sd]?|ball[- ]?fake[sd]?|head[- ]?fake[sd]?|"
    r"fake[sd]?|faking|hesitat(?:e[sd]?|ing|ion)|jab[- ]steps?|shimm(?:y|ies|ied))\b",
    re.IGNORECASE,
)
_FAKE_STRONG_EVIDENCE_ZH_RE = re.compile(
    r"假動作|試探步|假投|猶豫步|假晃"
)
# Weaker cues ("sizes up", body rock/sway, side-to-side) that a VLM sometimes
# uses loosely for ordinary ball-handling in front of a defender. On their own
# they must NOT force fake wording into a draft that never mentioned it — that
# over-triggered on routine standoffs and made "假動作" show up on nearly every
# background line. They still count as motion (not "quiet holding"), and an
# existing fake mention in the draft is left alone.
_FAKE_WEAK_EVIDENCE_RE = re.compile(
    r"\b(?:size[- ]?ups?|sizes up|"
    r"rock(?:s|ing)?(?:\s+(?:the\s+)?(?:ball|body|defender))?|"
    r"sway(?:s|ing)?|side[- ]to[- ]side)\b",
    re.IGNORECASE,
)
_FAKE_WEAK_EVIDENCE_ZH_RE = re.compile(r"晃球|左右晃|晃動|晃身|拉鋸")
# Pure quiet possession with no motion cue — strip invented fake wording here.
_QUIET_HOLD_RE = re.compile(
    r"\b(?:holds? (?:the )?ball|looking for an opening|"
    r"faces? (?:the )?(?:defender|opponent|each other)|"
    r"protects? (?:the )?ball|waits?(?:\s+for)?|observes? (?:the )?defen)\b|"
    r"持球觀察|尋找切入|雙方對峙|尋找進攻節奏|冷靜觀察",
    re.IGNORECASE,
)
# Fake claim tokens used for strip / inject checks.
_FAKE_CLAIM_ZH_RE = re.compile(
    r"用假動作(?:拉鋸)?|假動作(?:拉鋸)?|試探步|左右晃(?:對手)?"
)
_FAKE_CLAIM_EN_RE = re.compile(
    r"\b(?:fakes?(?:\s+and)?|faking|with a size-up fake|fake standoff|"
    r"size[- ]?ups?|hesitat(?:e[sd]?|ing|ion)|jab[- ]steps?)\b",
    re.IGNORECASE,
)
_FAKE_WORD_ZH_RE = re.compile(r"假動作|試探步|左右晃|假投|晃球|假晃")


def compose_result_evidence(result_text: str, recent_action: str = "") -> str:
    """Attach prior visible action without weakening the exact result cue."""
    action = " ".join(recent_action.split()).strip()
    return f"Previous visible action: {action}\n{result_text}" if action else result_text


def strip_fake_moves(text: str) -> str:
    """Drop fake / hesitation / size-up wording before reading a real attempt."""
    return _FAKE_MOVE_RE.sub(" ", text)


def _normalize_actor_wording(text: str) -> str:
    """Apply the standard English robot/opponent/player naming fixes.

    Shared by the broadcast draft and by raw LiveCC evidence text, so both
    can be safely used as the base sentence for grounded commentary.
    """
    text = re.sub(
        r"\bkicks? (?:the ball|it) back out\b",
        "retreats to the perimeter",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^Player\b", "The player", text)
    text = re.sub(r"^The robot\b(?!\s+opponent)", "The robot opponent", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe robot\b(?!\s+opponent)", "the robot opponent", text, flags=re.IGNORECASE)
    text = re.sub(r"\btest_?bot\d*\b", "the robot opponent", text, flags=re.IGNORECASE)
    text = re.sub(r"^The opponent\b", "The robot opponent", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!robot )\bthe opponent\b", "the robot opponent", text, flags=re.IGNORECASE)
    text = re.sub(r"^Players\b", "The player and robot opponent", text, flags=re.IGNORECASE)
    return text


def _inject_fake_zh(text: str) -> str:
    if _FAKE_WORD_ZH_RE.search(text):
        return text
    if "持球" in text:
        return text.replace("持球", "持球用假動作", 1)
    return text.rstrip("。！?") + "，用假動作拉鋸。"


def _inject_fake_en(text: str) -> str:
    if _FAKE_CLAIM_EN_RE.search(text):
        return text
    match = re.match(r"^(The player|The robot opponent)\s+", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(0)}fakes and {text[match.end():]}"
    return f"Fakes — {text}"


def _strip_fake_claims(text: str, zh: bool) -> str:
    if zh:
        text = _FAKE_CLAIM_ZH_RE.sub("", text)
        text = re.sub(r"，{2,}", "，", text)
        text = re.sub(r"。{2,}", "。", text)
        text = text.strip("，。 ")
        if text and not text.endswith(("。", "！", "？", "!")):
            text += "。"
        return text
    text = _FAKE_CLAIM_EN_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    return text.strip()


def _ground_fake_wording(broadcast_text: str, raw_visual_text: str, zh: bool) -> str:
    """Gate fake wording on motion evidence — not quiet holding alone.

    - LiveCC names an explicit fake/jab/hesitation (strong evidence) → force
      keep or inject 假動作. The caption is unambiguous, so trust it.
    - LiveCC only says "sizes up" / rocks / sways (weak evidence) → this is
      real motion, but VLMs use these words loosely for routine ball-handling
      in front of a defender. Do NOT force-inject fake wording the Gemini
      draft never chose; only keep it if the draft already said so. Forcing
      on weak evidence alone is what made 假動作 show up on nearly every
      background line during ordinary standoffs.
    - Quiet hold / stare only → strip invented fake claims.
    - Otherwise trust the Gemini draft (it may have seen rocking in frames
      even when LiveCC only said "dribbles").
    """
    if not broadcast_text:
        return broadcast_text
    has_strong = bool(
        _FAKE_STRONG_EVIDENCE_RE.search(raw_visual_text)
        or _FAKE_STRONG_EVIDENCE_ZH_RE.search(raw_visual_text)
    )
    if has_strong:
        return _inject_fake_zh(broadcast_text) if zh else _inject_fake_en(broadcast_text)
    has_weak = bool(
        _FAKE_WEAK_EVIDENCE_RE.search(raw_visual_text)
        or _FAKE_WEAK_EVIDENCE_ZH_RE.search(raw_visual_text)
    )
    if has_weak:
        # Motion is real, but not decisive enough to force wording the draft
        # itself didn't choose. Leave the draft exactly as Gemini wrote it —
        # this also means an existing fake mention is naturally kept.
        return broadcast_text
    quiet_only = bool(_QUIET_HOLD_RE.search(raw_visual_text))
    if quiet_only:
        stripped = _strip_fake_claims(broadcast_text, zh)
        return stripped or broadcast_text
    return broadcast_text


def _latest_finish(blob: str) -> str:
    """Return the finishing move mentioned last, i.e. closest to the result.

    A caption may walk through a sequence ("fakes the three, then dunks it"),
    so the last surviving mention describes how the ball actually went in.
    """
    hits = [
        (match.start(), name)
        for name, pattern in _FINISH_PATTERNS
        for match in pattern.finditer(blob)
    ]
    return max(hits)[1] if hits else ""


def pick_result_action(
    recent_actions: Sequence[str], *, for_score: bool = False
) -> str:
    """Choose the caption that explains a freshly detected result banner.

    Banners are read straight off the raw frame while LiveCC runs a second or
    more behind, so the newest caption at banner time is often the pre-shot
    size-up while the actual attempt sits one caption earlier. Prefer the newest
    caption that names a real attempt over the newest caption overall.

    ``recent_actions`` must be ordered newest first.

    When ``for_score`` is True (confirmed Scored! banner), skip miss-claim
    captions and prefer dunk → layup → three → jumper so a lagged dunk line
    outranks an earlier "jump shot but misses".
    """
    candidates = [action for action in recent_actions if action.strip()]
    if not candidates:
        return ""
    if for_score:
        non_miss = [
            action
            for action in candidates
            if not (
                _MISS_CLAIM_RE.search(action) or _ZH_MISS_CLAIM_RE.search(action)
            )
        ]
        # Never ground a confirmed make on a miss caption alone.
        if not non_miss:
            return ""
        for finish_name in ("dunk", "layup", "three", "jumper"):
            for action in non_miss:
                if _latest_finish(strip_fake_moves(action)) == finish_name:
                    return action
        for action in non_miss:
            if _SHOT_CONTEXT_RE.search(strip_fake_moves(action)):
                return action
        return non_miss[0]
    for action in candidates:
        if _latest_finish(strip_fake_moves(action)):
            return action
    for action in candidates:
        if _SHOT_CONTEXT_RE.search(strip_fake_moves(action)):
            return action
    return candidates[0]


def finish_type(text: str) -> str:
    """Return dunk / layup / three / jumper if ``text`` names a finish."""
    return _latest_finish(strip_fake_moves(text or ""))


def _result_context(broadcast_text: str, raw_visual_text: str) -> str:
    cue_matches = [
        *_SCORE_CUE_RE.finditer(raw_visual_text),
        *_OOB_CUE_RE.finditer(raw_visual_text),
        *_SHOT_CLOCK_CUE_RE.finditer(raw_visual_text),
    ]
    evidence = raw_visual_text[: max((match.start() for match in cue_matches), default=0)]
    blob = strip_fake_moves(f"{broadcast_text}\n{evidence}")
    if _PRESSURE_CONTEXT_RE.search(blob):
        return "pressure"
    if _DRIVE_CONTEXT_RE.search(blob):
        return "drive"
    if _SHOT_CONTEXT_RE.search(blob):
        return "shot"
    return ""


def _score_finish_type(broadcast_text: str, raw_visual_text: str) -> str:
    """Infer dunk / layup / three / jumper from Gemini text + prior LiveCC action.

    The referee banner only proves that someone scored; the finish style comes
    from the preceding visible action (and Gemini's draft) so we do not wipe
    dunk/layup/jumper details with a generic "attacks the basket and scores".
    Fakes are stripped first: a faked jumper that ends at the rim is a drive.
    Miss-claim captions must not supply the finish for a confirmed make.
    """
    evidence = _score_evidence(raw_visual_text)
    if _MISS_CLAIM_RE.search(evidence) or _ZH_MISS_CLAIM_RE.search(evidence):
        evidence = ""
    blob = strip_fake_moves(f"{broadcast_text}\n{evidence}")
    return _latest_finish(blob) or _result_context(broadcast_text, raw_visual_text)


def _score_result_text(
    side: str | None, finish: str, zh: bool, *, style: str = "objective"
) -> str | None:
    """Build a minimal who-scored sentence that still names the finish style.

    This is only the last-resort fallback when there is no usable real
    visual description to ground on (see ``_ground_score_evidence_text``).
    Facts stay fixed; wording rotates across a small style pool so the same
    finish is not spoken with an identical sentence twice in a row.
    """
    actor_en = "the player" if side == "home" else "the robot opponent" if side == "away" else ""
    actor_zh = "玩家" if side == "home" else "機器人對手" if side == "away" else ""
    if not actor_en:
        return None
    pool_key = f"score:{style}:{finish}:{'zh' if zh else 'en'}:{side}"
    if zh:
        styled = _as_variants(_STYLE_SCORE_FALLBACK_ZH.get(style, {}).get(finish))
        fallback = _as_variants(_SCORE_FALLBACK_ZH.get(finish))
        templates = styled or fallback
        return _pick_formatted_line(pool_key, templates, actor=actor_zh)
    styled = _as_variants(_STYLE_SCORE_FALLBACK_EN.get(style, {}).get(finish))
    fallback = _as_variants(_SCORE_FALLBACK_EN.get(finish))
    templates = styled or fallback
    if not templates:
        return None
    cap = actor_en[0].upper() + actor_en[1:]
    return _pick_formatted_line(pool_key, templates, Actor=cap, actor=actor_en)


def _generic_result_line(key: str, zh: bool, *, style: str = "objective") -> str | None:
    """Return a style-flavored generic P1 line, or None to use the objective constant."""
    table = _STYLE_GENERIC_ZH if zh else _STYLE_GENERIC_EN
    # Prefer the active style pool; fall back to objective variants so even the
    # default style can rotate wording instead of repeating one fixed sentence.
    templates = _as_variants(table.get(style, {}).get(key))
    if not templates:
        templates = _as_variants(table.get("objective", {}).get(key))
    if not templates:
        return None
    return _pick_formatted_line(f"generic:{style}:{key}:{'zh' if zh else 'en'}", templates)


def _score_evidence(raw_visual_text: str) -> str:
    """Return the text LiveCC captured right before the last Scored! cue."""
    cue_matches = list(_SCORE_CUE_RE.finditer(raw_visual_text))
    text = raw_visual_text[: cue_matches[-1].start()] if cue_matches else raw_visual_text
    return re.sub(r"^\s*previous visible action:\s*", "", text, flags=re.IGNORECASE).strip()


def _ground_score_evidence_text(evidence: str, side: str | None, zh: bool) -> str | None:
    """Build the scoring sentence from what LiveCC actually saw, not a template.

    The confirmed banner only proves someone scored; everything else in the
    sentence - the drive, the contest, the exact finish - should be the real
    pre-shot action LiveCC reported. We only enforce two things: the actor
    (must match the confirmed banner side, since the banner outranks a
    possibly-stale caption actor) and the scoring outcome. Returns ``None``
    when there is nothing usable to ground on (empty, garbled, hallucinated
    equipment/teamwork talk, or no attacking action at all), so the caller
    falls back to a minimal templated line instead.

    English only: the evidence is LiveCC's English caption. There is no
    reliable rule-based way to translate arbitrary free text into natural
    Chinese, so the zh path always uses the deterministic finish template.
    """
    if zh or side is None:
        return None
    text = _normalize_actor_wording(evidence)
    text = strip_fake_moves(text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,")
    text = _LEADING_COMMA_RE.sub(r"\1 ", text)
    text = _LEADING_CONNECTOR_RE.sub(r"\1 ", text)
    text = text.strip(" ,")
    if not text or _UI_META_CLAIM_RE.search(text) or _TEAMWORK_CLAIM_RE.search(text):
        return None
    if not (_latest_finish(text) or _SHOT_CONTEXT_RE.search(text) or _DRIVE_CONTEXT_RE.search(text)):
        return None
    if not _SUBJECT_RE.match(text):
        # Unrecognized sentence shape - do not risk forcing a subject onto it.
        return None

    actor_en = "the player" if side == "home" else "the robot opponent"
    cap = actor_en[0].upper() + actor_en[1:]
    text = _SUBJECT_RE.sub(cap, text, count=1)
    if _MAKE_CLAIM_RE.search(text):
        return text if text.endswith((".", "!")) else f"{text}."
    return f"{text.rstrip('.!')} and scores."


def _contextual_result_text(
    cue: str, side: str | None, context: str, zh: bool, *, style: str = "objective"
) -> str | None:
    actor_en = "the player" if side == "home" else "the robot opponent" if side == "away" else ""
    actor_zh = "玩家" if side == "home" else "機器人對手" if side == "away" else ""
    if cue == "score":
        return _score_result_text(side, context, zh, style=style)
    if cue == "out_of_bounds" and context:
        ctx_key = (
            "pressure" if context == "pressure"
            else "drive" if context == "drive"
            else "shot"
        )
        if side in ("home", "away"):
            styled = _generic_result_line(f"{side}_oob_{ctx_key}", zh, style=style)
            if styled:
                return styled
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
        *(
            (match.start(), "shot_clock_violation")
            for match in _SHOT_CLOCK_CUE_RE.finditer(raw_visual_text)
        ),
    ]
    return max(cues)[1] if cues else None


def result_side(raw_visual_text: str, cue: str | None = None) -> str | None:
    """Return Home/Away for the latest explicit result banner, never the clock."""
    cue = cue or result_cue(raw_visual_text)
    pattern = (
        _SCORE_CUE_RE
        if cue == "score"
        else _OOB_CUE_RE
        if cue == "out_of_bounds"
        else _SHOT_CLOCK_CUE_RE
        if cue == "shot_clock_violation"
        else None
    )
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
    broadcast_text: str,
    raw_visual_text: str,
    *,
    language: str = "en",
    style: str = "objective",
) -> str:
    zh = language == "zh"
    style = (style or "objective").strip() or "objective"
    if not zh:
        # In one-on-one play, "kick it back out" falsely implies a pass. The
        # observed action is the same ballhandler retreating to the perimeter.
        broadcast_text = _normalize_actor_wording(broadcast_text)
    cue = result_cue(raw_visual_text)
    if cue == "score":
        side = result_side(raw_visual_text, cue)
        score = result_score(raw_visual_text)
        grounded = _ground_score_evidence_text(_score_evidence(raw_visual_text), side, zh)
        if grounded:
            return _append_score(grounded, score, zh)
        finish = _score_finish_type(broadcast_text, raw_visual_text)
        contextual = _contextual_result_text(cue, side, finish, zh, style=style)
        if contextual:
            return _append_score(contextual, score, zh)
        if side == "home":
            line = _generic_result_line("home_score", zh, style=style)
            return _append_score(
                line or (PLAYER_SCORE_TEXT_ZH if zh else PLAYER_SCORE_TEXT), score, zh
            )
        if side == "away":
            line = _generic_result_line("away_score", zh, style=style)
            return _append_score(
                line or (OPPONENT_SCORE_TEXT_ZH if zh else OPPONENT_SCORE_TEXT), score, zh
            )
        line = _generic_result_line("unknown_score", zh, style=style)
        return _append_score(
            line or (UNKNOWN_SCORE_TEXT_ZH if zh else UNKNOWN_SCORE_TEXT), score, zh
        )
    if cue == "out_of_bounds":
        side = result_side(raw_visual_text, cue)
        contextual = _contextual_result_text(
            cue, side, _result_context(broadcast_text, raw_visual_text), zh, style=style
        )
        if contextual:
            return contextual
        if side == "home":
            return (
                _generic_result_line("home_oob", zh, style=style)
                or (PLAYER_OOB_CALL_TEXT_ZH if zh else PLAYER_OOB_CALL_TEXT)
            )
        if side == "away":
            return (
                _generic_result_line("away_oob", zh, style=style)
                or (OPPONENT_OOB_CALL_TEXT_ZH if zh else OPPONENT_OOB_CALL_TEXT)
            )
        return (
            _generic_result_line("unknown_oob", zh, style=style)
            or (UNKNOWN_OUT_OF_BOUNDS_TEXT_ZH if zh else UNKNOWN_OUT_OF_BOUNDS_TEXT)
        )
    if cue == "shot_clock_violation":
        side = result_side(raw_visual_text, cue)
        if side == "home":
            return (
                _generic_result_line("home_scv", zh, style=style)
                or (PLAYER_SHOT_CLOCK_TEXT_ZH if zh else PLAYER_SHOT_CLOCK_TEXT)
            )
        if side == "away":
            return (
                _generic_result_line("away_scv", zh, style=style)
                or (OPPONENT_SHOT_CLOCK_TEXT_ZH if zh else OPPONENT_SHOT_CLOCK_TEXT)
            )
        return (
            _generic_result_line("unknown_scv", zh, style=style)
            or (UNKNOWN_SHOT_CLOCK_TEXT_ZH if zh else UNKNOWN_SHOT_CLOCK_TEXT)
        )
    if _UI_META_CLAIM_RE.search(broadcast_text):
        if _opponent_is_primary(raw_visual_text):
            return OPPONENT_CONTROL_TEXT_ZH if zh else OPPONENT_CONTROL_TEXT
        return PLAYER_CONTROL_TEXT_ZH if zh else PLAYER_CONTROL_TEXT
    if _TEAMWORK_CLAIM_RE.search(broadcast_text):
        if _opponent_is_primary(raw_visual_text):
            return OPPONENT_CONTROL_TEXT_ZH if zh else OPPONENT_CONTROL_TEXT
        return PLAYER_CONTROL_TEXT_ZH if zh else PLAYER_CONTROL_TEXT
    make_claim = _MAKE_CLAIM_RE.search(broadcast_text) or _ZH_MAKE_CLAIM_RE.search(broadcast_text)
    miss_claim = _MISS_CLAIM_RE.search(broadcast_text) or _ZH_MISS_CLAIM_RE.search(broadcast_text)
    if miss_claim and not make_claim:
        # No banner ever confirms a miss, but reporting one is low-risk: a
        # basket that actually went in is corrected by the next Scored! cue.
        # broadcast_text is already the real, actor-normalized LiveCC/Gemini
        # description of the miss, so just speak it - no need to invent or
        # pick from a canned line when the real one is right here.
        return broadcast_text.strip()
    if make_claim:
        # An unconfirmed make claim stays neutral until a Scored! banner
        # actually appears, so we never invent a basket ahead of the referee.
        return NEUTRAL_SHOT_TEXT_ZH if zh else NEUTRAL_SHOT_TEXT
    if _OOB_CLAIM_RE.search(broadcast_text) or _ZH_OOB_CLAIM_RE.search(broadcast_text):
        if _opponent_is_primary(raw_visual_text):
            return OPPONENT_CONTROL_TEXT_ZH if zh else OPPONENT_CONTROL_TEXT
        return PLAYER_CONTROL_TEXT_ZH if zh else PLAYER_CONTROL_TEXT
    return _ground_fake_wording(broadcast_text.strip(), raw_visual_text, zh)
