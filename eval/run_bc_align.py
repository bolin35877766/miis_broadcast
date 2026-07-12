#!/usr/bin/env python3
"""
eval/run_bc_align.py — BC-Align: Broadcast Commentary Alignment Evaluation

Evaluates whether the FULL broadcast pipeline output (LiveCC -> Gemini ->
TTS playback text, as recorded in log/combination_output.log) is semantically
aligned with the human-annotated Ground Truth (eval/groundTruth.json), using
an LLM as judge.

This adapts the LLM-as-judge protocol from LiveCC (Chen et al., CVPR 2025,
evaluation/livesports3kcc/llm_judge.py) from PAIRWISE model-vs-baseline
comparison to REFERENCE-BASED single-pipeline alignment scoring, and adds
an explicit temporal overlap step because our commentary is produced by a
live streaming pipeline (sparse, variable-length utterances) rather than by
offline inference over a pre-cut [begin, end] clip as in the original paper.

────────────────────────────────────────────────────────────────────────────
DESIGN NOTES (read before modifying the pipeline below)
────────────────────────────────────────────────────────────────────────────

1) Why "runs" must be auto-detected first
   -----------------------------------------
   log/combination_output.log accumulates every broadcast session ever run
   (multiple dates, multiple restarts of the same video). Ground truth event
   boundaries (`begin`/`end`) are only valid within ONE contiguous playback
   of the annotated video, so we must isolate a single run before any time
   alignment happens. A run boundary is detected whenever a segment's start
   time drops well below the running maximum start time seen so far
   (`_RESET_GAP_SEC`) — i.e. the video timeline restarted. This is far more
   reliable than grouping by wall-clock gaps, because TTS playback can pause
   for long stretches mid-run (Gemini/network latency) yet the log's video
   timeline should still be monotonically increasing.

2) Why assignment uses "max overlap wins", not strict containment
   -----------------------------------------------------------------
   A commentary segment [t_s, t_e] is assigned to the ground-truth event
   [begin, end] with which it has the largest temporal intersection,
   provided that intersection is > 0. This is a relaxed form of temporal
   IoU matching (cf. temporal action localization literature): we do not
   require the commentary segment to fall entirely inside the event window,
   only that it overlaps it more than any other event. Ties (equal overlap
   with two events) resolve to the lower `event_id`, since earlier events
   are evaluated first and a tie almost always means the segment sits
   exactly on an event boundary.

3) Why uncovered events are NOT sent to the judge
   -------------------------------------------------
   If no commentary segment overlaps a given event at all, there is nothing
   for the LLM to compare — an empty prediction is unambiguously score=1
   ("mostly wrong / nothing said"). Scoring it programmatically (instead of
   spending an API call on a degenerate case) keeps the metric well-defined
   and avoids judge non-determinism inflating a trivial case.

4) Why the score is a single macro-average (BC-Align Score), not a bundle
   of metrics
   -----------------------------------------------------------------------
   Mirroring LiveCC's own choice to report a single Win Rate number, we
   report a single BC-Align Score = mean(score_e) over all N ground-truth
   events, where uncovered events contribute score=1. This bakes coverage
   directly into the headline number (a pipeline that never speaks cannot
   score well) while still being a single, comparable scalar across runs.
   Coverage rate is still recorded per-event for diagnostics, but it is not
   a second headline metric.

5) Judge protocol
   ----------------
   - Reference-based (not pairwise): the judge only ever sees ONE prediction
     against the ground truth for that event, adapting LiveCC's "Semantic
     Alignment" criterion and dropping its pairwise "Stylistic Consistency"
     criterion (irrelevant when there is no second candidate to compare
     style against, and broadcast style is expected to differ from ASR
     transcripts).
   - Team-agnostic action matching: do NOT penalize when commentary attributes
     an action to "our player" vs "the opponent" (or "he"/"home side") if the
     underlying basketball action (steal, dribble, shot, score, miss, rebound,
     out of bounds, etc.) still corresponds to something in the ground truth.
     Still penalize contradictory OUTCOMES (e.g. score vs airball/miss).
   - temperature=0 and a fixed rubric (1-5) for reproducibility, matching
     LiveCC's judge configuration (seed=42, temperature=0).
   - Structured JSON output (score + one-sentence reason) rather than free
     text, so results are machine-parseable and auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# Windows terminals often default to a legacy codepage (e.g. cp950/cp936)
# rather than UTF-8, which garbles Chinese broadcast text when printed to
# the console. This only affects display; parsed data and JSON output are
# always UTF-8 regardless. Best-effort only (older Python lacks reconfigure).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = Path(__file__).resolve().parent

DEFAULT_LOG_PATH = _PROJECT_ROOT / "log" / "combination_output.log"
DEFAULT_GT_PATH = _EVAL_DIR / "groundTruth.json"
DEFAULT_RESULTS_DIR = _EVAL_DIR / "results"

# Run-boundary detection: a new run starts when a segment's start time is
# more than this many seconds BELOW the running max start time of the
# current run (i.e. the video timeline restarted). See design note (1).
DEFAULT_RESET_GAP_SEC = 2.0

DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_JUDGE_DELAY_SEC = 0.3  # throttle between API calls


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    """One broadcast utterance parsed from combination_output.log."""
    start: float
    end: float
    text: str
    wall_clock: str
    line_no: int


@dataclass
class GTEvent:
    event_id: int
    begin: float
    end: float
    gt_asr_text: str
    video_title: str = ""


@dataclass
class EventResult:
    event_id: int
    begin: float
    end: float
    gt_asr_text: str
    pred_text: str
    covered: bool
    num_segments: int
    score: Optional[int]
    reason: str
    judge_error: bool = False


# ──────────────────────────────────────────────────────────────────────────
# Step 1: parse combination_output.log
# ──────────────────────────────────────────────────────────────────────────

_LINE_RE = re.compile(
    r"^\[(?P<wall>[\d\-: ]+)\]\s*"
    r"\[(?P<sm>\d+):(?P<ss>\d+\.\d+)-(?P<em>\d+):(?P<es>\d+\.\d+)\]\s*"
    r"(?P<text>.+?)\s*$"
)


def parse_log(path: Path) -> list[Segment]:
    """Parse every '[wall-clock] [mm:ss.ss-mm:ss.ss] text' line in the log."""
    segments: list[Segment] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            m = _LINE_RE.match(raw_line.strip())
            if not m:
                continue
            start = int(m["sm"]) * 60 + float(m["ss"])
            end = int(m["em"]) * 60 + float(m["es"])
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=m["text"].strip(),
                    wall_clock=m["wall"],
                    line_no=line_no,
                )
            )
    return segments


# ──────────────────────────────────────────────────────────────────────────
# Step 2: split into contiguous "runs" and select one
# ──────────────────────────────────────────────────────────────────────────

def split_into_runs(
    segments: list[Segment], reset_gap: float = DEFAULT_RESET_GAP_SEC
) -> list[list[Segment]]:
    """Group segments into contiguous playback runs.

    A new run begins whenever the video timeline goes backwards by more
    than `reset_gap` seconds relative to the max start time seen so far in
    the current run. See module design note (1).
    """
    runs: list[list[Segment]] = []
    current: list[Segment] = []
    max_start_seen = -1.0

    for seg in segments:
        if current and seg.start < max_start_seen - reset_gap:
            runs.append(current)
            current = []
            max_start_seen = -1.0
        current.append(seg)
        max_start_seen = max(max_start_seen, seg.start)

    if current:
        runs.append(current)
    return runs


def describe_runs(runs: list[list[Segment]]) -> str:
    lines = []
    for i, run in enumerate(runs):
        span = run[-1].end - run[0].start
        lines.append(
            f"  run #{i}: {len(run):3d} lines, "
            f"log lines {run[0].line_no}-{run[-1].line_no}, "
            f"video {run[0].start:.1f}s-{run[-1].end:.1f}s (span={span:.1f}s), "
            f"wall_clock={run[0].wall_clock} ~ {run[-1].wall_clock}"
        )
    return "\n".join(lines)


def select_run(
    runs: list[list[Segment]], run_index: "int | str"
) -> list[Segment]:
    if run_index == "auto":
        # Default heuristic: the run with the most utterances is almost
        # always the single complete playthrough of the annotated video.
        return max(runs, key=len)
    idx = int(run_index)
    if not (0 <= idx < len(runs)):
        raise ValueError(f"--run-index {idx} out of range (0..{len(runs) - 1})")
    return runs[idx]


# ──────────────────────────────────────────────────────────────────────────
# Step 3: load ground truth
# ──────────────────────────────────────────────────────────────────────────

def load_ground_truth(path: Path) -> list[GTEvent]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = [
        GTEvent(
            event_id=int(item["event_id"]),
            begin=float(item["begin"]),
            end=float(item["end"]),
            gt_asr_text=item["gt_asr_text"],
            video_title=item.get("video_title", ""),
        )
        for item in data
    ]
    events.sort(key=lambda e: e.event_id)
    return events


# ──────────────────────────────────────────────────────────────────────────
# Step 4: temporal assignment (overlap) + aggregation into per-event Pred
# ──────────────────────────────────────────────────────────────────────────

def _overlap(seg: Segment, event: GTEvent) -> float:
    return max(0.0, min(seg.end, event.end) - max(seg.start, event.begin))


def assign_segments(
    segments: list[Segment], events: list[GTEvent]
) -> tuple[dict[int, list[Segment]], list[Segment]]:
    """Assign each segment to the GT event with which it overlaps most.

    Returns (event_id -> list[Segment], unassigned_segments).
    """
    assigned: dict[int, list[Segment]] = {e.event_id: [] for e in events}
    unassigned: list[Segment] = []

    for seg in segments:
        best_event_id = None
        best_overlap = 0.0
        for event in events:  # events are sorted ascending by event_id
            ov = _overlap(seg, event)
            if ov > best_overlap:  # strict '>' => ties keep the lower event_id
                best_overlap = ov
                best_event_id = event.event_id
        if best_event_id is None:
            unassigned.append(seg)
        else:
            assigned[best_event_id].append(seg)

    return assigned, unassigned


def build_predictions(assigned: dict[int, list[Segment]]) -> dict[int, str]:
    """Concatenate each event's assigned segments in temporal order."""
    preds: dict[int, str] = {}
    for event_id, segs in assigned.items():
        ordered = sorted(segs, key=lambda s: s.start)
        preds[event_id] = " ".join(s.text for s in ordered).strip()
    return preds


# ──────────────────────────────────────────────────────────────────────────
# Step 5: LLM-as-judge
# ──────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert sports video commentary evaluator.

Compare MODEL COMMENTARY against GROUND TRUTH for the same video segment.
Judge whether the commentary describes basketball ACTIONS and OUTCOMES that
correspond to the ground truth.

Scoring principles:
1. ACTION MATCHING (primary): Credit commentary that mentions key player actions
   present in the ground truth — e.g. dribble, steal, drive, shot attempt,
   score, miss, rebound, block, layup, dunk, out of bounds, change of possession.
2. TEAM-AGNOSTIC: Do NOT penalize if commentary attributes an action to the
   wrong side (our player vs opponent, home vs away, "he" without naming team).
   If the described ACTION matches a ground-truth action, treat it as aligned
   even when team identity differs.
3. OUTCOME MATTERS: Still penalize contradictory OUTCOMES — e.g. commentary
   says the shot scored but ground truth says airball/miss; or in-bounds vs
   out of bounds.
4. STYLE-AGNOSTIC: Do NOT penalize commentary style, tone, hype, or wording.

---MODEL COMMENTARY---
{pred}
---GROUND TRUTH---
{gt}
---

Score 1-5:
5 = key actions and outcomes in ground truth are reflected; no contradictions
4 = most key actions reflected; minor omissions; team labels may differ
3 = some key actions match; several missing; no major outcome contradiction
2 = few actions match, OR one major outcome contradiction (e.g. score vs miss)
1 = empty, unrelated, or outcomes mostly contradict ground truth

Reply with JSON only, no markdown code fences:
{{"score": <1-5>, "reason": "<one sentence>"}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class Judge:
    """OpenAI GPT judge — same model family as LiveCC llm_judge.py (GPT-4o)."""

    def __init__(self, model_name: str, temperature: float, delay_sec: float):
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment (.env)")

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._delay_sec = delay_sec

    def score(self, pred_text: str, gt_text: str) -> tuple[Optional[int], str, bool]:
        """Returns (score, reason, judge_error)."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(pred=pred_text, gt=gt_text)

        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    seed=42,
                )
                raw = (response.choices[0].message.content or "").strip()
                m = _JSON_RE.search(raw)
                if not m:
                    raise ValueError(f"no JSON object found in judge response: {raw!r}")
                obj = json.loads(m.group(0))
                score = int(obj["score"])
                if not (1 <= score <= 5):
                    raise ValueError(f"score out of range: {score}")
                reason = str(obj.get("reason", "")).strip()
                return score, reason, False
            except Exception as exc:  # noqa: BLE001 - want to retry on any failure
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                return None, f"judge_error: {exc}", True
            finally:
                if self._delay_sec > 0:
                    time.sleep(self._delay_sec)

        return None, "judge_error: exhausted retries", True


# ──────────────────────────────────────────────────────────────────────────
# Step 6: orchestration
# ──────────────────────────────────────────────────────────────────────────

def evaluate(
    events: list[GTEvent],
    predictions: dict[int, str],
    judge: Optional[Judge],
    dry_run: bool,
) -> list[EventResult]:
    results: list[EventResult] = []

    for event in events:
        pred_text = predictions.get(event.event_id, "")
        covered = bool(pred_text)

        if not covered:
            # Design note (3): uncovered events are scored programmatically.
            results.append(
                EventResult(
                    event_id=event.event_id,
                    begin=event.begin,
                    end=event.end,
                    gt_asr_text=event.gt_asr_text,
                    pred_text="",
                    covered=False,
                    num_segments=0,
                    score=1,
                    reason="No commentary overlapped this event window.",
                )
            )
            continue

        if dry_run or judge is None:
            results.append(
                EventResult(
                    event_id=event.event_id,
                    begin=event.begin,
                    end=event.end,
                    gt_asr_text=event.gt_asr_text,
                    pred_text=pred_text,
                    covered=True,
                    num_segments=pred_text.count(" ") + 1,
                    score=None,
                    reason="(dry-run: judge not called)",
                )
            )
            continue

        score, reason, judge_error = judge.score(pred_text, event.gt_asr_text)
        results.append(
            EventResult(
                event_id=event.event_id,
                begin=event.begin,
                end=event.end,
                gt_asr_text=event.gt_asr_text,
                pred_text=pred_text,
                covered=True,
                num_segments=pred_text.count(" ") + 1,
                score=score,
                reason=reason,
                judge_error=judge_error,
            )
        )
        print(
            f"  [event {event.event_id:2d}] score={score} covered=True  {reason[:80]}"
        )

    return results


def summarize(results: list[EventResult]) -> dict:
    scored = [r.score for r in results if r.score is not None]
    covered = [r for r in results if r.covered]
    errors = [r for r in results if r.judge_error]

    bc_align_score = (sum(scored) / len(scored)) if scored else None

    return {
        "bc_align_score": round(bc_align_score, 3) if bc_align_score is not None else None,
        "num_events": len(results),
        "num_scored": len(scored),
        "num_covered": len(covered),
        "coverage_rate": round(len(covered) / len(results), 3) if results else None,
        "num_judge_errors": len(errors),
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "BC-Align: evaluate whether the broadcast pipeline output "
            "(log/combination_output.log) is semantically aligned with "
            "eval/groundTruth.json, using GPT-4o as an LLM judge."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log", default=str(DEFAULT_LOG_PATH), help="combination_output.log path")
    p.add_argument("--groundtruth", default=str(DEFAULT_GT_PATH), help="groundTruth.json path")
    p.add_argument(
        "--run-index",
        default="auto",
        help="Which detected playback run to evaluate ('auto' = the run with the most lines, "
        "or an integer index; use --list-runs to inspect detected runs first)",
    )
    p.add_argument(
        "--reset-gap",
        type=float,
        default=DEFAULT_RESET_GAP_SEC,
        help="Seconds the video timeline must go backwards by to count as a new run",
    )
    p.add_argument(
        "--list-runs",
        action="store_true",
        help="Print detected runs and exit, without calling the judge",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run temporal assignment only, skip LLM judge calls (no API key required)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="OpenAI model used as judge")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--judge-delay", type=float, default=DEFAULT_JUDGE_DELAY_SEC)
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: eval/results/bc_align_<timestamp>.json)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log_path = Path(args.log)
    gt_path = Path(args.groundtruth)

    if not log_path.exists():
        print(f"[BC-Align] ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.exists():
        print(f"[BC-Align] ERROR: ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    all_segments = parse_log(log_path)
    if not all_segments:
        print(f"[BC-Align] ERROR: no valid lines parsed from {log_path}", file=sys.stderr)
        sys.exit(1)

    runs = split_into_runs(all_segments, reset_gap=args.reset_gap)
    print(f"[BC-Align] Detected {len(runs)} run(s) in {log_path.name}:")
    print(describe_runs(runs))

    if args.list_runs:
        return

    run_segments = select_run(runs, args.run_index)
    chosen_idx = runs.index(run_segments)
    print(f"\n[BC-Align] Using run #{chosen_idx} ({len(run_segments)} lines) for evaluation.\n")

    events = load_ground_truth(gt_path)
    print(f"[BC-Align] Loaded {len(events)} ground-truth events from {gt_path.name}")

    assigned, unassigned = assign_segments(run_segments, events)
    predictions = build_predictions(assigned)

    if unassigned:
        print(
            f"[BC-Align] {len(unassigned)} segment(s) fell outside all GT event "
            f"windows and were dropped (video time exceeds annotated range, or "
            f"precedes it):"
        )
        for seg in unassigned[:5]:
            print(f"    line {seg.line_no}: [{seg.start:.2f}-{seg.end:.2f}] {seg.text[:60]!r}")
        if len(unassigned) > 5:
            print(f"    ... and {len(unassigned) - 5} more")

    judge = None
    if not args.dry_run:
        judge = Judge(
            model_name=args.model,
            temperature=args.temperature,
            delay_sec=args.judge_delay,
        )
        print(f"\n[BC-Align] Judging {len(events)} events with model={args.model} ...")
    else:
        print("\n[BC-Align] --dry-run set: skipping LLM judge, showing assignment only.")

    results = evaluate(events, predictions, judge, dry_run=args.dry_run)
    summary = summarize(results)
    if args.dry_run:
        # Covered events were never sent to the judge, so the aggregate score
        # would only reflect the deterministic score=1 given to uncovered
        # events — report it as unavailable rather than a misleading 1.0.
        summary["bc_align_score"] = None

    print("\n" + "=" * 70)
    print("BC-Align RESULT")
    print("=" * 70)
    if not args.dry_run and summary["bc_align_score"] is not None:
        print(f"  BC-Align Score   : {summary['bc_align_score']:.3f}  (1-5 scale, higher is better)")
    else:
        print("  BC-Align Score   : N/A (dry-run: covered events were not judged)")
    print(f"  Coverage rate    : {summary['coverage_rate']}  ({summary['num_covered']}/{summary['num_events']} events received commentary)")
    print(f"  Judge errors     : {summary['num_judge_errors']}")
    print("=" * 70)

    out_path = Path(args.output) if args.output else (
        DEFAULT_RESULTS_DIR / f"bc_align_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "log_file": str(log_path),
        "groundtruth_file": str(gt_path),
        "run_index": chosen_idx,
        "run_span": {
            "start": run_segments[0].start,
            "end": run_segments[-1].end,
            "num_lines": len(run_segments),
        },
        "num_unassigned_segments": len(unassigned),
        "judge_model": args.model if not args.dry_run else None,
        "summary": summary,
        "events": [
            {
                "event_id": r.event_id,
                "begin": r.begin,
                "end": r.end,
                "covered": r.covered,
                "score": r.score,
                "reason": r.reason,
                "judge_error": r.judge_error,
                "pred_text": r.pred_text,
                "gt_asr_text": r.gt_asr_text,
            }
            for r in results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[BC-Align] Full results written to: {out_path}")


if __name__ == "__main__":
    main()
