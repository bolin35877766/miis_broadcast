#!/usr/bin/env python3
"""
eval/run_gveval_align.py — Temporal Alignment + G-VEval Scoring

A second, independent evaluation pipeline for the broadcast commentary
(log/combination_output.log) against eval/groundTruth.json. Stage 1 (time
alignment) is unchanged from eval/run_bc_align.py; Stage 2 (scoring) is
replaced with a faithful re-implementation of G-VEval (Tang et al., AAAI
2025, https://github.com/ztangaj/gveval) instead of BC-Align's custom 1-5
judge.

────────────────────────────────────────────────────────────────────────────
WHY A SEPARATE SCRIPT INSTEAD OF EDITING run_bc_align.py
────────────────────────────────────────────────────────────────────────────
BC-Align and this script share the exact same Stage 1 (parse log -> detect
runs -> pick one run -> overlap-assign segments to GT events -> concatenate
per-event Pred_e). That stage is imported directly from run_bc_align.py so
the two scripts can never silently drift apart on alignment logic. Only
Stage 2 differs:

                    Stage 1 (shared)            Stage 2 (differs)
run_bc_align.py     overlap alignment    ->     custom 1-5 rubric judge
run_gveval_align.py overlap alignment    ->     G-VEval ACCR judge (0-100)

Why overlap (not strict t-IoU) for Stage 1: verified empirically on the
project's actual log — argmax-overlap and argmax-t-IoU produce IDENTICAL
event assignments for all 33 broadcast segments recorded so far (segments
average ~4s vs. events averaging ~16s, so t-IoU values are naturally low
even for correct matches; a t-IoU *threshold* like the 0.5 common in
temporal-action-localization papers would incorrectly reject nearly every
correct assignment in this dataset). t-IoU is therefore not used to filter
matches here, only overlap-argmax, matching run_bc_align.py exactly.

────────────────────────────────────────────────────────────────────────────
WHAT IS FAITHFULLY REPRODUCED FROM THE G-VEVAL REFERENCE IMPLEMENTATION
────────────────────────────────────────────────────────────────────────────
(see evaluation/gveval/scorer.py and evaluation/gveval/prompts/vid/accr/
ref-only.txt in ztangaj/gveval)

1. ACCR rubric: every event is scored on 4 sub-dimensions — Accuracy,
   Completeness, Conciseness, Relevance — each 0-100, and the reported
   final_score is their unweighted mean. This is G-VEval's video/ACCR mode
   (accr=True), the mode the paper actually validated on MSVD-Eval.
2. Greek-letter score markers (alpha/beta/psi/delta) wrapping each
   sub-score in the model's own text, e.g. "...is <alpha>85<alpha>.".
3. Probability-weighted "expected value" scoring: rather than trusting the
   single greedy digit(s) the model happened to sample, we request
   logprobs=True, top_logprobs=5 on the completion, locate the token
   immediately AFTER each marker, and take the probability-weighted average
   of whichever 0-100 integer strings appear among its top-5 alternatives.
   This is G-Eval's/G-VEval's core trick for turning a discrete LLM sample
   into a continuous score and is reproduced in _expected_value_score()
   below, mirroring Scorer.extract_and_normalize_responses /
   normalize_responses in the original repo.
4. Sampling params for the scoring call (temperature=1, top_p=1,
   frequency_penalty=0, presence_penalty=0) — matches scorer.py exactly.
   NOTE: this is intentionally NOT temperature=0. The logprob-expected-value
   trick needs a non-degenerate probability distribution over the top-5
   tokens to be meaningful; temperature=0 (as BC-Align uses) would defeat
   the point of computing an expected value at all.
5. Text-only (no video frames) code path: our GT events have no associated
   video clip extracted per-event, so this mirrors the original repo's own
   fallback when no image is supplied (`messages = [{"role": "system",
   "content": prompt}]`, single system message, no image_url content).

────────────────────────────────────────────────────────────────────────────
DELIBERATE DEVIATIONS FROM THE REFERENCE IMPLEMENTATION
────────────────────────────────────────────────────────────────────────────
- Model: defaults to "gpt-4o" (an always-current alias) rather than hard-
  coding the paper's exact "gpt-4o-2024-05-13" snapshot, which may no
  longer be served by the API by the time this runs. Pass
  --model gpt-4o-2024-05-13 to reproduce the paper's exact snapshot if it
  is still available on your account.
- Reference formatting: the original repo's `"'; '".join(ref)` string-
  building leaves a dangling, unclosed quote for the single-reference case
  (a cosmetic bug, not a scoring difference). We just wrap gt_asr_text in a
  clean pair of quotes.
- Domain rubric hint: the Evaluation Criteria paragraph adds one sentence
  telling the judge not to penalize team/side attribution errors (our
  player vs. opponent) as long as the underlying action still matches —
  the same relaxation BC-Align uses, needed because this is 1-on-1 practice
  commentary where "team" labels are often ambiguous/interchangeable, unlike
  the third-person captions G-VEval was originally validated on.
- Fallback scoring: if logprobs are unavailable or no top-5 alternative for
  a marker is a plain 0-100 integer string (extraction failure), we fall
  back to regex-parsing the literal digits between each pair of markers in
  the message text (a deterministic point estimate, not an expected value).
  This never happens in the paper's own reported results but is needed for
  robustness against tokenizer/model differences the paper did not test.
- Uncovered events (no commentary overlapped the GT window at all) are not
  sent to the judge and are scored 0/100 deterministically, mirroring
  BC-Align's uncovered-event score=1/5 (both are "the floor of the scale").

Usage:
    python eval/run_gveval_align.py --list-runs
    python eval/run_gveval_align.py --dry-run
    python eval/run_gveval_align.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Stage 1 (temporal alignment) is intentionally reused, not re-implemented.
# Running this script directly (`python eval/run_gveval_align.py`) puts
# eval/ at the front of sys.path, so this plain import resolves to the
# sibling file eval/run_bc_align.py.
from run_bc_align import (  # noqa: E402
    DEFAULT_RESET_GAP_SEC,
    DEFAULT_RESULTS_DIR,
    GTEvent,
    Segment,
    assign_segments,
    build_predictions,
    describe_runs,
    load_ground_truth,
    parse_log,
    select_run,
    split_into_runs,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EVAL_DIR = Path(__file__).resolve().parent

DEFAULT_LOG_PATH = _PROJECT_ROOT / "log" / "combination_output.log"
DEFAULT_GT_PATH = _EVAL_DIR / "groundTruth.json"

DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_TEMPERATURE = 1.0  # required by the logprob expected-value trick
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_LOGPROBS = 5
DEFAULT_JUDGE_DELAY_SEC = 0.3


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────

ACCR_DIMENSIONS = ("accuracy", "completeness", "conciseness", "relevance")

# Greek-letter markers, exactly as in ztangaj/gveval's
# evaluation/gveval/prompts/vid/accr/ref-only.txt
_MARKERS = {
    "accuracy": "\u03b1",      # alpha
    "completeness": "\u03b2",  # beta
    "conciseness": "\u03c8",   # psi
    "relevance": "\u03b4",     # delta
}


@dataclass
class GVEvalScore:
    accuracy: Optional[float]
    completeness: Optional[float]
    conciseness: Optional[float]
    relevance: Optional[float]
    final_score: Optional[float]
    reason: str
    judge_error: bool
    extraction_mode: str  # "expected_value" | "regex_fallback" | "n/a"


@dataclass
class EventResult:
    event_id: int
    begin: float
    end: float
    gt_asr_text: str
    pred_text: str
    covered: bool
    num_segments: int
    score: GVEvalScore


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: G-VEval judge (ACCR, reference-only, no video frames)
# ──────────────────────────────────────────────────────────────────────────

GVEVAL_PROMPT_TEMPLATE = """You will be given a piece of commentary generated for a segment of a 1-on-1 basketball broadcast video.

Your task is to rate the generated commentary based on how well it captures the essential basketball actions and outcomes described in the reference commentary for the same video segment.

Evaluation Criteria:

Score is from 0 to 100 - The generated commentary should accurately reflect the actions and outcomes described in the reference commentary, and appropriately describe the key events without hallucinating actions the reference does not support. Annotators should penalize commentary that includes irrelevant details, omits significant elements indicated in the reference, or contradicts the reference's outcome (e.g. reference says the shot missed but commentary says it scored). Do NOT penalize commentary purely for attributing an action to the wrong side (our player vs. the opponent) if the underlying basketball action still corresponds to something in the reference.

Evaluation Dimensions:

Accuracy: Does the commentary correctly describe the actions and outcomes in the reference, without errors, hallucinations, or contradicted outcomes? (Team/side attribution alone should not lower this score.)
Completeness: Does the commentary cover all significant actions and events in the reference (e.g. dribble, steal, shot attempt, score, miss, rebound, block, out of bounds), rather than only a subset?
Conciseness: Is the commentary clear and succinct, avoiding unnecessary details and repetition?
Relevance: Is the commentary pertinent to this basketball segment, without including irrelevant information?

Evaluation Steps:
1. Read the reference commentary carefully; it describes the ground-truth actions and outcomes for this video segment.
2. Read the generated commentary that needs to be evaluated.
3. Compare the generated commentary with the reference commentary and assess how well it captures the actions and outcomes.
4. Evaluate how accurately and completely the generated commentary describes the events shown, and check for contradicted outcomes, irrelevant details, or omissions.
5. Assign an integer score from 0 to 100 for each of the four dimensions below.

Reference Commentary:
{ref}

Generated Commentary:
{pred}

You should first give a detailed reason for your scores, and end with one sentence per score, exactly in this form:
..... The Accuracy score is \u03b1{{accuracy_score}}\u03b1.
..... The Completeness score is \u03b2{{completeness_score}}\u03b2.
..... The Conciseness score is \u03c8{{conciseness_score}}\u03c8.
..... The Relevance score is \u03b4{{relevance_score}}\u03b4.

Note that each score must be an integer from 0 to 100, wrapped in the corresponding Greek letter with nothing else between the letter and the digits:
Wrap Accuracy score in \u03b1
Wrap Completeness score in \u03b2
Wrap Conciseness score in \u03c8
Wrap Relevance score in \u03b4"""


def _format_reference(gt_text: str) -> str:
    return f"'{gt_text}'"


def _expected_value_score(all_tokens: list, marker: str) -> Optional[float]:
    """Probability-weighted expected value over the top-5 alternatives of
    the token right after `marker`'s first occurrence in the response.

    Mirrors Scorer.extract_and_normalize_responses + normalize_responses
    in ztangaj/gveval's evaluation/gveval/scorer.py: only candidates whose
    token text is a bare 0-100 integer string contribute; everything else
    is treated as zero probability, and the remaining mass is renormalized
    to sum to 1 before taking the weighted average.
    """
    marker_index = None
    for i, tok in enumerate(all_tokens):
        if marker in tok.token:
            marker_index = i
            break
    if marker_index is None or marker_index + 1 >= len(all_tokens):
        return None

    candidates = all_tokens[marker_index + 1].top_logprobs
    weights: dict[str, float] = {}
    for c in candidates:
        key = c.token.strip()
        weights[key] = weights.get(key, 0.0) + exp(c.logprob)

    expected, matched_mass = 0.0, 0.0
    for tok_str, w in weights.items():
        if tok_str.isdigit() and 0 <= int(tok_str) <= 100:
            expected += int(tok_str) * w
            matched_mass += w
    if matched_mass <= 0:
        return None
    return expected / matched_mass


def _regex_fallback_score(message_text: str, marker: str) -> Optional[float]:
    """Deterministic fallback: read the literal digits the model wrote
    between a pair of markers, e.g. '...is \u03b185\u03b1.' -> 85.0. Used only
    when logprobs are unavailable or extraction otherwise fails.
    """
    m = re.search(re.escape(marker) + r"\s*(\d{1,3})\s*" + re.escape(marker), message_text)
    if not m:
        return None
    val = int(m.group(1))
    if not (0 <= val <= 100):
        return None
    return float(val)


class GVEvalJudge:
    """Faithful re-implementation of ztangaj/gveval's Scorer.gveval (video,
    ACCR, reference-only, no image/video attachment).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        top_p: float,
        top_logprobs: int,
        delay_sec: float,
    ):
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment (.env)")

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._top_logprobs = top_logprobs
        self._delay_sec = delay_sec

    def score(self, pred_text: str, gt_text: str) -> GVEvalScore:
        prompt = GVEVAL_PROMPT_TEMPLATE.format(
            ref=_format_reference(gt_text), pred=pred_text
        )

        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self._temperature,
                    top_p=self._top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                    logprobs=True,
                    top_logprobs=self._top_logprobs,
                )
                choice = response.choices[0]
                message_text = (choice.message.content or "").strip()
                all_tokens = (choice.logprobs.content if choice.logprobs else None) or []

                dims: dict[str, Optional[float]] = {}
                mode = "expected_value" if all_tokens else "regex_fallback"
                for dim, marker in _MARKERS.items():
                    val = _expected_value_score(all_tokens, marker) if all_tokens else None
                    if val is None:
                        val = _regex_fallback_score(message_text, marker)
                        if val is not None:
                            mode = "regex_fallback"
                    dims[dim] = val

                if any(v is None for v in dims.values()):
                    missing = [d for d, v in dims.items() if v is None]
                    raise ValueError(f"could not extract score(s) for: {missing}")

                final_score = sum(dims.values()) / len(dims)
                return GVEvalScore(
                    accuracy=dims["accuracy"],
                    completeness=dims["completeness"],
                    conciseness=dims["conciseness"],
                    relevance=dims["relevance"],
                    final_score=round(final_score, 2),
                    reason=message_text,
                    judge_error=False,
                    extraction_mode=mode,
                )
            except Exception as exc:  # noqa: BLE001 - want to retry on any failure
                last_exc = exc
                if attempt == 0:
                    time.sleep(1.0)
                    continue
            finally:
                if self._delay_sec > 0:
                    time.sleep(self._delay_sec)

        return GVEvalScore(
            accuracy=None,
            completeness=None,
            conciseness=None,
            relevance=None,
            final_score=None,
            reason=f"judge_error: {last_exc}",
            judge_error=True,
            extraction_mode="n/a",
        )


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────

def evaluate(
    events: list[GTEvent],
    predictions: dict[int, str],
    judge: Optional[GVEvalJudge],
    dry_run: bool,
    segment_counts: Optional[dict[int, int]] = None,
) -> list[EventResult]:
    results: list[EventResult] = []

    for event in events:
        pred_text = predictions.get(event.event_id, "")
        covered = bool(pred_text)

        if not covered:
            results.append(
                EventResult(
                    event_id=event.event_id,
                    begin=event.begin,
                    end=event.end,
                    gt_asr_text=event.gt_asr_text,
                    pred_text="",
                    covered=False,
                    num_segments=0,
                    score=GVEvalScore(
                        accuracy=0.0,
                        completeness=0.0,
                        conciseness=0.0,
                        relevance=0.0,
                        final_score=0.0,
                        reason="No commentary overlapped this event window.",
                        judge_error=False,
                        extraction_mode="n/a",
                    ),
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
                    num_segments=(segment_counts or {}).get(event.event_id, 0),
                    score=GVEvalScore(
                        accuracy=None,
                        completeness=None,
                        conciseness=None,
                        relevance=None,
                        final_score=None,
                        reason="(dry-run: judge not called)",
                        judge_error=False,
                        extraction_mode="n/a",
                    ),
                )
            )
            continue

        score = judge.score(pred_text, event.gt_asr_text)
        results.append(
            EventResult(
                event_id=event.event_id,
                begin=event.begin,
                end=event.end,
                gt_asr_text=event.gt_asr_text,
                pred_text=pred_text,
                covered=True,
                num_segments=(segment_counts or {}).get(event.event_id, 0),
                score=score,
            )
        )
        print(
            f"  [event {event.event_id:2d}] final={score.final_score} "
            f"(A={score.accuracy} C={score.completeness} "
            f"Cn={score.conciseness} R={score.relevance}) "
            f"mode={score.extraction_mode}"
        )

    return results


def summarize(results: list[EventResult]) -> dict:
    scored = [r.score for r in results if r.score.final_score is not None]
    covered = [r for r in results if r.covered]
    errors = [r for r in results if r.score.judge_error]

    def _avg(attr: str) -> Optional[float]:
        vals = [getattr(s, attr) for s in scored if getattr(s, attr) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    gveval_score = _avg("final_score")

    return {
        "gveval_align_score": gveval_score,
        "avg_accuracy": _avg("accuracy"),
        "avg_completeness": _avg("completeness"),
        "avg_conciseness": _avg("conciseness"),
        "avg_relevance": _avg("relevance"),
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
            "Temporal-alignment + G-VEval: evaluate whether the broadcast "
            "pipeline output (log/combination_output.log) matches "
            "eval/groundTruth.json, using a faithful re-implementation of "
            "G-VEval (Tang et al., AAAI 2025) as the per-event judge."
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
        help="Run temporal assignment only, skip G-VEval calls (no API key required)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="OpenAI model used as judge")
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature. G-VEval's expected-value trick assumes temperature=1",
    )
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--top-logprobs", type=int, default=DEFAULT_TOP_LOGPROBS)
    p.add_argument("--judge-delay", type=float, default=DEFAULT_JUDGE_DELAY_SEC)
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: eval/results/gveval_align_<timestamp>.json)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log_path = Path(args.log)
    gt_path = Path(args.groundtruth)

    if not log_path.exists():
        print(f"[GVEval-Align] ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.exists():
        print(f"[GVEval-Align] ERROR: ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    all_segments = parse_log(log_path)
    if not all_segments:
        print(f"[GVEval-Align] ERROR: no valid lines parsed from {log_path}", file=sys.stderr)
        sys.exit(1)

    runs = split_into_runs(all_segments, reset_gap=args.reset_gap)
    print(f"[GVEval-Align] Detected {len(runs)} run(s) in {log_path.name}:")
    print(describe_runs(runs))

    if args.list_runs:
        return

    run_segments = select_run(runs, args.run_index)
    chosen_idx = runs.index(run_segments)
    print(f"\n[GVEval-Align] Using run #{chosen_idx} ({len(run_segments)} lines) for evaluation.\n")

    events = load_ground_truth(gt_path)
    print(f"[GVEval-Align] Loaded {len(events)} ground-truth events from {gt_path.name}")

    assigned, unassigned = assign_segments(run_segments, events)
    predictions = build_predictions(assigned)

    if unassigned:
        print(
            f"[GVEval-Align] {len(unassigned)} segment(s) fell outside all GT event "
            f"windows and were dropped (video time exceeds annotated range, or "
            f"precedes it):"
        )
        for seg in unassigned[:5]:
            print(f"    line {seg.line_no}: [{seg.start:.2f}-{seg.end:.2f}] {seg.text[:60]!r}")
        if len(unassigned) > 5:
            print(f"    ... and {len(unassigned) - 5} more")

    judge = None
    if not args.dry_run:
        judge = GVEvalJudge(
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_logprobs=args.top_logprobs,
            delay_sec=args.judge_delay,
        )
        print(f"\n[GVEval-Align] Judging {len(events)} events with model={args.model} ...")
    else:
        print("\n[GVEval-Align] --dry-run set: skipping G-VEval judge, showing assignment only.")

    segment_counts = {event_id: len(segs) for event_id, segs in assigned.items()}
    results = evaluate(
        events,
        predictions,
        judge,
        dry_run=args.dry_run,
        segment_counts=segment_counts,
    )
    summary = summarize(results)
    if args.dry_run:
        summary["gveval_align_score"] = None

    print("\n" + "=" * 70)
    print("GVEval-Align RESULT")
    print("=" * 70)
    if not args.dry_run and summary["gveval_align_score"] is not None:
        print(f"  GVEval-Align Score : {summary['gveval_align_score']:.2f}  (0-100 scale, higher is better)")
        print(
            f"  ACCR breakdown     : Acc={summary['avg_accuracy']}  "
            f"Comp={summary['avg_completeness']}  "
            f"Conc={summary['avg_conciseness']}  "
            f"Rel={summary['avg_relevance']}"
        )
    else:
        print("  GVEval-Align Score : N/A (dry-run: covered events were not judged)")
    print(f"  Coverage rate      : {summary['coverage_rate']}  ({summary['num_covered']}/{summary['num_events']} events received commentary)")
    print(f"  Judge errors       : {summary['num_judge_errors']}")
    print("=" * 70)

    out_path = Path(args.output) if args.output else (
        DEFAULT_RESULTS_DIR / f"gveval_align_{time.strftime('%Y%m%d_%H%M%S')}.json"
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
                "num_segments": r.num_segments,
                "final_score": r.score.final_score,
                "accuracy": r.score.accuracy,
                "completeness": r.score.completeness,
                "conciseness": r.score.conciseness,
                "relevance": r.score.relevance,
                "extraction_mode": r.score.extraction_mode,
                "reason": r.score.reason,
                "judge_error": r.score.judge_error,
                "pred_text": r.pred_text,
                "gt_asr_text": r.gt_asr_text,
            }
            for r in results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[GVEval-Align] Full results written to: {out_path}")


if __name__ == "__main__":
    main()
