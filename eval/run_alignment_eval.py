"""
Audio-Visual Alignment Evaluation
===================================
Evaluates how well the broadcast system's output aligns with ground-truth
event annotations in a collage test video.

Metrics
-------
- R@1, IoU=0.5   : Temporal hit rate (best matching segment has t-IoU >= 0.5)
- Semantic Hit Rate : Matched segment contains at least one outcome keyword
                      (e.g. 得分/沒進 — not action words like 投籃/出手)
- Joint Hit Rate    : Both temporal AND semantic conditions satisfied (main score)

Usage
-----
    python eval/run_alignment_eval.py \
        --gt  eval/collage_gt_example.yaml \
        --log log/combination_output.log

Options
-------
    --gt   PATH       Path to ground-truth YAML file (required)
    --log  PATH       Path to the broadcast log to evaluate
                      (default: log/combination_output.log)
    --iou  FLOAT      t-IoU threshold for temporal hit (default: 0.5)
    --verbose         Print per-event detail

Log format expected (combination_output.log)
--------------------------------------------
    [YYYY-MM-DD HH:MM:SS] [MM:SS.ss-MM:SS.ss] broadcast text here

    Timestamps in brackets are video playback time when TTS audio starts
    (written at playback start, not when Gemini produces text).

GT YAML format
--------------
    See eval/collage_gt_example.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
except ImportError:
    print("[Error] PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class GtEvent:
    id: str
    label: str
    t_start: float
    t_end: float
    keywords: List[str]
    expect_no_tts: bool = False


@dataclass
class EventResult:
    event: GtEvent
    best_seg: Optional[Segment]
    best_iou: float
    temporal_hit: bool
    semantic_hit: bool
    joint_hit: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches: [MM:SS.ss-MM:SS.ss] text
# Also tolerates optional priority/source prefix inside the brackets
_SEG_RE = re.compile(
    r'\[(\d{1,2}:\d{2}\.\d+)-(\d{1,2}:\d{2}\.\d+)\]\s*(.*)'
)


def _parse_time(s: str) -> float:
    """Convert 'MM:SS.ss' string to seconds as float."""
    parts = s.split(":", 1)
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60.0 + seconds


def parse_log(log_path: Path) -> List[Segment]:
    """Parse combination_output.log (or any log with [start-end] format)."""
    segments: List[Segment] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = _SEG_RE.search(line)
            if m:
                start = _parse_time(m.group(1))
                end = _parse_time(m.group(2))
                text = m.group(3).strip()
                if end < start:
                    end = start
                segments.append(Segment(start=start, end=end, text=text))
    return segments


def load_gt(gt_path: Path) -> List[GtEvent]:
    """Load ground-truth event list from YAML."""
    with open(gt_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    events = []
    for ev in data.get("events", []):
        events.append(GtEvent(
            id=str(ev["id"]),
            label=str(ev["label"]),
            t_start=float(ev["t_start"]),
            t_end=float(ev["t_end"]),
            keywords=[str(k) for k in ev.get("keywords", [])],
            expect_no_tts=bool(ev.get("expect_no_tts", False)),
        ))
    return events


def calculate_tiou(pred: Segment, gt: GtEvent) -> float:
    """Temporal IoU between a predicted segment and a GT event."""
    inter = max(0.0, min(pred.end, gt.t_end) - max(pred.start, gt.t_start))
    union = (pred.end - pred.start) + (gt.t_end - gt.t_start) - inter
    return inter / union if union > 0 else 0.0


def check_semantic(seg: Segment, gt: GtEvent) -> bool:
    """True if any keyword from GT appears in the segment text (case-insensitive)."""
    if not gt.keywords:
        return True  # no keywords required → always pass
    text_lower = seg.text.lower()
    return any(kw.lower() in text_lower for kw in gt.keywords)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    segments: List[Segment],
    events: List[GtEvent],
    iou_threshold: float = 0.5,
) -> List[EventResult]:
    results: List[EventResult] = []
    for ev in events:
        if ev.expect_no_tts:
            # For silent events, check whether any segment overlaps at all
            overlapping = [s for s in segments if calculate_tiou(s, ev) > 0]
            false_alarm = len(overlapping) > 0
            results.append(EventResult(
                event=ev,
                best_seg=overlapping[0] if overlapping else None,
                best_iou=calculate_tiou(overlapping[0], ev) if overlapping else 0.0,
                temporal_hit=not false_alarm,   # hit = correctly silent
                semantic_hit=not false_alarm,
                joint_hit=not false_alarm,
            ))
            continue

        # Find the segment with the highest t-IoU for this event
        best_seg: Optional[Segment] = None
        best_iou = 0.0
        for seg in segments:
            iou = calculate_tiou(seg, ev)
            if iou > best_iou:
                best_iou = iou
                best_seg = seg

        temporal_hit = best_iou >= iou_threshold
        semantic_hit = check_semantic(best_seg, ev) if best_seg else False
        joint_hit = temporal_hit and semantic_hit

        results.append(EventResult(
            event=ev,
            best_seg=best_seg,
            best_iou=best_iou,
            temporal_hit=temporal_hit,
            semantic_hit=semantic_hit,
            joint_hit=joint_hit,
        ))
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    if d == 0:
        return "N/A"
    return f"{n / d * 100:.1f}%  ({n}/{d})"


def print_report(
    results: List[EventResult],
    log_path: Path,
    gt_path: Path,
    iou_threshold: float,
    verbose: bool,
) -> None:
    # Separate silent events from normal events
    normal = [r for r in results if not r.event.expect_no_tts]
    silent = [r for r in results if r.event.expect_no_tts]

    n = len(normal)
    temporal_hits = sum(1 for r in normal if r.temporal_hit)
    semantic_hits = sum(1 for r in normal if r.semantic_hit)
    joint_hits = sum(1 for r in normal if r.joint_hit)

    print()
    print("=" * 60)
    print("  Audio-Visual Alignment Evaluation")
    print("=" * 60)
    print(f"  GT file : {gt_path}")
    print(f"  Log file: {log_path}")
    print(f"  Events  : {n} normal  +  {len(silent)} silent")
    print(f"  IoU threshold: {iou_threshold}")
    print("-" * 60)
    print(f"  R@1, IoU={iou_threshold:<3}  (temporal) : {_pct(temporal_hits, n)}")
    print(f"  Semantic Hit Rate          : {_pct(semantic_hits, n)}")
    print(f"  Joint Hit Rate  ← MAIN     : {_pct(joint_hits, n)}")

    if silent:
        silent_ok = sum(1 for r in silent if r.joint_hit)
        print(f"  False Speak Rate (silent)  : {_pct(len(silent) - silent_ok, len(silent))}")

    if verbose:
        print("-" * 60)
        print("  Per-event detail:")
        for r in results:
            ev = r.event
            if ev.expect_no_tts:
                tag = "✓ no-TTS" if r.joint_hit else "✗ SPOKE!"
                pred_str = f"pred=[{r.best_seg.start:.2f}-{r.best_seg.end:.2f}]" if r.best_seg else "pred=none"
                print(f"    {ev.id:<12} {ev.label:<20} {tag}  {pred_str}")
            else:
                t_mark = "✓" if r.temporal_hit else "✗"
                s_mark = "✓" if r.semantic_hit else "✗"
                j_mark = "✓" if r.joint_hit else "✗"
                iou_str = f"IoU={r.best_iou:.2f}"
                if r.best_seg:
                    pred_str = f"[{r.best_seg.start:.2f}-{r.best_seg.end:.2f}] \"{r.best_seg.text[:40]}\""
                else:
                    pred_str = "(no segment found)"
                print(f"    {ev.id:<12} {ev.label:<20} T={t_mark} S={s_mark} J={j_mark}  {iou_str}  {pred_str}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio-visual alignment evaluation for miis_broadcast"
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to ground-truth YAML file (e.g. eval/collage_gt_example.yaml)",
    )
    parser.add_argument(
        "--log",
        default=str(Path(__file__).parent.parent / "log" / "combination_output.log"),
        help="Path to broadcast log file (default: log/combination_output.log)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="t-IoU threshold for temporal hit (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-event detail",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt)
    log_path = Path(args.log)

    if not gt_path.exists():
        print(f"[Error] GT file not found: {gt_path}")
        sys.exit(1)
    if not log_path.exists():
        print(f"[Error] Log file not found: {log_path}")
        print("  Make sure you have run the system and the log has been generated.")
        sys.exit(1)

    events = load_gt(gt_path)
    segments = parse_log(log_path)
    print(f"[Info] Loaded {len(events)} GT events, parsed {len(segments)} log segments.")

    results = evaluate(segments, events, iou_threshold=args.iou)
    print_report(results, log_path, gt_path, args.iou, verbose=args.verbose)


if __name__ == "__main__":
    main()
