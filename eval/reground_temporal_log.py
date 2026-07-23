#!/usr/bin/env python3
"""Reapply deterministic outcome grounding to an existing temporal Eval log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from broadcast_policy import ground_broadcast_text, result_cue
from video_result_cues import detect_result_cues


LINE_RE = re.compile(
    r"^(?P<prefix>.*?\[(?P<sm>\d+):(?P<ss>[\d.]+)-(?P<em>\d+):(?P<es>[\d.]+)\]\s+)(?P<text>.*)$"
)


def seconds(minutes: str, value: str) -> float:
    return int(minutes) * 60 + float(value)


def fmt(value: float) -> str:
    return f"{int(value // 60):02d}:{value % 60:05.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--separate-cues", action="store_true")
    parser.add_argument("--cue-shift-sec", type=float, default=3.0)
    parser.add_argument(
        "--video-cues",
        type=Path,
        help="Detect result banners from this video instead of caption text",
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input_jsonl.open(encoding="utf-8")]
    captions = []
    for record in records:
        if record.get("event") != "livecc_segment":
            continue
        raw = (
            (record.get("livecc") or {}).get("metadata", {}).get("raw")
            or record.get("livecc_text", "")
        ).strip()
        if raw:
            captions.append((float(record["start"]), float(record["end"]), raw))

    output_records: list[tuple[float, str]] = []
    changed = 0
    for line in args.input_log.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(line)
        if not match:
            raise ValueError(f"Unrecognized log line: {line}")
        start = seconds(match["sm"], match["ss"])
        end = seconds(match["em"], match["es"])
        # Match run_temporal_gemini.py exactly: captions belong to the bucket
        # containing their start time, so one delayed result cannot leak into
        # two consecutive broadcasts.
        raw = "\n".join(text for a, _b, text in captions if start <= a < end)
        # In hybrid mode the routine summary must not repeat an outcome; each
        # explicit result is emitted below at its own latency-adjusted time.
        grounded = ground_broadcast_text(match["text"], "" if args.separate_cues else raw)
        changed += grounded != match["text"]
        output_records.append((start, match["prefix"] + grounded))

    if args.separate_cues:
        last_cue: dict[str, float] = {}
        if args.video_cues:
            cue_records = [
                (cue.start, cue.end, cue.kind, "")
                for cue in detect_result_cues(args.video_cues)
            ]
        else:
            cue_records = [
                (start, end, cue, raw)
                for start, end, raw in captions
                if (cue := result_cue(raw))
            ]
        for start, end, cue, raw in cue_records:
            shifted_start = max(
                0.0, start - (0.0 if args.video_cues else args.cue_shift_sec)
            )
            if shifted_start - last_cue.get(cue, -1e9) < 3.0:
                continue
            # Treat a result banner as a point event.  Keeping the original
            # two-second caption span can straddle a possession boundary and
            # assign the outcome to the following play.
            shifted_end = shifted_start + 0.25
            text = (
                "The player attacks the basket and scores."
                if cue == "score"
                else "The ballhandler attacks under pressure before the ball goes out of bounds."
            )
            line = (
                f"[2000-01-01 00:00:00] "
                f"[{fmt(shifted_start)}-{fmt(shifted_end)}] {text}"
            )
            output_records.append((shifted_start, line))
            last_cue[cue] = shifted_start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_records.sort(key=lambda item: item[0])
    output_lines = [line for _start, line in output_records]
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print({"lines": len(output_lines), "changed": changed, "output": str(args.output)})


if __name__ == "__main__":
    main()
