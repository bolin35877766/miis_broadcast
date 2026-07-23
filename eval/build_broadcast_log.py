#!/usr/bin/env python3
"""Convert commentary-pipeline JSONL into a grounded, paced Eval log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from broadcast_policy import BroadcastPacer, ground_broadcast_text, result_cue


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def build_log(source: Path, output: Path, routine_interval_sec: float) -> dict[str, int]:
    records = [json.loads(line) for line in source.open(encoding="utf-8")]
    sources = {
        r["index"]: r
        for r in records
        if r.get("event") in {"livecc_segment", "result_banner_cue"}
    }
    tracker_cues = [
        r for r in records if r.get("event") == "result_banner_cue"
    ]
    pacer = BroadcastPacer(routine_interval_sec=routine_interval_sec)
    lines: list[str] = []
    total = 0
    grounded = 0

    for record in records:
        gemini = record.get("gemini") or {}
        if record.get("event") != "gemini_result" or not gemini.get("broadcast_text"):
            continue
        total += 1
        source_record = sources[record["index"]]
        raw = (
            (source_record.get("livecc") or {}).get("metadata", {}).get("raw")
            or source_record.get("livecc_text", "")
        )
        # The frame tracker is authoritative. If LiveCC happens to read the
        # same banner slightly earlier, drop that duplicate so a bare/partial
        # caption cannot suppress the tracker's Home/Away side a moment later.
        cue = result_cue(raw)
        if source_record.get("event") == "livecc_segment" and cue:
            if any(
                item.get("result_kind") == cue
                and abs(float(item["start"]) - float(record["start"])) <= 2.5
                for item in tracker_cues
            ):
                continue
        if not pacer.should_keep(float(record["start"]), raw):
            continue
        original = gemini["broadcast_text"].strip()
        text = ground_broadcast_text(original, raw)
        grounded += int(text != original)
        lines.append(
            f"[2000-01-01 00:00:00] [{_fmt(float(record['start']))}-{_fmt(float(record['end']))}] {text}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"input": total, "selected": len(lines), "grounded": grounded}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routine-interval", type=float, default=6.0)
    args = parser.parse_args()
    print(build_log(args.input, args.output, args.routine_interval))


if __name__ == "__main__":
    main()
