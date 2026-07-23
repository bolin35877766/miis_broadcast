#!/usr/bin/env python3
"""Apply independently classified actor labels to an existing pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miis_broadcast.core.models.broadcast_grounding import ground_broadcast_text
from miis_broadcast.core.models.gemini_broadcaster import _override_actor_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--actor-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = [json.loads(line) for line in args.base.open(encoding="utf-8")]
    actor_records = [json.loads(line) for line in args.actor_run.open(encoding="utf-8")]
    actor_by_index = {
        record["index"]: (record.get("gemini") or {}).get("actor_label", "unclear")
        for record in actor_records
        if record.get("event") == "gemini_result"
    }
    sources = {
        record["index"]: record
        for record in base
        if record.get("event") in {"livecc_segment", "result_banner_cue"}
    }

    output_records = [
        record for record in base
        if record.get("event") not in {"gemini_result", "run_finished"}
    ]
    changed = 0
    for record in base:
        if record.get("event") != "gemini_result" or not (record.get("gemini") or {}).get("broadcast_text"):
            continue
        updated = json.loads(json.dumps(record))
        actor = actor_by_index.get(record["index"], "unclear")
        original = updated["gemini"]["broadcast_text"]
        actor_text = _override_actor_subject(original, actor)
        source = sources[record["index"]]
        raw = (
            (source.get("livecc") or {}).get("metadata", {}).get("raw")
            or source.get("livecc_text", "")
        )
        updated["gemini"]["broadcast_text"] = ground_broadcast_text(actor_text, raw)
        updated["gemini"]["actor_label"] = actor
        updated["actor_label_applied"] = actor != "unclear"
        changed += updated["gemini"]["broadcast_text"] != original
        output_records.append(updated)

    output_records.append({
        "event": "run_finished",
        "source": str(args.base),
        "actor_labels": str(args.actor_run),
        "actor_overrides": changed,
        "output": str(args.output.resolve()),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output_records),
        encoding="utf-8",
    )
    print({"records": len(output_records), "actor_overrides": changed, "output": str(args.output)})


if __name__ == "__main__":
    main()
