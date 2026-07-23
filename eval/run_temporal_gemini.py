#!/usr/bin/env python3
"""Aggregate noisy LiveCC captions into one evidence-grounded broadcast per window."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

from broadcast_policy import ground_broadcast_text


SYSTEM_PROMPT = """You produce one concise English basketball play-by-play sentence.
The input contains consecutive captions from one short observation window. Captions
may be fragmented, repetitive, stale, or hallucinated. Reconcile them; do not list
them. An exact on-screen result phrase such as 'Scored!' or 'Out of Bounds!' is the
strongest outcome evidence. A numeric score, body gesture, celebration, hoop close-up,
or a caption merely claiming success is not proof. If evidence conflicts or no result
phrase is present, describe only the visible attempt, drive, dribble, defense, reset,
or waiting without claiming a make or miss. There are no teammates or passes. Never
mention VR, the screen, interface, names, score, clock, or uncertainty. Output exactly
one natural present-tense sentence of at most 14 words, with no label or explanation."""


def _raw(record: dict) -> str:
    return (
        (record.get("livecc") or {}).get("metadata", {}).get("raw")
        or record.get("livecc_text", "")
    ).strip()


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required")

    records = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    captions = [r for r in records if r.get("event") == "livecc_segment" and _raw(r)]
    buckets: dict[int, list[dict]] = {}
    for record in captions:
        buckets.setdefault(int(float(record["start"]) // args.window_sec), []).append(record)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=40,
    )
    lines: list[str] = []
    for bucket, group in sorted(buckets.items()):
        start = bucket * args.window_sec
        end = (bucket + 1) * args.window_sec
        prompt = "\n".join(
            f"[{float(r['start']):.2f}-{float(r['end']):.2f}] {_raw(r)}" for r in group
        )
        response = client.models.generate_content(model=args.model, contents=prompt, config=config)
        text = " ".join((response.text or "").strip().splitlines()).strip()
        if not text:
            text = "The players reset for the next possession."
        text = ground_broadcast_text(text, "\n".join(_raw(r) for r in group))
        lines.append(
            f"[2000-01-01 00:00:00] [{_fmt(start)}-{_fmt(end)}] {text}"
        )
        print(f"[{_fmt(start)}-{_fmt(end)}] {text}", flush=True)
        if args.delay:
            time.sleep(args.delay)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"windows": len(lines), "output": str(args.output)}))


if __name__ == "__main__":
    main()
