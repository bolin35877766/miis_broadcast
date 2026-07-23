#!/usr/bin/env python3
"""Rerun Gemini on an existing pipeline JSONL with actor-grounding video frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _right_jpegs(
    cap: cv2.VideoCapture, start: float, end: float
) -> list[bytes]:
    midpoint = (start + end) / 2.0
    frames: list[bytes] = []
    for timestamp in (midpoint - 0.25, midpoint, midpoint + 0.25):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        right = frame[:, frame.shape[1] // 2 :]
        ok, encoded = cv2.imencode(".jpg", right, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            frames.append(encoded.tobytes())
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))
    records = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    run_started = next(r for r in records if r.get("event") == "run_started")
    sources = [
        r for r in records if r.get("event") in {"livecc_segment", "result_banner_cue"}
        and float(r.get("end", 0.0)) >= args.start
        and (not args.end or float(r.get("start", 0.0)) <= args.end)
    ]

    from miis_broadcast.core.models.gemini_broadcaster import (
        set_language,
        set_style,
        stream_gemini,
    )

    set_style(run_started.get("style", "objective"))
    set_language(run_started.get("language", "en"))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    errors = 0
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(run_started, ensure_ascii=False) + "\n")
        for source in sources:
            handle.write(json.dumps(source, ensure_ascii=False) + "\n")
        for source in sorted(sources, key=lambda item: (item["start"], item["end"], item["index"])):
            event = json.loads(json.dumps(source["livecc"]))
            if source["event"] == "livecc_segment":
                event.setdefault("metadata", {})["actor_frames_jpeg"] = _right_jpegs(
                    cap, float(source["start"]), float(source["end"])
                )
            started = time.perf_counter()
            result = None
            error = None
            try:
                for streamed in stream_gemini(event):
                    result = streamed.to_dict()
                if not result or not result.get("broadcast_text"):
                    raise RuntimeError("Gemini returned no broadcast_text")
                success += 1
            except Exception as exc:
                errors += 1
                error = {"stage": "gemini", "message": str(exc)}
            record = {
                "event": "gemini_result",
                "index": source["index"],
                "start": source["start"],
                "end": source["end"],
                "gemini": result,
                "gemini_latency_s": round(time.perf_counter() - started, 4),
                "error": error,
                "actor_frames": 3 if source["event"] == "livecc_segment" else 0,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{float(source['start']):7.2f}-{float(source['end']):7.2f}] "
                f"{(result or {}).get('broadcast_text', error)}",
                flush=True,
            )
            if args.delay:
                time.sleep(args.delay)
        summary = {
            "event": "run_finished",
            "source_segments": len(sources),
            "gemini_success": success,
            "gemini_errors": errors,
            "actor_frame_grounding": True,
            "output": str(args.output.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    cap.release()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
