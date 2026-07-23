#!/usr/bin/env python3
"""Probe a frame-based player/robot-opponent possession classifier."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types


SYSTEM_PROMPT = """Classify visible basketball possession from the supplied consecutive images.
Each image is ONLY the player's first-person game view. The foreground black/yellow hands
belong to the player. The human-shaped avatar, often labeled test_bot1 and often having
green hair, is the robot opponent.

Return player only when the ball is visibly held, touched, or released by the foreground
first-person hands. Return robot_opponent only when the ball is visibly held, touched, or
released by the avatar. Camera direction, proximity, score, result text, and body pose are
not possession evidence. If the ball is absent, occluded, between actors, or ownership is
not visually clear, return unclear. Output exactly one label: player, robot_opponent, or
unclear."""


def _jpeg_right_frame(cap: cv2.VideoCapture, timestamp: float) -> bytes:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at {timestamp:.2f}s")
    right = frame[:, frame.shape[1] // 2 :]
    ok, encoded = cv2.imencode(".jpg", right, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"Could not encode frame at {timestamp:.2f}s")
    return encoded.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=10,
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    records = []
    try:
        for timestamp in args.times:
            contents: list[object] = ["The images are ordered from earlier to later."]
            for offset in (-0.25, 0.0, 0.25):
                contents.append(types.Part.from_bytes(
                    data=_jpeg_right_frame(cap, timestamp + offset),
                    mime_type="image/jpeg",
                ))
            response = client.models.generate_content(
                model=args.model, contents=contents, config=config
            )
            label = (response.text or "").strip().lower()
            record = {"time": timestamp, "label": label}
            records.append(record)
            print(json.dumps(record), flush=True)
            if args.delay:
                time.sleep(args.delay)
    finally:
        cap.release()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
