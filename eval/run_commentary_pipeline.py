#!/usr/bin/env python3
"""Run a video through LiveCC -> Gemini without starting the desktop GUI.

This is an evaluation harness, not an alternative application frontend.  It uses
the same model configuration, sport prompts, Gemini styles and parsing functions
as the GUI, but deliberately skips TTS, cameras, LiveKit and Qt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _json_default(value: Any) -> str:
    return str(value)


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    handle.flush()


def _attach_recent_action(
    banner: dict[str, Any], sources: list[dict[str, Any]], *, max_gap: float = 8.0
) -> dict[str, Any]:
    """Give a result cue the latest preceding visible action for richer wording."""
    from miis_broadcast.core.models.broadcast_grounding import compose_result_evidence

    recent = next(
        (
            source
            for source in reversed(sources)
            if source.get("event") != "result_banner_cue"
            and 0.0 <= float(banner["start"]) - float(source["end"]) <= max_gap
        ),
        None,
    )
    if recent is None:
        return banner
    action = str(recent.get("livecc_text", "")).strip()
    if not action:
        return banner
    raw = compose_result_evidence(str(banner["livecc_text"]), action)
    banner["livecc_text"] = raw
    banner["livecc"]["metadata"]["raw"] = raw
    banner["context_source_index"] = recent.get("index")
    return banner


def _gemini_result(event: dict[str, Any]) -> dict[str, Any]:
    # Use the streaming API, just like GeminiWorker.  Unlike call_gemini(), do not
    # silently turn API failures into RAG output: evaluation must expose failures.
    from miis_broadcast.core.models.gemini_broadcaster import stream_gemini

    final = None
    for streamed in stream_gemini(event):
        final = streamed
    if final is None:
        raise RuntimeError("Gemini returned no parseable output")
    result = final.to_dict()
    if not result.get("broadcast_text"):
        raise RuntimeError(f"Gemini response has no broadcast_text: {result!r}")
    return result


def _cleanup_livecc(model: Any, state: dict[str, Any]) -> None:
    for key in ("past_key_values", "past_ids", "video_pts"):
        state.pop(key, None)
    state.clear()
    model._cached_video_readers_with_hw.clear()
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        logging.debug("CUDA cleanup failed", exc_info=True)


def _detect_result_banner_records(video: Path, *, max_seconds: float = 0.0) -> list[dict[str, Any]]:
    """Run the same low-latency result tracker used by basketball file-mode GUI."""
    import cv2

    from miis_broadcast.core.models.result_banner_tracker import ResultBannerTracker

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for result tracking: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    sample_every = max(1, int(round(fps / 10.0)))
    tracker = ResultBannerTracker()
    records: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_seconds and frame_index / fps > max_seconds:
                break
            if frame_index % sample_every:
                frame_index += 1
                continue
            cue = tracker.update(frame, frame_index / fps, is_rgb=False)
            if cue is not None:
                result = "Scored!" if cue.kind == "score" else "Out of Bounds!"
                raw = f"{result} {cue.side.title()}" if cue.side else result
                records.append({
                    "event": "result_banner_cue",
                    "start": cue.start,
                    "end": cue.end,
                    "livecc_text": raw,
                    "livecc": {
                        "event": "raw_description",
                        "target": "unknown",
                        "urgency": 3,
                        "metadata": {"raw": raw},
                    },
                    "livecc_latency_s": 0.0,
                    "result_kind": cue.kind,
                    "result_side": cue.side,
                })
            frame_index += 1
    finally:
        cap.release()
    return records


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

    from miis_broadcast.core.models.gemini_broadcaster import (
        load_game_context_file,
        set_language,
        set_style,
    )
    from miis_broadcast.core.models.livecc_transformers import LiveCCInfer
    from miis_broadcast.core.prompt.prompt_manager import PromptManager
    from miis_broadcast.core.utils.config import parse_configs

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not os.getenv("GEMINI_API_KEY") and not args.livecc_only:
        raise RuntimeError("GEMINI_API_KEY is required unless --livecc-only is used")

    models_cfg = parse_configs(args.models_config)
    classifier = dict(models_cfg.get("classifiers", {}).get(args.classifier, {}) or {})
    if not classifier:
        raise ValueError(f"Unknown classifier {args.classifier!r} in {args.models_config}")
    classifier["device_id"] = args.device

    prompt_manager = PromptManager(args.prompts_config, sport=args.sport)
    query = (
        Path(args.query_file).expanduser().read_text(encoding="utf-8").strip()
        if args.query_file
        else (
            prompt_manager.livecc_query_splitscreen()
            if args.layout == "split_screen"
            else prompt_manager.livecc_query()
        )
    )
    if not query:
        raise ValueError(f"No LiveCC query for sport={prompt_manager.current_sport()!r}")

    set_style(args.style)
    set_language(args.language)
    if args.context:
        load_game_context_file(args.context)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = LiveCCInfer(device_id=args.device, classifier_cfg=classifier)
    state = model.init_state(str(video))
    model._cached_video_readers_with_hw.clear()

    counts = {"livecc_segments": 0, "banner_cues": 0, "gemini_success": 0, "gemini_errors": 0}
    latencies: list[float] = []
    pending_gemini: list[dict[str, Any]] = []
    stopped_by_limit = False

    metadata = {
        "event": "run_started",
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video),
        "sport": prompt_manager.current_sport(),
        "style": args.style,
        "language": args.language,
        "layout": args.layout,
        "classifier": args.classifier,
        "device": args.device,
        "livecc_only": args.livecc_only,
        "query": query,
        "response_prefix": args.response_prefix,
    }

    try:
        with output.open("w", encoding="utf-8") as handle:
            _write_jsonl(handle, metadata)
            while True:
                state["video_timestamp"] = time.perf_counter() - started
                produced = 0
                infer_started = time.perf_counter()
                segments: Iterable[Any] = model.live_cc(
                    query,
                    state,
                    response_prefix=(
                        args.response_prefix
                        if args.response_prefix is not None
                        else prompt_manager.livecc_response_prefix()
                    ),
                )
                for (start_t, stop_t), response, state in segments:
                    produced += 1
                    livecc_latency = time.perf_counter() - infer_started
                    if model._is_degenerate(response, query):
                        _write_jsonl(handle, {
                            "event": "livecc_rejected",
                            "start": float(start_t),
                            "end": float(stop_t),
                            "livecc_text": response.strip(),
                            "livecc_latency_s": livecc_latency,
                            "reason": "degenerate_filter",
                        })
                        continue

                    parsed = model._parse_visual_json(response)
                    record: dict[str, Any] = {
                        "event": "livecc_segment",
                        "index": counts["livecc_segments"],
                        "start": float(start_t),
                        "end": float(stop_t),
                        "livecc_text": response.strip(),
                        "livecc": parsed,
                        "livecc_latency_s": round(livecc_latency, 4),
                    }
                    counts["livecc_segments"] += 1
                    _write_jsonl(handle, record)
                    pending_gemini.append(record)
                    print(f"[{start_t:7.2f}-{stop_t:7.2f}] LiveCC: {response.strip()}", flush=True)

                    if args.max_segments and counts["livecc_segments"] >= args.max_segments:
                        stopped_by_limit = True
                        break
                    if args.max_video_seconds and float(stop_t) >= args.max_video_seconds:
                        stopped_by_limit = True
                        break

                if stopped_by_limit or state.get("video_end", False):
                    break
                if produced == 0:
                    time.sleep(0.2)

            # Gemini is intentionally a second phase.  The GUI runs it in another
            # worker thread; blocking on the API inside the LiveCC loop would move
            # video_timestamp forward and change which frames LiveCC observes.
            if not args.livecc_only:
                if args.track_result_banners and prompt_manager.current_sport() == "basketball":
                    scan_limit = max((float(item["end"]) for item in pending_gemini), default=0.0) if stopped_by_limit else 0.0
                    for banner in _detect_result_banner_records(video, max_seconds=scan_limit):
                        _attach_recent_action(banner, pending_gemini)
                        banner["index"] = counts["livecc_segments"] + counts["banner_cues"]
                        counts["banner_cues"] += 1
                        _write_jsonl(handle, banner)
                        pending_gemini.append(banner)
                pending_gemini.sort(key=lambda item: (item["start"], item["end"], item["index"]))
                for source in pending_gemini:
                    gemini_record: dict[str, Any] = {
                        "event": "gemini_result",
                        "index": source["index"],
                        "start": source["start"],
                        "end": source["end"],
                        "gemini": None,
                        "gemini_latency_s": None,
                        "error": None,
                    }
                    gemini_started = time.perf_counter()
                    try:
                        gemini_record["gemini"] = _gemini_result(source["livecc"])
                        counts["gemini_success"] += 1
                    except Exception as exc:
                        counts["gemini_errors"] += 1
                        gemini_record["error"] = {"stage": "gemini", "message": str(exc)}
                        logging.exception("Gemini failed for segment %s", source["index"])
                    gemini_latency = time.perf_counter() - gemini_started
                    gemini_record["gemini_latency_s"] = round(gemini_latency, 4)
                    latencies.append(gemini_latency)
                    _write_jsonl(handle, gemini_record)
                    print(
                        f"[{source['start']:7.2f}-{source['end']:7.2f}] "
                        f"Gemini: {(gemini_record.get('gemini') or {}).get('broadcast_text', '-')}",
                        flush=True,
                    )

            summary = {
                "event": "run_finished",
                **counts,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "avg_gemini_latency_s": (
                    round(sum(latencies) / len(latencies), 4) if latencies else None
                ),
                "stopped_by_limit": stopped_by_limit,
                "output": str(output),
            }
            _write_jsonl(handle, summary)
            return summary
    finally:
        _cleanup_livecc(model, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless LiveCC -> Gemini video evaluation (GUI is unaffected)"
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", default="eval/results/commentary_pipeline.jsonl")
    parser.add_argument("--sport", choices=("basketball", "boxing"), default=None)
    parser.add_argument("--style", default="objective")
    parser.add_argument("--language", choices=("en", "zh"), default="zh")
    parser.add_argument("--layout", choices=("single", "split_screen"), default="split_screen")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--classifier", default="livecc_7b")
    parser.add_argument("--context", help="Optional UTF-8 game context file")
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Optional LiveCC query text file; overrides the configured sport/layout query",
    )
    parser.add_argument(
        "--response-prefix",
        default=None,
        help="Optional LiveCC response prefix; overrides the configured prefix",
    )
    parser.add_argument("--max-segments", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--max-video-seconds", type=float, default=0, help="0 means unlimited")
    parser.add_argument("--livecc-only", action="store_true", help="Skip Gemini API calls")
    parser.add_argument(
        "--no-result-tracker",
        dest="track_result_banners",
        action="store_false",
        help="Disable the basketball Home/Away result tracker used by the GUI",
    )
    parser.set_defaults(track_result_banners=True)
    parser.add_argument("--models-config", type=Path, default=PROJECT_ROOT / "configs/models.yml")
    parser.add_argument("--prompts-config", type=Path, default=PROJECT_ROOT / "configs/livecc_prompts.yml")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    )
    try:
        summary = run_pipeline(args)
    except KeyboardInterrupt:
        logging.warning("Evaluation interrupted")
        return 130
    except Exception as exc:
        logging.error("Evaluation failed: %s", exc)
        return 1
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["gemini_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
