# src/miis_broadcast/core/models/openai_tts.py

import os
import time
import time as _t
import threading
import queue
import asyncio
import json
import base64
import re
import subprocess
import shutil
import logging
from typing import Optional

import numpy as np
import websockets
from dotenv import load_dotenv, find_dotenv
from miis_broadcast.core.utils.config import load_app_config, load_system_prompts

# ==========================================
# 🔐 讀取設定與 API Key
# ==========================================
# find_dotenv() searches upward from this file's location,
# so .env is always found regardless of the working directory.
load_dotenv(find_dotenv(usecwd=False, raise_error_if_not_found=False))
MY_API_KEY = os.getenv("OPENAI_API_KEY")
if not MY_API_KEY:
    raise RuntimeError("[OpenAI TTS] OPENAI_API_KEY not found in environment")

_app_cfg = load_app_config().get("openai_tts", {})
_prompts_cfg = load_system_prompts()

_log = logging.getLogger(__name__)
# Avoid spamming the console when the key is wrong (reconnect every ~2s)
_tts_invalid_api_key_logged: bool = False


def _looks_like_openai_api_key_rejection(msg: str) -> bool:
    s = (msg or "").lower()
    return (
        "invalid_api_key" in s
        or "incorrect api key" in s
        or "invalid api key" in s
        or ("3000" in s and ("invalid_request_error" in s or "registered" in s))
    )


def _log_openai_tts_rejection_once(source: str, msg: str) -> None:
    global _tts_invalid_api_key_logged
    if not _looks_like_openai_api_key_rejection(msg):
        return
    if _tts_invalid_api_key_logged:
        return
    _tts_invalid_api_key_logged = True
    _log.warning(
        "OpenAI TTS: API key rejected (%s). Further identical errors are not printed; "
        "set OPENAI_API_KEY and restart. Detail: %s",
        source,
        (msg or "")[:220],
    )

TTS_MODEL_URL: str = _app_cfg.get("model_url", "wss://api.openai.com/v1/realtime?model=gpt-realtime")
TTS_HEADERS = {
    "Authorization": f"Bearer {MY_API_KEY}",
}
SYSTEM_INSTRUCTIONS: str = _prompts_cfg.get("openai_tts", "")

# ==========================================
# 🧪 Dry-Run 模擬模式（不連 OpenAI，不播音）
# ==========================================
_DRY_RUN: bool = os.environ.get("TTS_DRY_RUN", "0") == "1"

_tts_logger = logging.getLogger("TTS.DryRun")

def enable_dry_run(enabled: bool = True) -> None:
    global _DRY_RUN
    _DRY_RUN = enabled
    if enabled:
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(message)s", datefmt="%H:%M:%S")
        if not _tts_logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(fmt)
            _tts_logger.addHandler(_h)
            _tts_logger.setLevel(logging.DEBUG)
        root = logging.getLogger()
        if not any(
            isinstance(h, logging.StreamHandler) and h.stream.name == "<stdout>"
            for h in root.handlers
        ):
            _rh = logging.StreamHandler()
            _rh.setFormatter(fmt)
            root.addHandler(_rh)
    print(f"[TTS] Dry-run 模式{'已啟用 — 不播音，log 驗證 interrupt 機制' if enabled else '已關閉'}")

# ==========================================
# TTS latency stats
# ==========================================
_perf_stats = {
    "tts_latencies": [],
    "e2e_latencies": [],    # Vision-to-audio end-to-end samples
    "e2e_latencies_seg": [],  # [延遲][語音][段落] — frame-anchored (small start_t)
    "e2e_latencies_bg": [],   # [延遲][語音][背景] — Gemini background narration (epoch start_t)
    "e2e_latencies_interrupt": [],  # [延遲][語音][中斷] — P1/P2 fast-blade interrupts
    "last_text_sent_ts": 0.0,
    "current_ref_ts": 0.0,  # Reference timestamp for current utterance (vision side)
    "current_start_t": 0.0,    # video timestamp of segment being spoken
    "current_priority": 5,      # priority of segment being spoken
}


def _log_tts_latency(value: float) -> None:
    _perf_stats["tts_latencies"].append(value)

def print_tts_stats() -> None:
    """Print TTS and vision-to-audio latency summary when shutting down TTS."""
    print("\n" + "=" * 40)
    print("Latency Performance Report")
    print("=" * 40)

    if _perf_stats["tts_latencies"]:
        avg_tts = sum(_perf_stats["tts_latencies"]) / len(_perf_stats["tts_latencies"])
        print(f"Average TTS Latency (Text->Audio):    {avg_tts:.3f} s")
    else:
        print("Average TTS Latency:                   N/A")

    if _perf_stats["e2e_latencies"]:
        avg_e2e = sum(_perf_stats["e2e_latencies"]) / len(_perf_stats["e2e_latencies"])
        print(f"Average E2E Latency (Vision->Audio):   {avg_e2e:.3f} s")
    else:
        print("Average E2E Latency:                   N/A")

    print("=" * 40 + "\n")

    _perf_stats["tts_latencies"].clear()
    _perf_stats["e2e_latencies"].clear()
    _perf_stats["e2e_latencies_seg"].clear()
    _perf_stats["e2e_latencies_bg"].clear()
    _perf_stats["e2e_latencies_interrupt"].clear()
    _perf_stats["last_text_sent_ts"] = 0.0
    _perf_stats["current_ref_ts"] = 0.0
    _perf_stats["current_start_t"] = 0.0


# ==========================================
# Queues and thread coordination
# ==========================================
_text_queue: "queue.Queue[tuple[str, float]]" = queue.Queue()
# Bounded backlog when inference outruns TTS (drops oldest pending utterances).
_MAX_PENDING_UTTERANCES: int = 40

_audio_output_queue: "queue.Queue[np.ndarray]" = queue.Queue()
_stop_event = threading.Event()
_tts_threads_started = False
_interrupt_event = threading.Event()
_connect_requested = threading.Event()  # set by warmup / enqueue; prevents DNS/SSL at app startup

# ==========================================
# PCM sink (audience second screen)
# ==========================================
# When registered, each PCM chunk is forwarded to the sink callback
# (e.g. AudiencePublisher.push_audio_chunk) in addition to or instead of
# local playback depending on _pcm_sink_mute_local.
_pcm_sink: Optional[callable] = None
_pcm_sink_mute_local: bool = False
_pcm_sink_flush: Optional[callable] = None  # e.g. flush LiveKit audio queue on interrupt

# Recording sink — receives each int16 numpy chunk for WAV capture
_recording_sink: Optional[callable] = None


def register_recording_sink(callback: Optional[callable]) -> None:
    """Register a callback(np.ndarray int16) to capture TTS audio for recording."""
    global _recording_sink
    _recording_sink = callback


def clear_recording_sink() -> None:
    global _recording_sink
    _recording_sink = None


def register_pcm_sink(
    callback: Optional[callable],
    mute_local: bool = True,
    flush_callback: Optional[callable] = None,
) -> None:
    """Register a PCM sink for the audience publisher.

    Args:
        callback: called with each np.ndarray int16 chunk, or None to clear.
        mute_local: if True, suppress local audio playback while sink is active.
        flush_callback: called inside clear_audio_queue() to drop audience-side buffers
            (e.g. LiveKit pending queue) so interrupted TTS does not overlap on viewers.
    """
    global _pcm_sink, _pcm_sink_mute_local, _pcm_sink_flush
    _pcm_sink = callback
    _pcm_sink_mute_local = mute_local if callback is not None else False
    _pcm_sink_flush = flush_callback if callback is not None else None
    tag = "[AUDIO] PCM sink registered" if callback is not None else "[AUDIO] PCM sink cleared"
    print(f"{time.strftime('%H:%M:%S')} | {tag} | mute_local={_pcm_sink_mute_local}")

# ==========================================
# Runtime TTS settings (voice / speed)
# ==========================================
_TTS_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"}
_tts_cfg_lock = threading.Lock()
_tts_cfg = {
    "voice": _app_cfg.get("default_voice", "coral"),
    "speed": float(_app_cfg.get("default_speed", 1.5)),
}

_cfg_update_event = threading.Event()
_voice_update_event = threading.Event()

def set_tts_voice(voice: str) -> None:
    voice = (voice or "").strip()
    if voice not in _TTS_VOICES:
        return
    with _tts_cfg_lock:
        if _tts_cfg["voice"] != voice:
            _tts_cfg["voice"] = voice
            _voice_update_event.set()
            # Voice change must interrupt the current utterance
            _interrupt_event.set()
            print(f"🎚️ [TTS] voice set -> {voice}")

def set_tts_speed(speed: float) -> None:
    try: speed = float(speed)
    except: return
    speed = max(0.5, min(2.0, speed))
    with _tts_cfg_lock:
        if abs(_tts_cfg["speed"] - speed) > 1e-6:
            _tts_cfg["speed"] = speed
            _cfg_update_event.set()
            print(f"🎚️ [TTS] speed set -> {speed:.1f}x")

def _get_tts_cfg_snapshot() -> dict:
    with _tts_cfg_lock:
        return dict(_tts_cfg)

def _build_instructions(speed: float) -> str:
    return SYSTEM_INSTRUCTIONS + f"\n\n[系統參數] 語速倍率：{speed:.1f}x（請盡量遵守）\n"

def _build_session_payload(cfg: dict, *, full: bool = True) -> dict:
    """Build a GA-shape Realtime `session` object.

    The legacy beta shape (top-level `modalities` / `voice` / `speed` /
    `input_audio_format`) was disabled by OpenAI on 2026-05-12 and now returns
    `beta_api_shape_disabled`. GA requires `type: "realtime"`, `output_modalities`,
    and the nested `audio.output` block. `speed` is clamped to the GA range [0.25, 1.5].
    Note: GA Realtime rejects top-level `session.temperature` (use instructions only)."""
    speed = max(0.25, min(1.5, float(cfg.get("speed", 1.0))))
    session: dict = {
        "type": "realtime",
        "instructions": _build_instructions(speed),
        "audio": {
            "output": {
                "voice": cfg.get("voice", "coral"),
                "speed": speed,
                "format": {"type": "audio/pcm", "rate": 24000},
            }
        },
    }
    if full:
        # Audio-only output; transcript is delivered alongside automatically.
        session["output_modalities"] = ["audio"]
    return session

def contains_meaningful_text(text: Optional[str]) -> bool:
    if not text: return False
    return bool(re.search(r"[\w\u4e00-\u9fa5]", text))

def clear_text_queue() -> None:
    while not _text_queue.empty():
        try: _text_queue.get_nowait()
        except queue.Empty: break

def clear_audio_queue() -> None:
    while not _audio_output_queue.empty():
        try: _audio_output_queue.get_nowait()
        except queue.Empty: break
    if _pcm_sink_flush is not None:
        try:
            _pcm_sink_flush()
        except Exception:
            pass


def interrupt_tts(clear_text: bool = True) -> None:
    """Hard interrupt: e.g. Stop inference or explicit user cancel."""
    if clear_text:
        clear_text_queue()
    clear_audio_queue()
    _perf_stats["last_text_sent_ts"] = 0.0
    _interrupt_event.set()
    if _DRY_RUN:
        _tts_logger.info("⚡ [INTERRUPT Path-1] interrupt_tts() called → _interrupt_event set")


# ==========================================
# 🔔 Natural completion callback
# ==========================================
_natural_completion_callback = None
_natural_completion_lock = threading.Lock()

def set_natural_completion_callback(cb) -> None:
    """Register a callable fired on natural TTS completion (not on cancel/interrupt).
    Called from the TTS async thread — the callback must be thread-safe."""
    global _natural_completion_callback
    with _natural_completion_lock:
        _natural_completion_callback = cb

def _fire_natural_completion() -> None:
    with _natural_completion_lock:
        cb = _natural_completion_callback
    if cb is not None:
        try:
            cb()
        except Exception:
            pass


# ==========================================
# 🧪 Mock TTS Worker（Dry-Run 模式）
# ==========================================
def _mock_tts_worker() -> None:
    """不連 OpenAI、不播音，用 log 驗證 interrupt 機制是否正確執行。"""
    _tts_logger.info("=" * 60)
    _tts_logger.info("[DRY-RUN] Mock TTS worker 啟動")
    _tts_logger.info("=" * 60)
    time.sleep(0.3)
    _tts_logger.info("[DRY-RUN] Warmup 完成 (模擬)，開始監聽文字佇列")

    while not _stop_event.is_set():

        # Path 1 interrupt（priority signal 或手動觸發）
        if _interrupt_event.is_set():
            clear_audio_queue()
            _interrupt_event.clear()
            _tts_logger.info("⚡ [INTERRUPT Path-1] _interrupt_event 已清除（idle 中收到）")
            continue

        try:
            item = _text_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        _item = item if isinstance(item, tuple) else (item, 0.0, 0.0)
        text, ref_ts = _item[0], (_item[1] if len(_item) > 1 else 0.0)
        if not contains_meaningful_text(text):
            continue

        word_count = len(text.split())
        duration = max(1.0, word_count * 0.25)   # ~250ms/word 估算語音長度
        preview = text[:70] + ("…" if len(text) > 70 else "")

        _tts_logger.info(f'▶ [SPEAK START] [{word_count}w ~{duration:.1f}s] "{preview}"')

        elapsed = 0.0
        interrupted = False
        interrupt_path = ""

        while elapsed < duration and not _stop_event.is_set():
            time.sleep(0.05)
            elapsed += 0.05

            # Path 1: priority signal 在播放中途觸發
            if _interrupt_event.is_set():
                clear_audio_queue()
                _interrupt_event.clear()
                interrupted = True
                interrupt_path = "Path-1 (priority signal / manual)"
                break

            # Path 2: 有新文字進入 queue（模擬 WebSocket auto-cancel）
            if not _text_queue.empty():
                interrupted = True
                interrupt_path = "Path-2 (new text in queue)"
                break

        pct = int(elapsed / duration * 100)
        if interrupted:
            _tts_logger.info(
                f'⚡ [SPEAK INTERRUPTED {elapsed:.2f}s/{duration:.1f}s ({pct}%)] by {interrupt_path}'
            )
            _tts_logger.info(f'   └─ was: "{preview}"')
        else:
            _tts_logger.info(f'✓ [SPEAK DONE {duration:.1f}s] "{preview}"')
            _fire_natural_completion()


# ==========================================
# OpenAI Realtime WebSocket worker
# ==========================================
async def _openai_realtime_worker():
    # Wait until warmup_tts_connection() (on Start) or enqueue_tts_text() before
    # opening WebSocket — avoids DNS/SSL at app startup racing PyTorch CUDA threads.
    while not _stop_event.is_set() and not _connect_requested.is_set():
        await asyncio.sleep(0.2)
    if _stop_event.is_set():
        return
    print("🎙️ [TTS Worker] 啟動連線...")

    # Stop reconnecting once we know the API key is wrong
    while not _stop_event.is_set() and not _tts_invalid_api_key_logged:
        is_response_active = False
        warmed_up = False
        doing_warmup = False
        awaiting_cancel_ack = False

        try:
            async with websockets.connect(TTS_MODEL_URL, additional_headers=TTS_HEADERS) as websocket:
                # Don't print repeated "connected" messages after a known key rejection
                if not _tts_invalid_api_key_logged:
                    print("✅ [TTS Worker] connected")
                cfg = _get_tts_cfg_snapshot()

                # 1. Session Update (GA shape — see _build_session_payload)
                # warmed_up becomes True when OpenAI echoes back "session.updated"
                # (typically <300 ms) — no audio warmup round-trip needed.
                await websocket.send(json.dumps({
                    "type": "session.update",
                    "session": _build_session_payload(cfg, full=True),
                }))

                # Main loop: send config / text, recv deltas and lifecycle events
                while not _stop_event.is_set():

                    # (A) Voice change requests a websocket reconnect
                    if _voice_update_event.is_set():
                        _voice_update_event.clear()
                        await websocket.close()
                        break 

                    if _cfg_update_event.is_set():
                        cfg2 = _get_tts_cfg_snapshot()
                        await websocket.send(json.dumps({
                            "type": "session.update",
                            "session": _build_session_payload(cfg2, full=False),
                        }))
                        _cfg_update_event.clear()

                    # (B) Hard interrupt: cancel active response, clear audio + audience flush
                    if _interrupt_event.is_set():
                        if is_response_active:
                            await websocket.send(json.dumps({"type": "response.cancel"}))
                            awaiting_cancel_ack = True
                        clear_audio_queue()
                        is_response_active = False
                        _interrupt_event.clear()

                    # (C) 送出新文字邏輯 (加入對 active 狀態的嚴格檢查)
                    if warmed_up and not awaiting_cancel_ack:
                        if not _text_queue.empty():
                            target_text = None
                            ref_ts = 0.0

                            # 從 Queue 取出 (text, ts, start_t, priority)
                            priority = 5
                            while not _text_queue.empty():
                                item = _text_queue.get_nowait()
                                if isinstance(item, tuple):
                                    target_text = item[0]
                                    ref_ts = item[1] if len(item) > 1 else 0.0
                                    start_t = item[2] if len(item) > 2 else 0.0
                                    priority = item[3] if len(item) > 3 else 5
                                else:
                                    target_text, ref_ts, start_t = item, 0.0, 0.0

                            if target_text and contains_meaningful_text(target_text):
                                # 如果目前有在說話，先發送取消並等待確認
                                if is_response_active:
                                    await websocket.send(json.dumps({"type": "response.cancel"}))
                                    awaiting_cancel_ack = True
                                    clear_audio_queue()  # 同步清空已緩衝的音訊，避免舊內容繼續播
                                    _text_queue.put((target_text, ref_ts, start_t, priority))
                                    continue  # 跳出本次循環，去聽事件 (D)

                                # 確定沒有 active response，才發送
                                _perf_stats["last_text_sent_ts"] = time.time()
                                _perf_stats["current_ref_ts"] = ref_ts
                                _perf_stats["current_start_t"] = start_t
                                _perf_stats["current_priority"] = priority

                                await websocket.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": target_text}]},
                                }))
                                await websocket.send(json.dumps({"type": "response.create"}))
                                is_response_active = True

                    # (D) 接收 WebSocket 事件
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        event = json.loads(message)
                        etype = event.get("type", "")

                        if etype == "session.updated":
                            if not warmed_up:
                                warmed_up = True
                                print("✅ [TTS Worker] session.updated — ready for text")

                        # GA renamed response.audio.delta → response.output_audio.delta.
                        # Accept both so a future endpoint change can't silently mute us.
                        elif etype in ("response.output_audio.delta", "response.audio.delta"):
                            if warmed_up:
                                audio_bytes = base64.b64decode(event["delta"])
                                _audio_output_queue.put(np.frombuffer(audio_bytes, dtype=np.int16))

                        elif etype in ["response.done", "response.cancelled"]:
                            if etype == "response.done" and not warmed_up and doing_warmup:
                                warmed_up = True
                                doing_warmup = False
                                print("✅ [TTS Worker] Warmup 完成。")
                            elif etype == "response.done" and warmed_up:
                                # Natural completion (not warmup, not cancelled) — notify worker
                                _fire_natural_completion()

                            # 伺服器端已經清空狀態，現在可以接收新 response 了
                            is_response_active = False
                            awaiting_cancel_ack = False

                        elif etype == "error":
                            err = event.get("error", {})
                            msg = err.get("message", "")
                            # "active response" hints: align local flag with server state
                            if "active response" in msg:
                                is_response_active = True
                            if err.get("code") == "response_cancel_not_active":
                                pass
                            elif _looks_like_openai_api_key_rejection(str(msg) + " " + str(err.get("type", ""))):
                                _log_openai_tts_rejection_once("server_event", str(msg))
                            else:
                                print(f"❌ [OpenAI Error] {msg}")

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed as _cc_exc:
                        # Server may close with code 3000 + invalid_api_key reason
                        _cc_msg = str(_cc_exc)
                        if _looks_like_openai_api_key_rejection(_cc_msg):
                            _log_openai_tts_rejection_once("connection_closed", _cc_msg)
                        break  # exit inner recv loop; outer while checks the flag

        except Exception as e:
            if _stop_event.is_set() or _tts_invalid_api_key_logged:
                break
            es = str(e)
            if _looks_like_openai_api_key_rejection(es):
                _log_openai_tts_rejection_once("websocket", es)
                break  # don't sleep-and-retry for a known bad key
            else:
                print(f"❌ [TTS Error] {e}")
            await asyncio.sleep(2)

# ==========================================
# Playback worker (local device + PCM sink)
# ==========================================
def _apply_audio_chunk_latency_stats() -> None:
    """First audio chunk per utterance: record TTS and E2E latency stats."""
    if _perf_stats["last_text_sent_ts"] > 0:
        latency = time.time() - _perf_stats["last_text_sent_ts"]
        _log_tts_latency(latency)
        _perf_stats["last_text_sent_ts"] = 0.0
    if _perf_stats["current_ref_ts"] > 0:
        e2e_latency = time.time() - _perf_stats["current_ref_ts"]
        _perf_stats["e2e_latencies"].append(e2e_latency)
        if _perf_stats["current_priority"] <= 2:
            bucket_key, tag = "e2e_latencies_interrupt", "中斷"
        elif _perf_stats["current_start_t"] > 1e6:
            bucket_key, tag = "e2e_latencies_bg", "背景"
        else:
            bucket_key, tag = "e2e_latencies_seg", "段落"
        bucket = _perf_stats[bucket_key]
        bucket.append(e2e_latency)
        logging.info(
            "[延遲][語音][%s] latency=%.2fs (平均=%.2fs, n=%d)",
            tag, e2e_latency, sum(bucket) / len(bucket), len(bucket),
        )
        _perf_stats["current_ref_ts"] = 0.0


def _audio_player_worker_sounddevice() -> None:
    """Play PCM int16 @ 24 kHz mono via PortAudio (works on Windows without ffplay)."""
    import sounddevice as sd

    stream = sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype="int16",
        latency=0.2,
    )
    stream.start()
    print("🔊 [TTS] playing via sounddevice (default output device)")

    while not _stop_event.is_set():
        try:
            audio_chunk = _audio_output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        _apply_audio_chunk_latency_stats()

        # Forward to audience PCM sink if registered
        if _pcm_sink is not None and audio_chunk is not None:
            try:
                _pcm_sink(np.ascontiguousarray(audio_chunk, dtype=np.int16))
            except Exception:
                pass

        # Forward to recording sink if active
        if _recording_sink is not None and audio_chunk is not None:
            try:
                _recording_sink(np.ascontiguousarray(audio_chunk, dtype=np.int16))
            except Exception:
                pass

        try:
            if audio_chunk is not None and getattr(audio_chunk, "size", 0) > 0:
                x = np.ascontiguousarray(audio_chunk, dtype=np.int16).reshape(-1, 1)
                if _pcm_sink_mute_local:
                    # Write silence to keep hardware clock pacing — without this the worker
                    # races through _audio_output_queue at CPU speed, dumping all PCM into
                    # the LiveKit queue in one burst then going silent (sounds "cut off").
                    stream.write(np.zeros_like(x))
                else:
                    stream.write(x)
        except Exception as e:
            _log.debug("sounddevice write: %s", e)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass


def _audio_player_worker_ffplay() -> None:
    """Play via ffplay raw PCM pipe (legacy path)."""
    cmd = [
        "ffplay", "-f", "s16le", "-ar", "24000", "-ac", "1", "-nodisp",
        "-i", "pipe:0", "-loglevel", "quiet", "-fflags", "nobuffer",
        "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
    ]
    process: Optional[subprocess.Popen] = None

    while not _stop_event.is_set():
        try:
            audio_chunk = _audio_output_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        _apply_audio_chunk_latency_stats()

        # Forward to audience PCM sink if registered
        if _pcm_sink is not None and audio_chunk is not None:
            try:
                _pcm_sink(np.ascontiguousarray(audio_chunk, dtype=np.int16))
            except Exception:
                pass

        # Forward to recording sink if active
        if _recording_sink is not None and audio_chunk is not None:
            try:
                _recording_sink(np.ascontiguousarray(audio_chunk, dtype=np.int16))
            except Exception:
                pass

        if process is None or process.poll() is not None:
            try:
                process = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
                )
            except OSError as e:
                _log.debug("ffplay Popen: %s", e)
                process = None

        if process and process.stdin:
            try:
                # Send silence when muted to keep ffplay's internal clock running (same
                # reason as sounddevice: prevents burst-then-silence on LiveKit side).
                payload = (
                    np.zeros(len(audio_chunk), dtype=np.int16).tobytes()
                    if _pcm_sink_mute_local
                    else audio_chunk.tobytes()
                )
                process.stdin.write(payload)
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                process = None

    if process and process.poll() is None:
        try:
            process.terminate()
        except Exception:
            pass


def _audio_player_worker() -> None:
    # Prefer sounddevice (requirements.txt): works on Windows without ffplay in PATH.
    try:
        _audio_player_worker_sounddevice()
        return
    except Exception as e:
        _log.warning("sounddevice playback path failed: %s", e, exc_info=True)
        print(f"⚠️ [TTS] sounddevice unavailable ({e!s}), trying ffplay…")

    if shutil.which("ffplay"):
        print("🔊 [TTS] using ffplay for playback")
        _audio_player_worker_ffplay()
        return

    _log.error(
        "No audio backend: install sounddevice (pip) or add ffplay (FFmpeg) to PATH"
    )
    print(
        "❌ [TTS] no audio backend: install sounddevice (pip) or add ffplay (FFmpeg) to PATH"
    )


# ==========================================
# Public API
# ==========================================
def start_tts_system() -> None:
    global _tts_threads_started
    if _tts_threads_started: return
    # Audio: prefer sounddevice; ffplay is optional fallback (see _audio_player_worker).
    if not shutil.which("ffplay"):
        print(
            "ℹ️ [TTS] ffplay not found; using sounddevice (no FFmpeg required)."
        )
    _stop_event.clear()
    if _DRY_RUN:
        threading.Thread(target=_mock_tts_worker, daemon=True).start()
        print("🧪 [TTS] Dry-run 模式 — Mock TTS worker 已啟動（不播音）")
    else:
        t1 = threading.Thread(target=lambda: asyncio.run(_openai_realtime_worker()), daemon=True)
        t1.start()
        t2 = threading.Thread(target=_audio_player_worker, daemon=True)
        t2.start()
    _tts_threads_started = True
    print("🚀 [TTS] OpenAI TTS background service started")

def stop_tts_system() -> None:
    global _tts_threads_started
    _stop_event.set()
    _tts_threads_started = False

def warmup_tts_connection() -> None:
    """Pre-connect OpenAI Realtime on Start so the first utterance skips cold-start delay."""
    start_tts_system()
    _connect_requested.set()
    print("🔥 [TTS] warmup connect requested (pre-connect on Start)")

def _is_silence_token(text: str) -> bool:
    """Return True when the model returned the sentinel word 'silence' (case-insensitive).
    The system prompt instructs gpt-realtime to output this word instead of speaking
    when the input is empty, so we intercept it here and skip TTS entirely."""
    return text.strip().lower() == "silence"

def enqueue_tts_text(text: str, ref_ts: float = 0.0, drop_outdated: bool = True, priority: int = 5, start_t: float = 0.0) -> None:
    if _is_silence_token(text):
        return  # model signalled silence — do not play anything
    if contains_meaningful_text(text):
        _connect_requested.set()
        if drop_outdated:
            clear_text_queue()
        else:
            while _text_queue.qsize() >= _MAX_PENDING_UTTERANCES:
                try:
                    _text_queue.get_nowait()
                except queue.Empty:
                    break

        # ref_ts=0 means caller did not attach a vision timestamp; use wall clock
        ts = ref_ts if ref_ts > 0 else time.time()
        _text_queue.put((text, ts, start_t, priority))

        if _DRY_RUN:
            preview = text[:60] + ("…" if len(text) > 60 else "")
            _tts_logger.info(f'📥 [ENQUEUE P{priority}] "{preview}"')
