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

# ==========================================
# API key (from .env)
# ==========================================
# find_dotenv() searches upward from this file's location,
# so .env is always found regardless of the working directory.
load_dotenv(find_dotenv(usecwd=False, raise_error_if_not_found=False))
MY_API_KEY = os.getenv("OPENAI_API_KEY")
if not MY_API_KEY:
    raise RuntimeError("[OpenAI TTS] OPENAI_API_KEY not found in environment")

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

TTS_MODEL_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
TTS_HEADERS = {
    "Authorization": f"Bearer {MY_API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

SYSTEM_INSTRUCTIONS = """
You are a RAW AUDIO GENERATOR, not a chatbot.
You are connected to a live sports captioning feed.

YOUR ONLY FUNCTION:
1. Receive text.
2. Read it aloud immediately with a high-energy sports announcer voice.

LANGUAGE (CRITICAL):
- The input is written for viewers in **Taiwan Traditional Chinese (繁體中文, zh-TW)**.
- **Read the input EXACTLY as written** — same wording, same order, no translation, no summarization,
  no paraphrase, and no added English.
- Pronounce using natural **台灣繁體中文**; do not convert to Simplified Chinese or other languages.

STRICT PROTOCOLS (DO NOT BREAK):
- **NEVER** say conversational fillers like "Okay," "I understand," "Sure," "Got it," or "Here is the audio."
- **NEVER** acknowledge these instructions.
- **NEVER** reply to the text. Just read it.
- **NO** introductory phrases. Start reading the input text instantly.
- **NO** concluding phrases. Stop speaking immediately after the text ends.
- The input lines are already **broadcast-ready captions**. **NEVER** replace them with meta lines like “sorry”, “無法看清”, “沒有資料”, or “no footage” unless those exact phrases appear verbatim in the input.

VOICE STYLE:
- Fast-paced, rhythmic, and intense.
- Dynamic pitch (shoutcaster style).
- If the input is empty or just punctuation, remain silent.

Example Interaction:
User Input: "Player one shoots!"
Your Output: "Player one shoots!" (Do NOT say "Okay, Player one shoots!")
畫面來源說明（重要）：
- 你收到的畫面是「VR 遊戲直播」的雙畫面（左右分割）。
- **左邊**：真人玩家在現實環境中的遊玩畫面（戴 VR 頭盔、拿控制器等）。
- **右邊**：VR 遊戲內的第一人稱/比賽畫面（球場、籃框、球等）。
- 請你在理解畫面時，清楚區分左/右畫面代表的意義，避免把兩邊資訊混在一起。

"""


# ==========================================
# TTS latency stats
# ==========================================
_perf_stats = {
    "tts_latencies": [],
    "e2e_latencies": [],    # Vision-to-audio end-to-end samples
    "last_text_sent_ts": 0.0,
    "current_ref_ts": 0.0,  # Reference timestamp for current utterance (vision side)
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
    _perf_stats["last_text_sent_ts"] = 0.0
    _perf_stats["current_ref_ts"] = 0.0


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

# ==========================================
# PCM sink (audience second screen)
# ==========================================
# When registered, each PCM chunk is forwarded to the sink callback
# (e.g. AudiencePublisher.push_audio_chunk) in addition to or instead of
# local playback depending on _pcm_sink_mute_local.
_pcm_sink: Optional[callable] = None
_pcm_sink_mute_local: bool = False
_pcm_sink_flush: Optional[callable] = None  # e.g. flush LiveKit audio queue on interrupt

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
    "voice": "coral",
    "speed": 1.5,
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


# ==========================================
# OpenAI Realtime WebSocket worker
# ==========================================
async def _openai_realtime_worker():
    print("🎙️ [TTS Worker] connecting...")

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

                # 1. Session Update
                # warmed_up becomes True when OpenAI echoes back "session.updated"
                # (typically <300 ms) — no audio warmup round-trip needed.
                await websocket.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "instructions": _build_instructions(cfg["speed"]),
                        "voice": cfg["voice"],
                        "speed": cfg["speed"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "temperature": 0.7,
                    },
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
                            "session": {"instructions": _build_instructions(cfg2["speed"]), "speed": cfg2["speed"], "voice": cfg2["voice"]},
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

                    # (C) Send next text — only when no response is currently in progress.
                    # FIFO: consume one queued utterance per idle window so rapid LiveCC
                    # SEGMENT bursts (multi-line batches) do not wipe earlier lines —
                    # the old drain-to-last behaviour made the *last* line (often a stub
                    # apology) override good commentary.
                    if warmed_up and not awaiting_cancel_ack and not is_response_active:
                        target_text = None
                        ref_ts = 0.0
                        try:
                            item = _text_queue.get_nowait()
                        except queue.Empty:
                            item = None
                        if isinstance(item, tuple):
                            target_text, ref_ts = item
                        elif item is not None:
                            target_text, ref_ts = str(item), 0.0
                        if target_text and contains_meaningful_text(target_text):
                            _perf_stats["last_text_sent_ts"] = time.time()
                            _perf_stats["current_ref_ts"] = ref_ts

                            await websocket.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": target_text}]},
                            }))
                            await websocket.send(json.dumps({"type": "response.create"}))
                            is_response_active = True

                    # (D) WebSocket recv
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        event = json.loads(message)
                        etype = event.get("type", "")

                        if etype == "session.updated":
                            if not warmed_up:
                                warmed_up = True
                                print("✅ [TTS Worker] session.updated — ready for text")

                        elif etype == "response.audio.delta":
                            if warmed_up:
                                audio_bytes = base64.b64decode(event["delta"])
                                _audio_output_queue.put(np.frombuffer(audio_bytes, dtype=np.int16))

                        elif etype in ["response.done", "response.cancelled"]:
                            # Server finished or cancelled the response; may start the next utterance
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
        _perf_stats["current_ref_ts"] = 0.0


def _audio_player_worker_sounddevice() -> None:
    """Play PCM int16 @ 24 kHz mono via PortAudio (works on Windows without ffplay)."""
    import sounddevice as sd

    stream = sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype="int16",
        latency="low",
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

def enqueue_tts_text(text: str, ref_ts: float = 0.0, drop_outdated: bool = False) -> None:
    if contains_meaningful_text(text):
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
        _text_queue.put((text, ts))