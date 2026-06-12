# src/miis_broadcast/core/models/gemini_tts.py
"""
Gemini TTS backend using google.genai generate_content with AUDIO modality.
Architecture mirrors openai_tts.py: module-level singleton threads + queues,
plus set_natural_completion_callback() for the OpenAITTSWorker interface.
"""

import os
import re
import shutil
import subprocess
import threading
import time
import queue
import logging
from typing import Optional, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# Config
# ==========================================
_tts_cfg_lock = threading.Lock()
_tts_cfg: dict = {
    "model": "models/gemini-3.1-flash-tts-preview",
    "voice": "Kore",
}

def set_tts_voice(voice: str) -> None:
    with _tts_cfg_lock:
        _tts_cfg["voice"] = voice
    logging.info("[GeminiTTS] voice -> %s", voice)

def _get_cfg() -> dict:
    with _tts_cfg_lock:
        return dict(_tts_cfg)

# ==========================================
# Singleton client
# ==========================================
_client = None
_client_lock = threading.Lock()

def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            if not _GEMINI_API_KEY:
                raise RuntimeError("[GeminiTTS] GEMINI_API_KEY not found in environment")
            _client = genai.Client(api_key=_GEMINI_API_KEY)
            logging.info("[GeminiTTS] Client initialized")
        return _client

# ==========================================
# Queues / Events
# ==========================================
_text_queue: "queue.Queue[tuple]" = queue.Queue()
_stop_event = threading.Event()
_interrupt_event = threading.Event()
_tts_threads_started = False

# Bounded depth for routine commentary (drop_outdated=False): keeps speech
# continuous (always something queued up next) without unbounded backlog drift.
_MAX_QUEUE_DEPTH = 2

def clear_text_queue() -> None:
    while not _text_queue.empty():
        try:
            _text_queue.get_nowait()
        except queue.Empty:
            break

def _trim_text_queue(max_depth: int) -> None:
    """Drop oldest pending items until queue has room for one more (depth-bounded FIFO)."""
    while _text_queue.qsize() >= max_depth:
        try:
            _text_queue.get_nowait()
        except queue.Empty:
            break

def contains_meaningful_text(text: Optional[str]) -> bool:
    import re
    if not text:
        return False
    return bool(re.search(r"[\w一-龥]", text))

# ==========================================
# Natural completion callback
# ==========================================
_natural_completion_callback: Optional[Callable] = None
_natural_completion_lock = threading.Lock()

def set_natural_completion_callback(cb: Optional[Callable]) -> None:
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
# Recording sink — receives raw PCM bytes for WAV capture
# ==========================================
_recording_sink: Optional[callable] = None


def register_recording_sink(callback: Optional[callable]) -> None:
    """Register a callback(bytes) to capture TTS audio for recording."""
    global _recording_sink
    _recording_sink = callback


def clear_recording_sink() -> None:
    global _recording_sink
    _recording_sink = None


# ==========================================
# ffplay subprocess (interruptible playback)
# ==========================================
_current_proc: Optional[subprocess.Popen] = None
_current_proc_lock = threading.Lock()

def _play_pcm(data: bytes, sample_rate: int = 24000) -> bool:
    """Play raw PCM16 audio via ffplay. Returns True if interrupted."""
    global _current_proc

    # Forward to recording sink if active
    if _recording_sink is not None and data:
        try:
            _recording_sink(data)
        except Exception:
            pass

    if not shutil.which("ffplay"):
        # No ffplay: simulate duration and check interrupt
        duration = len(data) / (sample_rate * 2)
        start = time.time()
        while time.time() - start < duration:
            if _interrupt_event.is_set():
                return True
            time.sleep(0.05)
        return False

    cmd = [
        "ffplay", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
        "-nodisp", "-i", "pipe:0", "-loglevel", "quiet",
        "-fflags", "nobuffer", "-flags", "low_delay",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    with _current_proc_lock:
        _current_proc = proc

    try:
        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            if _interrupt_event.is_set():
                proc.terminate()
                return True
            try:
                proc.stdin.write(data[i : i + chunk_size])
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return True
        proc.stdin.close()
        proc.wait()
        return _interrupt_event.is_set()
    except Exception:
        return True
    finally:
        with _current_proc_lock:
            if _current_proc is proc:
                _current_proc = None

def _parse_sample_rate(mime_type: str) -> int:
    m = re.search(r"rate=(\d+)", mime_type or "")
    return int(m.group(1)) if m else 24000

# ==========================================
# TTS worker thread
# ==========================================
def _gemini_tts_worker() -> None:
    logging.info("[GeminiTTS] Worker thread started")
    print("🚀 [GeminiTTS] Gemini TTS 背景服務已啟動")
    client = _get_client()
    _first_success = [True]

    while not _stop_event.is_set():
        # Drain any pending interrupt before picking next item
        if _interrupt_event.is_set():
            _interrupt_event.clear()
            continue

        try:
            item = _text_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if isinstance(item, tuple):
            text = item[0]
        else:
            text = item

        if not contains_meaningful_text(text):
            continue

        cfg = _get_cfg()
        interrupted = False

        try:
            response = client.models.generate_content(
                model=cfg["model"],
                contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=cfg["voice"]
                            )
                        )
                    ),
                ),
            )

            # Check interrupt immediately after blocking API call returns
            if _interrupt_event.is_set():
                _interrupt_event.clear()
                interrupted = True
            else:
                if _first_success[0]:
                    print(f"✅ [GeminiTTS] 連線成功！(model={cfg['model']}, voice={cfg['voice']})")
                    _first_success[0] = False
                part = response.candidates[0].content.parts[0]
                audio_bytes: bytes = part.inline_data.data
                sample_rate = _parse_sample_rate(part.inline_data.mime_type)
                interrupted = _play_pcm(audio_bytes, sample_rate)
                if interrupted:
                    _interrupt_event.clear()

        except Exception as e:
            logging.exception("[GeminiTTS] generate_content failed")
            # Only a genuine user-initiated interrupt should suppress the
            # natural-completion signal. A pure API/generation failure (e.g.
            # 429 quota exhaustion) produced no audio and must still fire
            # _fire_natural_completion(), otherwise _post_p1_pending never
            # clears and GeminiBackgroundWorker stays paused forever.
            interrupted = _interrupt_event.is_set()
            if interrupted:
                _interrupt_event.clear()

        if not interrupted:
            _fire_natural_completion()

    logging.info("[GeminiTTS] Worker thread stopped")

# ==========================================
# Public API
# ==========================================
def start_tts_system() -> None:
    global _tts_threads_started
    if _tts_threads_started:
        return
    _stop_event.clear()
    t = threading.Thread(target=_gemini_tts_worker, daemon=True, name="GeminiTTSWorker")
    t.start()
    _tts_threads_started = True
    logging.info("[GeminiTTS] TTS system started")

def stop_tts_system() -> None:
    global _tts_threads_started
    _stop_event.set()
    with _current_proc_lock:
        proc = _current_proc
    if proc:
        proc.terminate()
    _tts_threads_started = False

def enqueue_tts_text(
    text: str,
    ref_ts: float = 0.0,
    drop_outdated: bool = True,
    priority: int = 5,
    start_t: float = 0.0,
) -> None:
    if not contains_meaningful_text(text):
        return
    if drop_outdated:
        clear_text_queue()
    else:
        # Routine commentary: bounded FIFO so continuous broadcast doesn't
        # silently lose every item to the next arrival before it's spoken.
        _trim_text_queue(_MAX_QUEUE_DEPTH)
    _text_queue.put((text, ref_ts, start_t))

def interrupt_tts() -> None:
    """Signal interrupt: kills current ffplay process and clears text queue."""
    clear_text_queue()
    with _current_proc_lock:
        proc = _current_proc
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
    _interrupt_event.set()
