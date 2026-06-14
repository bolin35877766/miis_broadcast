# src/miis_broadcast/core/models/gemini_tts.py
"""
Gemini TTS backend using google.genai generate_content with AUDIO modality.
Architecture mirrors openai_tts.py: module-level singleton threads + queues,
plus set_natural_completion_callback() for the OpenAITTSWorker interface.
"""

import concurrent.futures
import os
import re
import shutil
import subprocess
import threading
import time
import queue
import logging
from typing import Optional, Callable

import numpy as np
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

# [延遲][語音] running averages, split by origin:
# - priority<=2 (P1/P2 fast-blade): "中斷" — dimension 2, frame -> sound for interrupts.
# - "_background"-tagged items carry an epoch start_t (>1e6): "背景" — Gemini's own
#   continuous narration, no LiveCC frame anchor.
# - everything else carries a small video-relative start_t: "段落" — dimension 1,
#   normal frame -> spoken latency for Gemini-enriched commentary.
_voice_latencies_seg: list = []
_voice_latencies_bg: list = []
_voice_latencies_interrupt: list = []

# [延遲][語音][段間]: dead-air between one utterance's playback ending and the
# next one's audio becoming ready (queue-wait + generation-wait). Diagnoses
# "gap between consecutive voice clips" independent of per-utterance latency.
_voice_gaps: list = []

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
                _soft_stop_proc(proc)
                return True
            try:
                proc.stdin.write(data[i : i + chunk_size])
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
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

# Linear fade-in/out applied to every utterance before playback. Each utterance
# spawns a fresh ffplay process, so an un-faded clip starts/ends at full
# amplitude right as the audio device opens/closes — audible as a click/pop.
_FADE_MS = 15

def _apply_fade(data: bytes, sample_rate: int) -> bytes:
    if len(data) < 4:
        return data
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    fade_n = min(len(samples) // 2, int(sample_rate * _FADE_MS / 1000))
    if fade_n > 1:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        samples[:fade_n] *= ramp
        samples[-fade_n:] *= ramp[::-1]
    return samples.astype(np.int16).tobytes()

def _soft_stop_proc(proc: subprocess.Popen) -> None:
    """Stop ffplay without an abrupt kill: close stdin (EOF) so ffplay drains
    its small buffer and exits on its own; fall back to SIGTERM only if it
    doesn't exit quickly. Avoids the pop/click from killing ffplay mid-DMA."""
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=0.06)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
        except Exception:
            pass

# ==========================================
# Generation helper + prefetch executor
# ==========================================
# A single background slot used to generate the *next* utterance's audio
# while the *current* one is still playing — closes the generate-then-play
# gap (~0.5-2s of silence per utterance) that causes audible 卡頓.
_tts_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="GeminiTTSGen"
)

def _generate_audio(client, cfg: dict, text: str) -> tuple:
    """Blocking call: generate_content + fade. Returns (audio_bytes, sample_rate)."""
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
    part = response.candidates[0].content.parts[0]
    audio_bytes: bytes = part.inline_data.data
    sample_rate = _parse_sample_rate(part.inline_data.mime_type)
    audio_bytes = _apply_fade(audio_bytes, sample_rate)
    return audio_bytes, sample_rate

# ==========================================
# TTS worker thread
# ==========================================
def _gemini_tts_worker() -> None:
    logging.info("[GeminiTTS] Worker thread started")
    print("🚀 [GeminiTTS] Gemini TTS 背景服務已啟動")
    client = _get_client()
    _first_success = [True]
    prefetch: Optional[tuple] = None  # (Future, item) for the next utterance
    prev_play_end: float = 0.0  # wall-clock time the previous utterance's playback ended

    while not _stop_event.is_set():
        # Drain any pending interrupt before picking next item
        if _interrupt_event.is_set():
            _interrupt_event.clear()
            prefetch = None
            continue

        if prefetch is not None:
            future, item = prefetch
            prefetch = None
            prefetch_hit = True
        else:
            try:
                item = _text_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            future = None
            prefetch_hit = False

        t_dequeue = time.time()

        if isinstance(item, tuple):
            text = item[0]
            ref_ts = item[1] if len(item) > 1 else 0.0
            start_t = item[2] if len(item) > 2 else 0.0
            priority = item[3] if len(item) > 3 else 5
        else:
            text, ref_ts, start_t, priority = item, 0.0, 0.0, 5

        if future is None and not contains_meaningful_text(text):
            continue

        cfg = _get_cfg()
        interrupted = False

        try:
            audio_bytes, sample_rate = (
                future.result() if future is not None else _generate_audio(client, cfg, text)
            )
            t_ready = time.time()

            # Check interrupt immediately after the (blocking or prefetched) call returns
            if _interrupt_event.is_set():
                _interrupt_event.clear()
                interrupted = True
            else:
                if _first_success[0]:
                    print(f"✅ [GeminiTTS] 連線成功！(model={cfg['model']}, voice={cfg['voice']})")
                    _first_success[0] = False
                if ref_ts > 0:
                    latency = time.time() - ref_ts
                    if priority <= 2:
                        bucket, tag = _voice_latencies_interrupt, "中斷"
                    elif start_t > 1e6:
                        bucket, tag = _voice_latencies_bg, "背景"
                    else:
                        bucket, tag = _voice_latencies_seg, "段落"
                    bucket.append(latency)
                    avg = sum(bucket) / len(bucket)
                    logging.info(
                        "[延遲][語音][%s] latency=%.2fs (平均=%.2fs, n=%d)",
                        tag, latency, avg, len(bucket),
                    )

                if prev_play_end > 0:
                    queue_wait = max(0.0, t_dequeue - prev_play_end)
                    gen_wait = max(0.0, t_ready - t_dequeue)
                    total_gap = max(0.0, t_ready - prev_play_end)
                    _voice_gaps.append(total_gap)
                    logging.info(
                        "[延遲][語音][段間] gap=%.2fs (排隊=%.2fs, 生成=%.2fs, prefetch=%s, 平均=%.2fs, n=%d)",
                        total_gap, queue_wait, gen_wait,
                        "命中" if prefetch_hit else "未命中",
                        sum(_voice_gaps) / len(_voice_gaps), len(_voice_gaps),
                    )

                # Phase 1: immediate prefetch — grab item already in queue.
                while True:
                    try:
                        next_item = _text_queue.get_nowait()
                    except queue.Empty:
                        break
                    next_text = next_item[0] if isinstance(next_item, tuple) else next_item
                    if contains_meaningful_text(next_text):
                        prefetch = (
                            _tts_executor.submit(_generate_audio, client, cfg, next_text),
                            next_item,
                        )
                        break

                # Phase 2: deferred prefetch — if queue was empty, watch for the
                # next item to arrive *during* current playback so its generation
                # overlaps with the audio playing instead of blocking after it.
                # A cancel event prevents the watcher from stealing items on interrupt.
                _deferred_slot: list = [None]
                _deferred_event = threading.Event()
                _watch_cancel = threading.Event()
                _watcher_active = False
                if prefetch is None:
                    _play_dur = len(audio_bytes) / (sample_rate * 2)
                    _cli, _cfg_snap = client, cfg
                    def _watch_deferred(
                        _slot=_deferred_slot, _ev=_deferred_event,
                        _cancel=_watch_cancel,
                        _cli=_cli, _cfg=_cfg_snap, _dur=_play_dur,
                    ) -> None:
                        deadline = time.time() + max(0.1, _dur - 0.3)
                        while time.time() < deadline:
                            if _cancel.is_set():
                                break
                            try:
                                item = _text_queue.get(timeout=0.1)
                            except queue.Empty:
                                continue
                            _txt = item[0] if isinstance(item, tuple) else item
                            if contains_meaningful_text(_txt):
                                _slot[0] = (
                                    _tts_executor.submit(_generate_audio, _cli, _cfg, _txt),
                                    item,
                                )
                            _ev.set()
                            return
                        _ev.set()
                    threading.Thread(
                        target=_watch_deferred, daemon=True, name="GeminiTTSPrefetchWatch"
                    ).start()
                    _watcher_active = True

                interrupted = _play_pcm(audio_bytes, sample_rate)
                prev_play_end = time.time()
                if interrupted:
                    _watch_cancel.set()  # abort watcher before it can steal post-interrupt items
                if _watcher_active:
                    _deferred_event.wait(timeout=0.15)
                if prefetch is None and not interrupted and _deferred_slot[0] is not None:
                    prefetch = _deferred_slot[0]
                    logging.info("[GeminiTTS] Deferred prefetch acquired — generation overlapping next play")
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

        if interrupted:
            # Discard any in-flight prefetch — its text is now stale.
            prefetch = None
        else:
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
    _text_queue.put((text, ref_ts, start_t, priority))

def interrupt_tts() -> None:
    """Signal interrupt: gracefully stops the current ffplay process (see
    _soft_stop_proc) and clears the pending text queue."""
    clear_text_queue()
    with _current_proc_lock:
        proc = _current_proc
    if proc:
        _soft_stop_proc(proc)
    _interrupt_event.set()
