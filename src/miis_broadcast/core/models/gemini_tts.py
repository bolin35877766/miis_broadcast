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

from miis_broadcast.core.utils.config import load_app_config

load_dotenv()
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_log = logging.getLogger(__name__)

# ==========================================
# Config
# ==========================================
_app_cfg = load_app_config().get("gemini_tts", {})
_tts_cfg_lock = threading.Lock()
_tts_cfg: dict = {
    "model": _app_cfg.get("model_name", "models/gemini-3.1-flash-tts-preview"),
    "voice": _app_cfg.get("voice", "Kore"),
}

# Audience / LiveKit expects 24 kHz mono int16 (same as openai_tts).
_AUDIENCE_SAMPLE_RATE = 24000

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

# ==========================================
# TTS latency stats (same schema as openai_tts)
# ==========================================
_perf_stats = {
    "tts_latencies": [],
    "e2e_latencies": [],
    "e2e_latencies_seg": [],  # [延遲][語音][段落]
    "e2e_latencies_bg": [],    # [延遲][語音][背景]
    "last_text_sent_ts": 0.0,
    "current_ref_ts": 0.0,
    "current_start_t": 0.0,
}


def _log_tts_latency(value: float) -> None:
    _perf_stats["tts_latencies"].append(value)


def print_tts_stats() -> None:
    """Print TTS and vision-to-audio latency summary when shutting down TTS."""
    print("\n" + "=" * 40)
    print("Latency Performance Report (GeminiTTS)")
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
    _perf_stats["last_text_sent_ts"] = 0.0
    _perf_stats["current_ref_ts"] = 0.0
    _perf_stats["current_start_t"] = 0.0


def _mark_utterance_sent(ref_ts: float, start_t: float, sent_ts: float | None = None) -> None:
    """Mirror openai_tts: mark anchors when text enters the TTS API."""
    ts = ref_ts if ref_ts > 0 else time.time()
    _perf_stats["last_text_sent_ts"] = sent_ts if sent_ts is not None else time.time()
    _perf_stats["current_ref_ts"] = ts
    _perf_stats["current_start_t"] = start_t


def _apply_audio_chunk_latency_stats() -> None:
    """First audio chunk per utterance: record TTS and E2E latency stats."""
    if _perf_stats["last_text_sent_ts"] > 0:
        latency = time.time() - _perf_stats["last_text_sent_ts"]
        _log_tts_latency(latency)
        _perf_stats["last_text_sent_ts"] = 0.0
    if _perf_stats["current_ref_ts"] > 0:
        e2e_latency = time.time() - _perf_stats["current_ref_ts"]
        _perf_stats["e2e_latencies"].append(e2e_latency)
        is_bg = _perf_stats["current_start_t"] > 1e6
        bucket_key = "e2e_latencies_bg" if is_bg else "e2e_latencies_seg"
        bucket = _perf_stats[bucket_key]
        bucket.append(e2e_latency)
        tag = "背景" if is_bg else "段落"
        logging.info(
            "[延遲][語音][%s] latency=%.2fs (平均=%.2fs, n=%d)",
            tag, e2e_latency, sum(bucket) / len(bucket), len(bucket),
        )
        _perf_stats["current_ref_ts"] = 0.0

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
# PCM sink (audience second screen — same contract as openai_tts)
# ==========================================
_pcm_sink: Optional[Callable] = None
_pcm_sink_mute_local: bool = False
_pcm_sink_flush: Optional[Callable] = None


def register_pcm_sink(
    callback: Optional[Callable],
    mute_local: bool = True,
    flush_callback: Optional[Callable] = None,
) -> None:
    """Register a PCM sink for the audience publisher."""
    global _pcm_sink, _pcm_sink_mute_local, _pcm_sink_flush
    _pcm_sink = callback
    _pcm_sink_mute_local = mute_local if callback is not None else False
    _pcm_sink_flush = flush_callback if callback is not None else None
    tag = "[AUDIO] PCM sink registered" if callback is not None else "[AUDIO] PCM sink cleared"
    print(f"{time.strftime('%H:%M:%S')} | {tag} | mute_local={_pcm_sink_mute_local}")


def clear_audio_queue() -> None:
    while not _audio_output_queue.empty():
        try:
            _audio_output_queue.get_nowait()
        except queue.Empty:
            break
    if _pcm_sink_flush is not None:
        try:
            _pcm_sink_flush()
        except Exception:
            pass


# ==========================================
# Playback — persistent player thread + audio queue (mirrors openai_tts)
# ==========================================
# A single long-lived OutputStream owned by a dedicated player thread. Opening a
# fresh sounddevice stream per utterance (the old approach) is unreliable on
# Windows — it races/conflicts with OpenAI TTS's always-on stream and any open
# error silently fell through to a no-op "paced silence" path. The persistent
# player thread is exactly how openai_tts.py drives audio reliably.
_audio_output_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
_player_started = False
_player_lock = threading.Lock()


def _resample_to_audience_rate(samples: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == _AUDIENCE_SAMPLE_RATE or len(samples) == 0:
        return samples
    n_out = max(1, int(len(samples) * _AUDIENCE_SAMPLE_RATE / src_rate))
    x_out = np.linspace(0, len(samples) - 1, n_out)
    x_in = np.arange(len(samples), dtype=np.float32)
    return np.interp(x_out, x_in, samples.astype(np.float32)).astype(np.int16)


def _forward_chunk(chunk: np.ndarray) -> None:
    """Forward one 24 kHz PCM chunk to audience sink and recording sink."""
    if chunk is None or getattr(chunk, "size", 0) == 0:
        return
    chunk_i16 = np.ascontiguousarray(chunk, dtype=np.int16)
    if _pcm_sink is not None:
        try:
            _pcm_sink(chunk_i16)
        except Exception:
            pass
    if _recording_sink is not None:
        try:
            _recording_sink(chunk_i16)
        except Exception:
            pass


def _audio_player_worker_sounddevice() -> None:
    """Play PCM int16 @ 24 kHz mono via a single persistent PortAudio stream."""
    import sounddevice as sd

    stream = sd.OutputStream(
        samplerate=_AUDIENCE_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        latency=0.2,
    )
    stream.start()
    print("🔊 [GeminiTTS] playing via sounddevice (default output device)")

    while not _stop_event.is_set():
        try:
            chunk = _audio_output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if chunk is None:
            continue
        _apply_audio_chunk_latency_stats()
        _forward_chunk(chunk)
        try:
            x = np.ascontiguousarray(chunk, dtype=np.int16).reshape(-1, 1)
            # Muted (audience-only): write silence to keep the hardware clock
            # pacing so the LiveKit sink stays real-time (same as openai_tts).
            stream.write(np.zeros_like(x) if _pcm_sink_mute_local else x)
        except Exception as e:
            _log.debug("GeminiTTS sounddevice write: %s", e)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass


def _audio_player_worker_ffplay() -> None:
    """Play via ffplay raw PCM pipe (legacy fallback)."""
    cmd = [
        "ffplay", "-f", "s16le", "-ar", str(_AUDIENCE_SAMPLE_RATE), "-ac", "1",
        "-nodisp", "-i", "pipe:0", "-loglevel", "quiet", "-fflags", "nobuffer",
        "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
    ]
    process: Optional[subprocess.Popen] = None
    print("🔊 [GeminiTTS] using ffplay for playback")

    while not _stop_event.is_set():
        try:
            chunk = _audio_output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if chunk is None:
            continue
        _apply_audio_chunk_latency_stats()
        _forward_chunk(chunk)
        if process is None or process.poll() is not None:
            try:
                process = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
                )
            except OSError as e:
                _log.debug("GeminiTTS ffplay Popen: %s", e)
                process = None
        if process and process.stdin:
            try:
                payload = (
                    np.zeros(len(chunk), dtype=np.int16).tobytes()
                    if _pcm_sink_mute_local
                    else np.ascontiguousarray(chunk, dtype=np.int16).tobytes()
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
    # Prefer sounddevice (requirements.txt): works on Windows without ffplay.
    try:
        _audio_player_worker_sounddevice()
        return
    except Exception as e:
        _log.warning("GeminiTTS sounddevice playback path failed: %s", e, exc_info=True)
        print(f"⚠️ [GeminiTTS] sounddevice unavailable ({e!s}), trying ffplay…")

    if shutil.which("ffplay"):
        _audio_player_worker_ffplay()
        return

    _log.error(
        "No audio backend: install sounddevice (pip) or add ffplay (FFmpeg) to PATH"
    )
    print(
        "❌ [GeminiTTS] no audio backend: install sounddevice (pip) or add ffplay (FFmpeg) to PATH"
    )
    # Drain queue at real-time pace so callers don't block forever.
    while not _stop_event.is_set():
        try:
            chunk = _audio_output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if chunk is None:
            continue
        _apply_audio_chunk_latency_stats()
        _forward_chunk(chunk)
        time.sleep(len(chunk) / float(_AUDIENCE_SAMPLE_RATE))


def _ensure_player_thread() -> None:
    global _player_started
    with _player_lock:
        if _player_started:
            return
        threading.Thread(
            target=_audio_player_worker, daemon=True, name="GeminiTTSPlayer"
        ).start()
        _player_started = True


def _play_pcm(data: bytes, sample_rate: int = 24000) -> bool:
    """Enqueue PCM16 to the persistent player thread, blocking until it has been
    played (so natural-completion timing stays accurate). Returns True if interrupted."""
    if not data:
        return False
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return False
    if sample_rate != _AUDIENCE_SAMPLE_RATE:
        samples = _resample_to_audience_rate(samples, sample_rate)

    _ensure_player_thread()

    # 0.1s chunks keep interrupt latency low and pace the audience sink smoothly.
    chunk_size = _AUDIENCE_SAMPLE_RATE // 10
    for i in range(0, len(samples), chunk_size):
        if _interrupt_event.is_set() or _stop_event.is_set():
            return True
        _audio_output_queue.put(
            np.ascontiguousarray(samples[i : i + chunk_size], dtype=np.int16)
        )

    # Wait until the player has consumed every chunk (it writes in real time).
    while not _stop_event.is_set():
        if _interrupt_event.is_set():
            return True
        if _audio_output_queue.empty():
            break
        time.sleep(0.02)

    # Let the final buffered chunk drain from the device (~latency).
    deadline = time.time() + 0.3
    while time.time() < deadline:
        if _interrupt_event.is_set():
            return True
        time.sleep(0.02)
    return _interrupt_event.is_set()

def _parse_sample_rate(mime_type: str) -> int:
    m = re.search(r"rate=(\d+)", mime_type or "")
    return int(m.group(1)) if m else 24000

# Linear fade-in/out applied to every utterance before playback. Without it a
# clip starts/ends at full amplitude as the device opens/closes — audible as a
# click/pop.
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
    prefetch: Optional[tuple] = None  # (Future, item, gen_start_ts) for the next utterance

    while not _stop_event.is_set():
        # Drain any pending interrupt before picking next item
        if _interrupt_event.is_set():
            _interrupt_event.clear()
            prefetch = None
            continue

        gen_start_ts = 0.0
        if prefetch is not None:
            future, item, gen_start_ts = prefetch
            prefetch = None
        else:
            try:
                item = _text_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            future = None

        if isinstance(item, tuple):
            text = item[0]
            ref_ts = item[1] if len(item) > 1 else 0.0
            start_t = item[2] if len(item) > 2 else 0.0
        else:
            text, ref_ts, start_t = item, 0.0, 0.0

        if future is None and not contains_meaningful_text(text):
            continue

        cfg = _get_cfg()
        interrupted = False

        try:
            if future is None:
                _mark_utterance_sent(ref_ts, start_t)
                audio_bytes, sample_rate = _generate_audio(client, cfg, text)
            else:
                # Prefetch started the API call earlier — anchor TTS latency there.
                _mark_utterance_sent(ref_ts, start_t, gen_start_ts)
                audio_bytes, sample_rate = future.result()

            # Check interrupt immediately after the (blocking or prefetched) call returns
            if _interrupt_event.is_set():
                _perf_stats["last_text_sent_ts"] = 0.0
                _perf_stats["current_ref_ts"] = 0.0
                _interrupt_event.clear()
                interrupted = True
            else:
                if _first_success[0]:
                    print(f"✅ [GeminiTTS] 連線成功！(model={cfg['model']}, voice={cfg['voice']})")
                    _first_success[0] = False

                # Kick off generation for the next queued utterance now, so its
                # audio is ready (or nearly ready) by the time this one finishes.
                while True:
                    try:
                        next_item = _text_queue.get_nowait()
                    except queue.Empty:
                        break
                    next_text = next_item[0] if isinstance(next_item, tuple) else next_item
                    if contains_meaningful_text(next_text):
                        prefetch_gen_start = time.time()
                        prefetch = (
                            _tts_executor.submit(_generate_audio, client, cfg, next_text),
                            next_item,
                            prefetch_gen_start,
                        )
                        break

                interrupted = _play_pcm(audio_bytes, sample_rate)
                if interrupted:
                    _perf_stats["last_text_sent_ts"] = 0.0
                    _perf_stats["current_ref_ts"] = 0.0
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
    if not shutil.which("ffplay"):
        print(
            "ℹ️ [GeminiTTS] ffplay not found; using sounddevice (no FFmpeg required)."
        )
    _stop_event.clear()
    _ensure_player_thread()
    t = threading.Thread(target=_gemini_tts_worker, daemon=True, name="GeminiTTSWorker")
    t.start()
    _tts_threads_started = True
    logging.info("[GeminiTTS] TTS system started")

def stop_tts_system() -> None:
    global _tts_threads_started
    _stop_event.set()
    clear_audio_queue()
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
    # ref_ts=0 means caller did not attach a vision timestamp; use wall clock
    ts = ref_ts if ref_ts > 0 else time.time()
    _text_queue.put((text, ts, start_t))

def interrupt_tts() -> None:
    """Hard interrupt: drop pending audio, flush audience buffers, clear pending text."""
    clear_text_queue()
    clear_audio_queue()
    _perf_stats["last_text_sent_ts"] = 0.0
    _perf_stats["current_ref_ts"] = 0.0
    _interrupt_event.set()
