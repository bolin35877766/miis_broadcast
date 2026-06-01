# src/miis_broadcast/core/models/openai_tts.py

import os
import time
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
from dotenv import load_dotenv
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
# 📊 TTS 效能統計
# ==========================================
_perf_stats = {
    "tts_latencies": [],
    "e2e_latencies": [],
    "last_text_sent_ts": 0.0,
    "current_ref_ts": 0.0,
    "current_start_t": 0.0,    # video timestamp of segment being spoken
}

def _log_tts_latency(value: float) -> None:
    _perf_stats["tts_latencies"].append(value)

def print_tts_stats() -> None:
    """在系統收尾時印出 TTS 與 E2E 延遲統計"""
    print("\n" + "=" * 40)
    print("Latency Performance Report")
    print("=" * 40)

    # 1. 純 TTS 延遲 (API 反應速度)
    if _perf_stats["tts_latencies"]:
        avg_tts = sum(_perf_stats["tts_latencies"]) / len(_perf_stats["tts_latencies"])
        print(f"Average TTS Latency (Text->Audio):    {avg_tts:.3f} s")
    else:
        print("Average TTS Latency:                   N/A")

    # 2. End-to-End 延遲 (視覺生成開始 -> 聽到聲音)
    if _perf_stats["e2e_latencies"]:
        avg_e2e = sum(_perf_stats["e2e_latencies"]) / len(_perf_stats["e2e_latencies"])
        print(f"Average E2E Latency (Vision->Audio):   {avg_e2e:.3f} s")
    else:
        print("Average E2E Latency:                   N/A")

    print("=" * 40 + "\n")
    
    # 清空數據
    _perf_stats["tts_latencies"].clear()
    _perf_stats["e2e_latencies"].clear()
    _perf_stats["last_text_sent_ts"] = 0.0
    _perf_stats["current_ref_ts"] = 0.0
    _perf_stats["current_start_t"] = 0.0


# ==========================================
# 🔁 Queue / Thread 控制
# ==========================================
_text_queue: "queue.Queue[tuple[str, float]]" = queue.Queue() 
_audio_output_queue: "queue.Queue[np.ndarray]" = queue.Queue()
_stop_event = threading.Event()
_tts_threads_started = False
_interrupt_event = threading.Event()

# ==========================================
# 🎛️ Runtime TTS Settings
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
            _interrupt_event.set() # 換聲音時還是要中斷目前說話
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

def interrupt_tts(clear_text: bool = True) -> None:
    """手動強制中斷 (例如按了 Stop 按鈕)"""
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
# 🧠 OpenAI Realtime 主 Worker
# ==========================================
async def _openai_realtime_worker():
    print("🎙️ [TTS Worker] 啟動連線...")

    while not _stop_event.is_set():
        is_response_active = False
        warmed_up = False
        doing_warmup = False
        # 新增：確保取消已被伺服器確認
        awaiting_cancel_ack = False 

        try:
            async with websockets.connect(TTS_MODEL_URL, additional_headers=TTS_HEADERS) as websocket:
                print("✅ [TTS Worker] 連線成功！")
                cfg = _get_tts_cfg_snapshot()

                # 1. Session Update
                await websocket.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "instructions": _build_instructions(cfg["speed"]),
                        "voice": cfg["voice"],
                        "speed": cfg["speed"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "temperature": float(_app_cfg.get("temperature", 0.7)),
                    },
                }))

                # 2. Warmup
                await websocket.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "warmup"}]},
                }))
                await websocket.send(json.dumps({"type": "response.create"}))
                is_response_active = True
                doing_warmup = True

                # 3. 主循環
                while not _stop_event.is_set():
                    
                    # (A) 檢查設定更新
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

                    # (B) 處理強制中斷
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
                            
                            # [修改] 從 Queue 取出 (text, ts)
                            while not _text_queue.empty():
                                item = _text_queue.get_nowait()
                                if isinstance(item, tuple):
                                    target_text = item[0]
                                    ref_ts = item[1] if len(item) > 1 else 0.0
                                    start_t = item[2] if len(item) > 2 else 0.0
                                else:
                                    target_text, ref_ts, start_t = item, 0.0, 0.0

                            if target_text and contains_meaningful_text(target_text):
                                # 如果目前有在說話，先發送取消並等待確認
                                if is_response_active:
                                    await websocket.send(json.dumps({"type": "response.cancel"}))
                                    awaiting_cancel_ack = True
                                    clear_audio_queue()  # 同步清空已緩衝的音訊，避免舊內容繼續播
                                    _text_queue.put((target_text, ref_ts, start_t))
                                    continue # 跳出本次循環，去聽事件 (D)

                                # 確定沒有 active response，才發送
                                _perf_stats["last_text_sent_ts"] = time.time()
                                _perf_stats["current_ref_ts"] = ref_ts
                                _perf_stats["current_start_t"] = start_t

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

                        if etype == "response.audio.delta":
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
                            # 如果錯誤是因為 active response，同步一下狀態
                            if "active response" in msg:
                                is_response_active = True
                            if err.get("code") != "response_cancel_not_active":
                                print(f"❌ [OpenAI Error] {msg}")

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break 

        except Exception as e:
            if _stop_event.is_set(): break
            print(f"❌ [TTS Error] {e}")
            await asyncio.sleep(2)

# ==========================================
# 🔊 播放 Worker
# ==========================================
def _audio_player_worker():
    if not shutil.which("ffplay"):
        return
    cmd = [
        "ffplay", "-f", "s16le", "-ar", "24000", "-ac", "1", "-nodisp",
        "-i", "pipe:0", "-loglevel", "quiet", "-fflags", "nobuffer",
        "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0",
    ]
    process = None
    
    while not _stop_event.is_set():
        try:
            audio_chunk = _audio_output_queue.get(timeout=0.1)
            
            # --- 計算 TTS Latency (API 反應時間) ---
            if _perf_stats["last_text_sent_ts"] > 0:
                latency = time.time() - _perf_stats["last_text_sent_ts"]
                _log_tts_latency(latency)
                _perf_stats["last_text_sent_ts"] = 0.0

            # --- 計算 E2E Latency (視覺+傳輸+語音) ---
            if _perf_stats["current_ref_ts"] > 0:
                e2e_latency = time.time() - _perf_stats["current_ref_ts"]
                _perf_stats["e2e_latencies"].append(e2e_latency)
                vid_t = _perf_stats["current_start_t"]
                _perf_stats["current_ref_ts"] = 0.0
                _perf_stats["current_start_t"] = 0.0
                avg_e2e = sum(_perf_stats["e2e_latencies"]) / len(_perf_stats["e2e_latencies"])
                logging.info(
                    "[延遲] 影片 %.1fs → 開始播報 +%.2fs  (平均 %.2fs, n=%d)",
                    vid_t, e2e_latency, avg_e2e, len(_perf_stats["e2e_latencies"]),
                )

            if process is None or process.poll() is not None:
                try:
                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                except: pass

            if process:
                try:
                    process.stdin.write(audio_chunk.tobytes())
                    process.stdin.flush()
                except: process = None
        except queue.Empty: continue
        except Exception: pass

    if process: process.terminate()


# ==========================================
# 🔧 對外 API
# ==========================================
def start_tts_system() -> None:
    global _tts_threads_started
    if _tts_threads_started: return
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
    print("🚀 [TTS] OpenAI TTS 背景服務已啟動")

def stop_tts_system() -> None:
    global _tts_threads_started
    _stop_event.set()
    _tts_threads_started = False

def enqueue_tts_text(text: str, ref_ts: float = 0.0, drop_outdated: bool = True, priority: int = 5, start_t: float = 0.0) -> None:
    if contains_meaningful_text(text):
        if drop_outdated:
            clear_text_queue()

        ts = ref_ts if ref_ts > 0 else time.time()
        _text_queue.put((text, ts, start_t))

        if _DRY_RUN:
            preview = text[:60] + ("…" if len(text) > 60 else "")
            _tts_logger.info(f'📥 [ENQUEUE P{priority}] "{preview}"')