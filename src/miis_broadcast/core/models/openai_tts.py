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
from typing import Optional

import numpy as np
import websockets
from dotenv import load_dotenv

# ==========================================
# 🔐 讀取 API Key
# ==========================================
load_dotenv()
MY_API_KEY = os.getenv("OPENAI_API_KEY")
if not MY_API_KEY:
    raise RuntimeError("[OpenAI TTS] OPENAI_API_KEY not found in environment")

TTS_MODEL_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
TTS_HEADERS = {
    "Authorization": f"Bearer {MY_API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

SYSTEM_INSTRUCTIONS = """
你是一個專業的「翻譯」模型。當你收到輸入 (可能是英文，也可能是其他語言) 時，請你：

1. **先把輸入翻譯成通順的繁體中文**；
2. **拒絕翻譯腔 (No Translation-ese)**：不要逐字翻譯英文文法。請用台灣人直播、看比賽時習慣的口語。
3. **不要** 插入、補充、改寫、刪減任何內容 — 唸出的內容必須 **完全對應**翻譯後的文字。
4. **直接輸出，禁止廢話**：絕對禁止說「好的」、「我來翻譯」等，收到文字直接唸出播報內容。

語音風格指令 (instructions)：
- 口音：自然「台灣國語／台灣腔」，語調普通、不刻意外國腔；
- 語速：偏快、有節奏，但保持清楚可懂；
- 情緒／語氣：根據內容語意，呈現 **強烈、高起伏、帶張力／激昂** 的播報感 — 若原文有驚嘆、強烈語氣，請加強語調與情緒；若敘述／轉折，語調可稍微穩，但保留「主播感」；
- 音調／語氣：自然、不做作、不像讀稿；給人感覺像現場播報或報導。
"""

# ==========================================
# 📊 TTS 效能統計
# ==========================================
_perf_stats = {
    "tts_latencies": [],
    "last_text_sent_ts": 0.0,
}

def _log_tts_latency(value: float) -> None:
    _perf_stats["tts_latencies"].append(value)

def print_tts_stats() -> None:
    """在系統收尾時印出 TTS 延遲統計"""
    print("\n" + "=" * 40)
    print("OpenAI TTS Performance")
    print("=" * 40)

    if _perf_stats["tts_latencies"]:
        first_tts = _perf_stats["tts_latencies"][0]
        avg_tts = sum(_perf_stats["tts_latencies"]) / len(_perf_stats["tts_latencies"])
        print(f"First TTS latency (Text->Audio): {first_tts:.3f} s")
        print(f"Average TTS latency (Text->Audio): {avg_tts:.3f} s")
    else:
        print("no data for TTS latency")

    print("=" * 40 + "\n")
    _perf_stats["tts_latencies"].clear()
    _perf_stats["last_text_sent_ts"] = 0.0


# ==========================================
# 🔁 Queue / Thread 控制
# ==========================================
_text_queue: "queue.Queue[str]" = queue.Queue()
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
                        "temperature": 0.7,
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
                            while not _text_queue.empty():
                                target_text = _text_queue.get_nowait()
                            
                            if target_text and contains_meaningful_text(target_text):
                                # 如果目前有在說話，先發送取消並等待確認
                                if is_response_active:
                                    await websocket.send(json.dumps({"type": "response.cancel"}))
                                    awaiting_cancel_ack = True
                                    # 先把這句文字塞回去 Queue 的最前面，等 Cancel 成功後再來拿
                                    _text_queue.put(target_text) 
                                    continue # 跳出本次循環，去聽事件 (D)
                                
                                # 確定沒有 active response，才發送
                                _perf_stats["last_text_sent_ts"] = time.time()
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
            if _perf_stats["last_text_sent_ts"] > 0:
                latency = time.time() - _perf_stats["last_text_sent_ts"]
                _log_tts_latency(latency)
                _perf_stats["last_text_sent_ts"] = 0.0

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

def enqueue_tts_text(text: str, drop_outdated: bool = True) -> None:
    if contains_meaningful_text(text):
        if drop_outdated:
            # 1. 清空文字隊列
            clear_text_queue()
            
        _text_queue.put(text)