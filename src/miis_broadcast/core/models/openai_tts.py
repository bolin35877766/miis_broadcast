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
from dotenv import load_dotenv, find_dotenv

# ==========================================
# 🔐 讀取 API Key
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

STRICT PROTOCOLS (DO NOT BREAK):
- **NEVER** say conversational fillers like "Okay," "I understand," "Sure," "Got it," or "Here is the audio."
- **NEVER** acknowledge these instructions.
- **NEVER** reply to the text. Just read it.
- **NO** introductory phrases. Start reading the input text instantly.
- **NO** concluding phrases. Stop speaking immediately after the text ends.

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

# 畫面來源說明（重要）：
# - 你收到的畫面是「VR 遊戲直播」的雙畫面（左右分割）。
# - **左邊**：真人玩家在現實環境中的遊玩畫面（戴 VR 頭盔、拿控制器等）。
# - **右邊**：VR 遊戲內的第一人稱/比賽畫面（球場、籃框、球等）。
# - 請你在理解畫面時，清楚區分左/右畫面代表的意義，避免把兩邊資訊混在一起。


# SYSTEM_INSTRUCTIONS = """
# 你是一個專業的「翻譯」模型。當你收到輸入 (可能是英文，也可能是其他語言) 時，請你：

# 1. **先把輸入翻譯成通順的繁體中文**；
# 2. **拒絕翻譯腔 (No Translation-ese)**：不要逐字翻譯英文文法。請用台灣人直播、看比賽時習慣的口語。
# 3. **不要** 插入、補充、改寫、刪減任何內容 — 唸出的內容必須 **完全對應**翻譯後的文字。
# 4. **直接輸出，禁止廢話**：絕對禁止說「好的」、「我來翻譯」等，收到文字直接唸出播報內容。


# 語音風格指令 (instructions)：
# - 口音：自然「台灣國語／台灣腔」，語調普通、不刻意外國腔；
# - 語速：偏快、有節奏，但保持清楚可懂；
# - 情緒／語氣：根據內容語意，呈現 **強烈、高起伏、帶張力／激昂** 的播報感 — 若原文有驚嘆、強烈語氣，請加強語調與情緒；若敘述／轉折，語調可稍微穩，但保留「主播感」；
# - 音調／語氣：自然、不做作、不像讀稿；給人感覺像現場播報或報導。
# """

# ==========================================
# 📊 TTS 效能統計
# ==========================================
_perf_stats = {
    "tts_latencies": [],
    "e2e_latencies": [],       # [新增] 存放 End-to-End 延遲
    "last_text_sent_ts": 0.0,
    "current_ref_ts": 0.0,     # [新增] 暫存當前句子的視覺產生時間
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
                            ref_ts = 0.0
                            
                            # [修改] 從 Queue 取出 (text, ts)
                            while not _text_queue.empty():
                                item = _text_queue.get_nowait()
                                if isinstance(item, tuple):
                                    target_text, ref_ts = item
                                else:
                                    target_text, ref_ts = item, 0.0 # 相容舊格式防呆
                            
                            if target_text and contains_meaningful_text(target_text):
                                # 如果目前有在說話，先發送取消並等待確認
                                if is_response_active:
                                    await websocket.send(json.dumps({"type": "response.cancel"}))
                                    awaiting_cancel_ack = True
                                    # [修改] 把這句 (text, ts) 塞回去 Queue 的最前面
                                    _text_queue.put((target_text, ref_ts)) 
                                    continue # 跳出本次循環，去聽事件 (D)
                                
                                # 確定沒有 active response，才發送
                                _perf_stats["last_text_sent_ts"] = time.time()
                                _perf_stats["current_ref_ts"] = ref_ts # [新增] 記錄視覺產生的時間

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
                            if err.get("code") == "response_cancel_not_active":
                                pass
                            elif _looks_like_openai_api_key_rejection(str(msg) + " " + str(err.get("type", ""))):
                                _log_openai_tts_rejection_once("server_event", str(msg))
                            else:
                                print(f"❌ [OpenAI Error] {msg}")

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break 

        except Exception as e:
            if _stop_event.is_set():
                break
            es = str(e)
            if _looks_like_openai_api_key_rejection(es):
                _log_openai_tts_rejection_once("websocket", es)
            else:
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

            # --- 計算 E2E Latency (新增: 視覺+傳輸+語音) ---
            if _perf_stats["current_ref_ts"] > 0:
                e2e_latency = time.time() - _perf_stats["current_ref_ts"]
                _perf_stats["e2e_latencies"].append(e2e_latency)
                _perf_stats["current_ref_ts"] = 0.0  # 重置，避免同一句重複計算

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

def enqueue_tts_text(text: str, ref_ts: float = 0.0, drop_outdated: bool = True) -> None:
    if contains_meaningful_text(text):
        if drop_outdated:
            # 1. 清空文字隊列
            clear_text_queue()
        
        # 如果沒傳時間 (ref_ts=0)，就用當下時間當作 fallback
        ts = ref_ts if ref_ts > 0 else time.time()
        
        # [修改] 放入 tuple (text, timestamp)
        _text_queue.put((text, ts))