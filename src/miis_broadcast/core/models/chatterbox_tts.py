# src/miis_broadcast/core/models/local_tts.py

import sys
import subprocess
import torch
import numpy as np
import time
import threading
import queue
import shutil
import os
import warnings
import gc
from typing import Optional

# 忽略警告
warnings.filterwarnings("ignore")


from chatterbox_streaming.src.chatterbox.tts import ChatterboxTTS


# ==========================================
# ⚙️ 設定參數
# ==========================================
AUDIO_PROMPT_PATH = "/home/miislab/livecc_gui/ref_voice/announcer_ref.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🔁 Queue / Thread 控制
# ==========================================
_text_queue: "queue.Queue[str]" = queue.Queue()
_audio_output_queue: "queue.Queue[np.ndarray]" = queue.Queue()
_stop_event = threading.Event()
_tts_threads_started = False
_interrupt_event = threading.Event()

_model_instance = None
_tts_cfg_lock = threading.Lock()
_tts_cfg = {
    "temperature": 0.7,
    "cfg_weight": 1,
    "chunk_size": 25, # 小 Chunk 避免 OOM
}

def set_local_tts_params(temperature: float, cfg_weight: float) -> None:
    with _tts_cfg_lock:
        _tts_cfg["temperature"] = max(0.1, min(2.0, temperature))
        _tts_cfg["cfg_weight"] = max(0.1, min(2.0, cfg_weight))

def get_local_tts_cfg() -> dict:
    with _tts_cfg_lock:
        return dict(_tts_cfg)

def clear_queues() -> None:
    while not _text_queue.empty():
        try: _text_queue.get_nowait()
        except queue.Empty: break
    while not _audio_output_queue.empty():
        try: _audio_output_queue.get_nowait()
        except queue.Empty: break

def interrupt_local_tts() -> None:
    _interrupt_event.set()
    clear_queues()
    print("⚡ [Local TTS] 觸發中斷")

def _clean_vram():
    """清理 VRAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# 🛠️ 模型優化 (混合精度)
# ==========================================
def _optimize_model(model_instance):
    """
    進行「混合精度」優化：GPT=FP16, S3Gen=FP32
    """
    print("🔧 [Local TTS] 正在進行混合精度優化 ...")
    
    # 1. GPT 轉 FP16
    fp16_targets = ['gpt', 'transformer', 'decoder', 'encoder', 'bert']
    for attr in dir(model_instance):
        if attr in fp16_targets:
            sub_module = getattr(model_instance, attr)
            if isinstance(sub_module, torch.nn.Module):
                try:
                    sub_module.half()
                except: pass

    # 2. 修補 Config (避免 attention output 崩潰)
    targets = [model_instance]
    for attr in ['model', 'decoder', 'encoder', 'gpt', 'transformer', 's3gen']:
        if hasattr(model_instance, attr):
            targets.append(getattr(model_instance, attr))

    for target in targets:
        if hasattr(target, 'config') and hasattr(target.config, 'output_attentions'):
            target.config.output_attentions = False
        if hasattr(target, 'generation_config') and hasattr(target.generation_config, 'output_attentions'):
            target.generation_config.output_attentions = False
        if hasattr(target, 'set_attn_implementation'):
            try: target.set_attn_implementation("eager")
            except: pass

    print("✅ [Local TTS] 模型優化完成")

# ==========================================
# 🧠 Local TTS 生成 Worker
# ==========================================
def _local_tts_generator_worker():
    global _model_instance
    print("🎙️ [Local TTS Worker] 啟動...")

    if not os.path.exists(AUDIO_PROMPT_PATH):
        print(f"❌ [Local TTS] 找不到參考音檔: {AUDIO_PROMPT_PATH}")
        return

    # 1. 載入模型
    if _model_instance is None:
        try:
            _clean_vram()
            print(f"⏳ [Local TTS] 正在載入模型...")
            if ChatterboxTTS is not None:
                _model_instance = ChatterboxTTS.from_pretrained(device=DEVICE)
                _optimize_model(_model_instance)

                print("🔥 [Local TTS] Warmup...")
                try:
                    with torch.inference_mode():
                        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                            warmup_gen = _model_instance.generate_stream(
                                text="Ready.",
                                audio_prompt_path=AUDIO_PROMPT_PATH,
                                chunk_size=15, 
                                temperature=0.7,
                                cfg_weight=0.7
                            )
                            for _ in warmup_gen: pass
                    print("✅ [Local TTS] 模型就緒！")
                except Exception as e:
                    print(f"⚠️ [Local TTS] Warmup 警告: {e}")
            else:
                return
        except Exception as e:
            print(f"❌ [Local TTS] 載入失敗: {e}")
            return

    while not _stop_event.is_set():
        try:
            # 1. 拿下一句
            text = _text_queue.get(timeout=0.5)

            # 2. ✅ 去積壓機制：若有更新的，跳過舊的
            if not _text_queue.empty():
                skipped_count = 0
                print("⏩ [Local TTS] 偵測到堆積，正在跳過舊訊息...")
                while not _text_queue.empty():
                    try:
                        text = _text_queue.get_nowait()
                        skipped_count += 1
                    except queue.Empty: break
                print(f"⏩ [Local TTS] 已跳過 {skipped_count} 則舊訊息，直接播放: '{text}'")

            _interrupt_event.clear()
            if not text or not text.strip(): continue

            print(f"   [Local TTS] Generating: '{text}' ...")
            cfg = get_local_tts_cfg()
            _clean_vram()

            try:
                # 執行推論 (Mixed Precision handled by PyTorch automatic type promotion)
                with torch.inference_mode():
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        generator = _model_instance.generate_stream(
                            text=text,
                            audio_prompt_path=AUDIO_PROMPT_PATH,
                            chunk_size=cfg["chunk_size"],
                            temperature=cfg["temperature"],
                            cfg_weight=cfg["cfg_weight"]
                        )

                        for audio_chunk, _ in generator:
                            if _stop_event.is_set() or _interrupt_event.is_set():
                                break
                            
                            if audio_chunk is None: continue

                            audio_np = audio_chunk.squeeze().float().detach().cpu().numpy()
                            max_val = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
                            if max_val > 1.0: audio_np = audio_np / max_val
                            audio_i16 = (audio_np * 32767.0).astype(np.int16)

                            _audio_output_queue.put(audio_i16)

                _clean_vram()

            except Exception as e:
                print(f"❌ [Local TTS] 生成錯誤: {e}")
                _clean_vram()

        except queue.Empty: continue
        except Exception as e:
            print(f"❌ [Local TTS Worker Error] {e}")

# ==========================================
# 🔊 播放 Worker
# ==========================================
def _local_audio_player_worker():
    if not shutil.which("ffplay"):
        return
    cmd = [
        "ffplay", "-f", "s16le", "-ar", "24000", "-ac", "1",
        "-nodisp", "-i", "pipe:0", "-loglevel", "quiet",
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-probesize", "32", "-analyzeduration", "0", "-autoexit"
    ]
    process = None
    
    while not _stop_event.is_set():
        try:
            if process is None or process.poll() is not None:
                try:
                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                except: time.sleep(1); continue

            try:
                audio_chunk = _audio_output_queue.get(timeout=0.1)
                process.stdin.write(audio_chunk.tobytes())
            except IOError: process = None
            except queue.Empty: continue
            except Exception: process = None
        except: pass

    if process: process.terminate()

# ==========================================
# 🔧 對外 API
# ==========================================
def start_local_tts_system() -> None:
    global _tts_threads_started
    if _tts_threads_started: return
    _stop_event.clear()
    _interrupt_event.clear()
    threading.Thread(target=_local_tts_generator_worker, daemon=True).start()
    threading.Thread(target=_local_audio_player_worker, daemon=True).start()
    _tts_threads_started = True
    print("🚀 [Local TTS] 背景服務已啟動")

def stop_local_tts_system() -> None:
    global _tts_threads_started
    _stop_event.set()
    _tts_threads_started = False

def enqueue_local_tts_text(text: str) -> None:
    if text and len(text.strip()) > 0:
        _text_queue.put(text)