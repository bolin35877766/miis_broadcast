# src/camera_livecc.py

import time
import sys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch

# 引用路徑請依照你的實際專案結構
from miis_broadcast.core.models.livecc_transformers_camera import LiveCCInfer, VideoClip

@dataclass
class FrameItem:
    t: float          
    frame: np.ndarray 

def build_clip_from_buffer(buffer: deque, window_sec: float, target_fps: float):
    if not buffer:
        return None

    t_now = buffer[-1].t
    t_start = max(t_now - window_sec, buffer[0].t)
    frames_in_window = [fi for fi in buffer if fi.t >= t_start]
    
    if len(frames_in_window) < 2:
        return None

    duration = frames_in_window[-1].t - frames_in_window[0].t
    if duration <= 0:
        return None

    raw_fps = len(frames_in_window) / duration

    if target_fps < raw_fps:
        step = int(round(raw_fps / target_fps))
        step = max(step, 1)
        frames_sampled = frames_in_window[::step]
    else:
        frames_sampled = frames_in_window

    # 轉 RGB 並檢查記憶體連續性 (Contiguous)
    frames_rgb = []
    for fi in frames_sampled:
        # 這裡可能會崩潰，加個 try
        try:
            rgb = cv2.cvtColor(fi.frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)
        except Exception as e:
            print(f"❌ [DEBUG] OpenCV cvtColor failed: {e}")
            return None
    
    try:
        frames_np = np.stack(frames_rgb, axis=0)
        # 強制轉為 contiguous array，避免 PyTorch 轉換時崩潰
        frames_np = np.ascontiguousarray(frames_np)
    except Exception as e:
        print(f"❌ [DEBUG] Numpy stack failed: {e}")
        return None
        
    clip_obj = VideoClip(
        frames=frames_np,
        fps=target_fps,
        t_start=frames_sampled[0].t
    )
    return clip_obj

def main():
    print("🚀 [DEBUG] 初始化 OpenCV...")
    # 強制使用 MJPG 格式，WSL 相容性較佳
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return
    print("✅ [DEBUG] 攝影機開啟成功")

    print("🚀 [DEBUG] 初始化 LiveCC 模型...")
    # 這裡不要指定 torch_dtype，照你的要求保留原本設定
    infer = LiveCCInfer(device_id=0)
    print("✅ [DEBUG] LiveCC 模型物件建立完成")

    buffer = deque(maxlen=90)
    t0 = time.monotonic()
    last_infer_t = 0.0
    state = {}

    WINDOW_SEC = 1.0       
    TARGET_FPS = 5.0       
    INFER_INTERVAL = 0.5   

    print("▶ 開始鏡頭 + LiveCC 測試 (Debug Mode)")
    print("⚠️  注意：已關閉 cv2.imshow 以排除視窗系統導致的 Segfault")

    frame_count = 0

    try:
        while True:
            # 1. 讀取畫面
            ret, frame = cap.read()
            if not ret:
                print("⚠ 讀不到 frame，結束")
                break
            
            frame_count += 1
            # if frame_count % 30 == 0:
            #     print(f"👀 [DEBUG] 正在讀取第 {frame_count} 幀，目前 buffer size: {len(buffer)}")

            now = time.monotonic() - t0
            buffer.append(FrameItem(t=now, frame=frame))

            
            #cv2.imshow("Camera", frame)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
                #break
            

            if now - last_infer_t < INFER_INTERVAL:
                continue
            
            #print(f"⚡ [DEBUG] 觸發推論時間點: {now:.2f}s")
            
            # 2. 製作 Clip
            clip_obj = build_clip_from_buffer(buffer, WINDOW_SEC, TARGET_FPS)
            
            if clip_obj is None:
                print("🔸 [DEBUG] Buffer 不夠長或無法建立 Clip，跳過")
                continue
            
            #print(f"📦 [DEBUG] Clip 建立成功: Shape={clip_obj.frames.shape}, Type={clip_obj.frames.dtype}")
            
            last_infer_t = now

            # 3. 呼叫推論
            #print("🔄 [DEBUG] 進入 infer.live_cc_from_frames...")
            sys.stdout.flush() # 強制刷新 buffer 確保 log 印出來
            
            for (g0, g1), text, state in infer.live_cc_from_frames(
                clip=clip_obj,
                message="請描述畫面",
                state=state,
            ):
                print(f"💬 [RESULT] {text}")
                
            #print("✅ [DEBUG] infer.live_cc_from_frames 執行結束")

    except KeyboardInterrupt:
        print("\n👋 使用者中斷")
    except Exception as e:
        print(f"\n❌ [CRASH] 主迴圈發生 Python 層級錯誤: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 程式結束")

if __name__ == "__main__":
    main()