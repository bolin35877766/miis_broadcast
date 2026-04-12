from __future__ import annotations
from datetime import datetime
from typing import Optional
from PySide6 import QtCore
import logging

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Any, Tuple

import numpy as np
import time
from ..core.models.livecc_transformers import LiveCCInfer, print_final_stats
from ..core.models.openai_tts import print_tts_stats


class LiveCCWorker(QtCore.QObject):
    # 模型載入完成
    signal_model_loaded = QtCore.Signal()
    # 推論過程中的每一段字幕 (start_t, stop_t, text)
    signal_segment = QtCore.Signal(float, float, str)
    # 整段影片結束
    signal_finished = QtCore.Signal()
    # 出錯
    signal_error = QtCore.Signal(str)

    def __init__(self, device_id: int = 0, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.device_id = device_id
        self.livecc: Optional[LiveCCInfer] = None
        self._stop_requested = False

    @QtCore.Slot()
    def loadModel(self) -> None:
        """在 QThread 裡呼叫，用來載 LiveCC 模型。"""
        try:
            logging.info("[LiveCCWorker] Loading LiveCCInfer (device_id=%s)...", self.device_id)
            t0 = datetime.now()

            self.livecc = LiveCCInfer(device_id=self.device_id)

            t1 = datetime.now()
            logging.info(
                "[LiveCCWorker] LiveCCInfer loaded. Took %.2f s",
                (t1 - t0).total_seconds(),
            )
            self.signal_model_loaded.emit()
        except Exception as e:
            logging.exception("[LiveCCWorker] Error while loading LiveCC model")
            self.signal_error.emit(str(e))

    @QtCore.Slot(str, str)
    def runInference(self, video_path: str, query: str) -> None:
        """在 QThread 裡跑整段 LiveCC streaming 推論。"""
        if self.livecc is None:
            self.signal_error.emit("LiveCC model not loaded yet.")
            return

        try:
            logging.info("[LiveCCWorker] Start LiveCC inference on: %s", video_path)
            self._stop_requested = False

            t0 = datetime.now()

            # ✅ 新增：推論開始前清空舊的快取（解決重複選擇同一影片的問題）
            self.livecc._cached_video_readers_with_hw.clear()
            logging.info("[LiveCCWorker] Cleared cached video readers")

            # 初始化 state
            state = self.livecc.init_state(video_path)

            # ✅ 新增：記錄推論開始時間
            start_wall_time = datetime.now()

            # 推論迴圈
            while True:
                if self._stop_requested:
                    logging.info("[LiveCCWorker] Stop requested, breaking inference loop")
                    break

                # ✅ 新增：計算當前推論時間（相對於推論開始時的秒數）
                now_wall_time = datetime.now()
                elapsed_sec = (now_wall_time - start_wall_time).total_seconds()
                state["video_timestamp"] = elapsed_sec

                # 推論一個 segment（或多個，或零個）
                segment_count = 0
                for (start_t, stop_t), response, state in self.livecc.live_cc(query, state):
                    segment_count += 1
                    # emit segment
                    self.signal_segment.emit(float(start_t), float(stop_t), response)

                # 如果影片結束或沒有新 segment，就跳出迴圈
                if state.get("video_end", False):
                    logging.info("[LiveCCWorker] Video ended")
                    break

                # 如果沒有新 segment，等一下再試（不要忙輪詢）
                if segment_count == 0:
                    QtCore.QThread.msleep(200)

            t1 = datetime.now()
            logging.info(
                "[LiveCCWorker] LiveCC inference finished. Took %.2f s",
                (t1 - t0).total_seconds(),
            )

            self.signal_finished.emit()

        except Exception as e:
            logging.exception("[LiveCCWorker] Error during inference")
            self.signal_error.emit(str(e))
        finally:
            print_final_stats()
            print_tts_stats()
            # ✅ 新增：推論結束後也清空快取
            if self.livecc is not None:
                self.livecc._cached_video_readers_with_hw.clear()
                logging.info("[LiveCCWorker] Cleared cached video readers after inference")

    @QtCore.Slot()
    def requestStop(self) -> None:
        """外部呼叫，請求 LiveCC 迴圈結束。"""
        self._stop_requested = True

        # ✅ 新增：停止時也清空快取
        if self.livecc is not None:
            self.livecc._cached_video_readers_with_hw.clear()
            logging.info("[LiveCCWorker] Cleared cached video readers on stop request")

@dataclass
class FrameItem:
    t: float           # 收到 frame 的時間（秒）
    frame: np.ndarray  # BGR frame（cv2.VideoCapture 出來的）

from ..core.models.livecc_transformers import VideoClip
def build_clip_from_buffer(
    buffer: Deque[FrameItem],
    window_sec: float,
    target_fps: float,
) -> Optional["VideoClip"]:
    """
    從 buffer 中擷取最近 window_sec 秒的畫面，組成一個 VideoClip。
    - buffer 裡存的已經是相對時間了，直接用就好
    """
    from ..core.models.livecc_transformers import VideoClip

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

    raw_fps = len(frames_in_window) / max(duration, 1e-6)

    if target_fps < raw_fps:
        step = int(round(raw_fps / target_fps))
        step = max(step, 1)
        frames_sampled = frames_in_window[::step]
    else:
        frames_sampled = frames_in_window

    frames_rgb = []
    for fi in frames_sampled:
        try:
            rgb = fi.frame[..., ::-1]  # BGR -> RGB
            frames_rgb.append(rgb)
        except Exception as e:
            logging.exception(f"[LiveCCCameraWorker] BGR->RGB 失敗: {e}")
            return None

    try:
        frames_np = np.stack(frames_rgb, axis=0).astype(np.uint8)
    except Exception as e:
        logging.exception(f"[LiveCCCameraWorker] numpy.stack 失敗: {e}")
        return None

    # ✅ 簡化：直接用 frames_sampled[0].t，因為已經是相對時間了
    clip = VideoClip(
        frames=frames_np,
        fps=float(target_fps),
        t_start=frames_sampled[0].t,  # 已經是相對於開啟鏡頭時的秒數
    )
    return clip


class LiveCCCameraWorker(QtCore.QObject):
    """
    鏡頭版的 LiveCC Worker：
    - 接收 CameraWorker 丟進來的 frame (push_frame)
    - 定期從 buffer 組成一段 VideoClip
    - 呼叫 LiveCCInfer.live_cc_from_frames 做推論
    - 把結果用 signal_segment 丟回 GUI
    """

    signal_model_loaded = QtCore.Signal()
    signal_segment = QtCore.Signal(float, float, str)
    signal_finished = QtCore.Signal()
    signal_error = QtCore.Signal(str)

    def __init__(
        self,
        device_id: int = 0,
        window_sec: float = 2.0,
        target_fps: float = 2.0,
        infer_interval: float = 2.0,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.device_id = device_id
        self.window_sec = window_sec
        self.target_fps = target_fps
        self.infer_interval = infer_interval

        self.livecc: Optional[LiveCCInfer] = None
        self._stop_requested = False

        self._buffer: Deque[FrameItem] = deque(maxlen=180)
        self._state: Dict[str, Any] = {}
        self._query: str = "請描述畫面"
        self._inference_start_time = 0.0  # ✅ 新增：記錄推論開始時間

    @QtCore.Slot()
    def loadModel(self) -> None:
        """載入 LiveCC 模型（共用同一個 LiveCCInfer，只是來源變成 frames）"""
        try:
            logging.info("[LiveCCCameraWorker] Loading LiveCC model (camera mode)...")
            t0 = datetime.now()
            self.livecc = LiveCCInfer(device_id=self.device_id)
            t1 = datetime.now()
            logging.info(f"[LiveCCCameraWorker] Model loaded in {(t1 - t0).total_seconds():.2f} seconds")
            self.signal_model_loaded.emit()
        except Exception as e:
            logging.exception("[LiveCCCameraWorker] Failed to load LiveCC (camera mode)")
            self.signal_error.emit(str(e))

    @QtCore.Slot(np.ndarray, float)
    def push_frame(self, frame: np.ndarray, t: float) -> None:
        """
        由 CameraWorker 呼叫，把 BGR frame 丟進 buffer。
        """
        if self._stop_requested:
            return
        self._buffer.append(FrameItem(t=t, frame=frame.copy()))

    @QtCore.Slot(str)
    def runCameraInference(self, query: str) -> None:
        """
        主推論迴圈：
        - 維持在 QThread 中跑
        - 每 infer_interval 秒從 buffer 組 clip
        - 呼叫 live_cc_from_frames
        """
        if self.livecc is None:
            self.signal_error.emit("LiveCCCameraWorker: model not loaded")
            return

        self._query = query or "請描述畫面"
        self._stop_requested = False

        self._state = {}      # 初始清空 KV Cache 和 past_ids
        self._buffer.clear()  # 清空影像緩衝

        print(f"[LiveCCCameraWorker] 🚀 runCameraInference started | query='{self._query[:30]}...'")

        # ✅ 新增：初始化計數器
        inference_count = 0

        last_infer_t = time.time()

        logging.info("[LiveCCCameraWorker] Start camera inference loop")
        try:
            while not self._stop_requested:
                now = time.time()
                elapsed = now - last_infer_t
                
                if elapsed < self.infer_interval:
                    QtCore.QThread.msleep(100)
                    continue

                clip = build_clip_from_buffer(
                    self._buffer, 
                    self.window_sec, 
                    self.target_fps
                )
                if clip is None:
                    print(f"[LiveCCCameraWorker] ⏳ Buffer too thin (size={len(self._buffer)}) — waiting")
                    QtCore.QThread.msleep(100)
                    continue

                print(f"[LiveCCCameraWorker] 🎬 Clip built ({len(clip.frames)} frames) — running VLM")

                last_infer_t = now

                # ✅【關鍵修改】實作「短暫記憶」機制
                # 每推論 5 次後，清空一次記憶 (State Reset)
                # 假設 infer_interval=1.0s，代表每 5 秒會重置一次記憶。
                # 既能保留短期動作連貫性，又能斬斷無限跳針的迴圈。
                inference_count += 1
                if inference_count % 5 == 0:
                    self._state = {}
                    logging.info(f"[LiveCCCameraWorker] Memory Reset (Count={inference_count})")

                try:
                    for (start_ts, stop_ts), text, self._state in self.livecc.live_cc_from_frames(
                        clip=clip,
                        query=self._query,
                        state=self._state,
                    ):
                        self.signal_segment.emit(float(start_ts), float(stop_ts), text)
                except Exception as e:
                    logging.exception("[LiveCCCameraWorker] Error during camera inference")
                    self.signal_error.emit(str(e))
                    break
        finally:
            logging.info("[LiveCCCameraWorker] Camera inference loop finished")
            self.signal_finished.emit()

    @QtCore.Slot()
    def requestStop(self) -> None:
        """外部呼叫，結束鏡頭推論迴圈"""
        self._stop_requested = True


