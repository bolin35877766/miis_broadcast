# src/miis_broadcast/workers/dual_source.py

import time
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore


class DualSourceCameraThread(QtCore.QThread):
    """
    QThread that synchronizes two video sources simultaneously:
      - Webcam  (physical, cam_idx=0)
      - VR feed via OBS Virtual Camera (vr_idx=5)

    Synchronization strategy: grab() + retrieve()
      1. cap.grab()     -> hardware-level latch, no decode  (~0.05 ms per call)
      2. cap.retrieve() -> JPEG/YUV decode AFTER both sources have been latched

    Both grabs are called back-to-back, keeping Δt_grab << 1 ms (typical 0.1-0.3 ms).
    This is the OpenCV-recommended approach for multi-camera temporal alignment.

    Combined frame layout (side-by-side):
      [Webcam 640x480] | [VR/OBS 640x480]  ->  1280 x 480  RGB

    Signal interface is identical to OBSCameraThread so it plugs directly into
    the existing on_camera_frame -> cam_worker (LiveCC) pipeline without any
    changes to the inference logic.
    """

    signal_frame = QtCore.Signal(np.ndarray)   # RGB ndarray (1280x480)
    signal_error = QtCore.Signal(str)

    def __init__(
        self,
        cam_idx: int = 0,
        vr_idx: int = 5,
        target_fps: float = 30.0,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.vr_idx = vr_idx
        self.target_fps = target_fps
        self._stop_requested = False

    @staticmethod
    def _open_capture(idx: int, fps: float) -> Optional[cv2.VideoCapture]:
        """Try MSMF -> DSHOW -> default backend in order.
        Returns the first successfully opened VideoCapture, or None."""
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None

    def run(self) -> None:
        # ── Open Webcam (physical) ──
        cap_cam = self._open_capture(self.cam_idx, self.target_fps)
        if cap_cam is None:
            self.signal_error.emit(
                f"DualSource: Cannot open Webcam (idx={self.cam_idx}). "
                f"Check that the camera is connected and not in use."
            )
            return

        # Staggered start prevents USB hardware lock conflict on Windows
        time.sleep(0.5)

        # ── Open VR / OBS Virtual Camera ──
        cap_vr = self._open_capture(self.vr_idx, self.target_fps)
        if cap_vr is None:
            cap_cam.release()
            self.signal_error.emit(
                f"DualSource: Cannot open VR source (idx={self.vr_idx}). "
                f"Please make sure OBS is running and 'Start Virtual Camera' is active."
            )
            return

        # ── One-line perf stats: print to terminal only (no log file) ──
        frame_delay = 1.0 / self.target_fps

        print(
            f"\n{'='*60}\n"
            f"DualSource Live Session  |  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"CAM idx={self.cam_idx}  |  VR idx={self.vr_idx}  |  "
            f"Target: {self.target_fps:.0f} FPS\n"
            f"{'='*60}\n"
            f"[Timestamp]  TotalFPS | CamFPS | VrFPS | Dec | Proc | Gap | (reason)\n"
        )

        # FPS counter state
        fps_count = 0
        cam_fps_count = 0
        vr_fps_count = 0
        fps_window_start = time.perf_counter()
        last_delta_ms = 0.0

        while not self._stop_requested:
            t_loop = time.perf_counter()

            # ── Stage 1: grab() ──
            ok1 = cap_cam.grab()
            t_after_cam = time.perf_counter()
            
            ok2 = cap_vr.grab()
            t_after_vr = time.perf_counter()

            if ok1: cam_fps_count += 1
            if ok2: vr_fps_count += 1

            last_delta_ms = (t_after_vr - t_after_cam) * 1000.0 

            # ── Stage 2: retrieve() ──
            t_ret_start = time.perf_counter()
            ret1, frame_cam = cap_cam.retrieve()
            ret2, frame_vr  = cap_vr.retrieve()
            t_ret_end = time.perf_counter()

            if not ret1 or not ret2:
                continue

            # ── Composite / Processing ──
            t_proc_start = time.perf_counter()
            if frame_cam.shape[0] != frame_vr.shape[0]:
                frame_vr = cv2.resize(frame_vr, (frame_cam.shape[1], frame_cam.shape[0]))

            combined_bgr = np.hstack((frame_cam, frame_vr))
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            t_proc_end = time.perf_counter()

            self.signal_frame.emit(combined_rgb)

            # ── Stats: once per second, print to stdout ──
            fps_count += 1
            elapsed = time.perf_counter() - fps_window_start
            if elapsed >= 1.0:
                curr_total_fps = fps_count / elapsed
                curr_cam_fps = cam_fps_count / elapsed
                curr_vr_fps = vr_fps_count / elapsed
                
                ret_latency = (t_ret_end - t_ret_start) * 1000.0  # decode
                proc_latency = (t_proc_end - t_proc_start) * 1000.0  # hstack + color
                
                # Heuristic bottleneck label
                reason = "Normal"
                if curr_total_fps < self.target_fps * 0.8:
                    if ret_latency > 25: reason = f"Heavy-Decode({ret_latency:.1f}ms)"
                    elif proc_latency > 10: reason = f"Heavy-Proc({proc_latency:.1f}ms)"
                    elif last_delta_ms > 2: reason = "Bus-Congestion"
                    else: reason = "System-Lag"

                ts = time.strftime("%H:%M:%S")
                print(
                    f"[{ts}] TotalFPS:{curr_total_fps:5.2f} | Cam:{curr_cam_fps:5.2f} | Vr:{curr_vr_fps:5.2f} | "
                    f"Dec:{ret_latency:4.1f}ms | Proc:{proc_latency:4.1f}ms | Gap:{last_delta_ms:6.3f}ms | ({reason})"
                )
                
                # Reset counters
                fps_count = 0
                cam_fps_count = 0
                vr_fps_count = 0
                fps_window_start = time.perf_counter()

            # ── Pace loop to target FPS ──
            loop_elapsed = time.perf_counter() - t_loop
            sleep_t = frame_delay - loop_elapsed
            if sleep_t > 0.001:
                time.sleep(sleep_t)

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}]  DualSource session ended.")

        cap_cam.release()
        cap_vr.release()

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._stop_requested = True
