# src/miis_broadcast/workers/free_switch.py
"""
FreeSwitchCameraThread
======================
Runs Webcam (physical) and OBS Virtual Camera simultaneously from startup.
Switching sources = changing one variable; no camera reconnection, no delay.

Source constants
----------------
SOURCE_WEBCAM : "webcam"   640×480 RGB on ``signal_frame``
SOURCE_VR     : "vr"       native VR (e.g. 1920×1080) on ``signal_vr_frame``;
               downscaled 640×480 on ``signal_frame`` for LiveCC
SOURCE_DUAL   : "dual"     1280×480 RGB on ``signal_frame`` (Webcam | VR side-by-side)

Switching protocol
------------------
Call set_active_source(SOURCE_*) from any thread.
The change takes effect within the next grab/retrieve cycle (~1 frame @ 30 fps).
"""

import threading
import time
import platform
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore

# ── Source identifiers ────────────────────────────────────────────────────────
SOURCE_WEBCAM = "webcam"
SOURCE_VR     = "vr"
SOURCE_DUAL   = "dual"

ALL_SOURCES = (SOURCE_WEBCAM, SOURCE_VR, SOURCE_DUAL)

# Resolution for each capture device.
# Webcam stays at SD for lightweight LiveCC inference.
# VR/OBS is requested at 1080p so the audience second screen gets native quality.
_WEBCAM_W, _WEBCAM_H = 640, 480
_VR_W,     _VR_H     = 1920, 1080

# LiveCC inference expects 640×480; resize VR frames before emitting signal_frame.
_LIVECC_W, _LIVECC_H = 640, 480


class FreeSwitchCameraThread(QtCore.QThread):
    """
    QThread that opens both Webcam and OBS Virtual Camera at startup and
    emits frames from whichever source is currently selected.

    Signals
    -------
    signal_frame(np.ndarray)  — RGB ndarray, shape depends on active source
    signal_error(str)         — fatal open / read error
    signal_source_changed(str) — emitted after set_active_source takes effect
    """

    signal_frame          = QtCore.Signal(np.ndarray)  # RGB ndarray — active source for first frontend
    signal_vr_frame       = QtCore.Signal(np.ndarray)  # RGB ndarray — always VR, for audience second screen
    signal_error          = QtCore.Signal(str)
    signal_source_changed = QtCore.Signal(str)

    DEFAULT_FPS = 30.0

    def __init__(
        self,
        initial_source: str = SOURCE_WEBCAM,
        cam_idx: int = 0,
        vr_idx: int = 5,
        target_fps: float = DEFAULT_FPS,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.cam_idx    = cam_idx
        self.vr_idx     = vr_idx
        self.target_fps = target_fps

        self._active_source   = initial_source
        self._source_lock     = threading.Lock()
        self._stop_requested  = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active_source(self) -> str:
        with self._source_lock:
            return self._active_source

    @QtCore.Slot(str)
    def set_active_source(self, source: str) -> None:
        """Switch active source.  Safe to call from any thread (GUI or other)."""
        if source not in ALL_SOURCES:
            return
        with self._source_lock:
            self._active_source = source
        self.signal_source_changed.emit(source)

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._stop_requested = True

    # ── QThread entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        # Open Webcam first (SD: used directly for LiveCC inference)
        cap_cam = self._open_cap(self.cam_idx, self.target_fps, _WEBCAM_W, _WEBCAM_H)
        if cap_cam is None:
            self.signal_error.emit(
                f"FreeSwitchCamera: 無法開啟 Webcam (index={self.cam_idx})。"
                "請確認實體攝影機已連接且未被其他程式佔用。"
            )
            return

        # Stagger USB init to avoid Windows DirectShow conflict
        time.sleep(0.35)

        # Open OBS Virtual Camera at native resolution (1080p for audience quality).
        cap_vr = self._open_cap(self.vr_idx, self.target_fps, _VR_W, _VR_H)
        if cap_vr is None:
            cap_cam.release()
            self.signal_error.emit(
                f"FreeSwitchCamera: 無法開啟 VR/OBS 相機 (index={self.vr_idx})。"
                "請確認 OBS 已啟動並開啟「虛擬攝影機」。"
            )
            return

        frame_delay = 1.0 / self.target_fps
        _vr_res_logged = False  # print actual OBS resolution once on first frame

        print(
            f"[FreeSwitch] 已開啟 Webcam (idx={self.cam_idx}) + "
            f"OBS VR (idx={self.vr_idx})，初始來源：{self._active_source}"
        )

        while not self._stop_requested:
            t_loop = time.perf_counter()

            # Always grab() both cameras every iteration so the internal
            # hardware buffer stays current.  This is critical: if we only
            # grab the active source, the inactive one builds up stale frames
            # and will show an old frame when the user switches.
            cap_cam.grab()
            cap_vr.grab()

            with self._source_lock:
                src = self._active_source

            # Always retrieve both cameras so signal_vr_frame can always emit
            ret_cam, f_cam = cap_cam.retrieve()
            ret_vr,  f_vr  = cap_vr.retrieve()

            frame_out: Optional[np.ndarray] = None

            if src == SOURCE_DUAL:
                if ret_cam and ret_vr:
                    # Resize VR to webcam height for the side-by-side LiveCC frame.
                    f_vr_sd = cv2.resize(f_vr, (_LIVECC_W, _LIVECC_H))
                    combined_bgr = np.hstack((f_cam, f_vr_sd))   # 1280×480 BGR
                    frame_out = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

            elif src == SOURCE_VR:
                if ret_vr:
                    # Resize VR to LiveCC resolution for signal_frame (inference path).
                    f_vr_sd = cv2.resize(f_vr, (_LIVECC_W, _LIVECC_H))
                    frame_out = cv2.cvtColor(f_vr_sd, cv2.COLOR_BGR2RGB)

            else:  # SOURCE_WEBCAM (default)
                if ret_cam:
                    frame_out = cv2.cvtColor(f_cam, cv2.COLOR_BGR2RGB)

            if frame_out is not None:
                self.signal_frame.emit(frame_out)

            # Audience second screen: always emit VR at native resolution (no resize).
            # The publisher compositor handles output sizing independently.
            if ret_vr:
                if not _vr_res_logged:
                    h_vr, w_vr = f_vr.shape[:2]
                    print(f"[FreeSwitch] OBS VR 實際解析度: {w_vr}×{h_vr}")
                    _vr_res_logged = True
                self.signal_vr_frame.emit(cv2.cvtColor(f_vr, cv2.COLOR_BGR2RGB))

            # Pace loop to target FPS
            elapsed = time.perf_counter() - t_loop
            sleep_t = frame_delay - elapsed
            if sleep_t > 0.001:
                time.sleep(sleep_t)

        cap_cam.release()
        cap_vr.release()
        print("[FreeSwitch] 攝影機已釋放，執行緒結束。")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _open_cap(
        idx: int,
        fps: float,
        width: int = 640,
        height: int = 480,
    ) -> Optional[cv2.VideoCapture]:
        """Try MSMF → DSHOW → CAP_ANY; return first success or None.

        ``width`` / ``height`` are requested resolutions; the driver may
        silently snap to the nearest supported mode (e.g. OBS Virtual Camera
        will honour 1920×1080 when OBS is running at that resolution).
        """
        if platform.system() == "Windows":
            backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        else:
            backends = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS,          fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimal latency
                return cap
            cap.release()
        return None
