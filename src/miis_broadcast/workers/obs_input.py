# src/miis_broadcast/workers/obs_input.py

import time
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore

from ..core.io.obs_input import OBSVirtualCameraInput

# LiveCC inference / preview expects 640×480 — resize before signal_frame so
# behaviour matches the other camera modes regardless of OBS's canvas size.
_LIVECC_W, _LIVECC_H = 640, 480

# Requested capture resolution — OBS Virtual Camera is asked for 1080p so the
# Audience second screen (signal_vr_frame) gets native quality, mirroring
# FreeSwitchCameraThread's VR capture.
_VR_W, _VR_H = 1920, 1080


class OBSCameraThread(QtCore.QThread):
    """
    QThread that reads frames from OBS Virtual Camera and emits them
    via signal_frame.

    Drop-in replacement for CameraThread when the source is OBS Studio.
    The signal interface is identical so the same on_camera_frame handler
    in MainWindow is reused without modification.
    """

    signal_frame = QtCore.Signal(np.ndarray)   # emits RGB ndarray — 640x480 for LiveCC/preview
    signal_vr_frame = QtCore.Signal(np.ndarray)  # emits RGB ndarray — native/HD, for Audience second screen
    signal_error = QtCore.Signal(str)

    DEFAULT_FPS_FALLBACK = 30.0

    def __init__(
        self,
        device_name: str = OBSVirtualCameraInput.DEFAULT_DEVICE_NAME,
        fallback_index: int = 1,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.device_name = device_name
        self.fallback_index = fallback_index
        self._stop_requested = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        # Attempt to open OBS Virtual Camera
        try:
            cam = OBSVirtualCameraInput(
                device_name=self.device_name,
                fallback_index=self.fallback_index,
                request_width=_VR_W,
                request_height=_VR_H,
            )
        except RuntimeError as e:
            self.signal_error.emit(str(e))
            return

        fps = cam.fps if cam.fps > 0 else self.DEFAULT_FPS_FALLBACK
        frame_delay = 1.0 / fps

        while not self._stop_requested:
            t_start = time.perf_counter()

            try:
                frame_rgb = cam.get_frame()
            except EOFError as e:
                self.signal_error.emit(str(e))
                break

            # Audience second screen: always emit native resolution.
            self.signal_vr_frame.emit(frame_rgb)

            # LiveCC inference / local preview: resize to a fixed 640x480
            # regardless of OBS's actual canvas size.
            h, w = frame_rgb.shape[:2]
            if (w, h) != (_LIVECC_W, _LIVECC_H):
                frame_small = cv2.resize(frame_rgb, (_LIVECC_W, _LIVECC_H))
            else:
                frame_small = frame_rgb
            self.signal_frame.emit(frame_small)

            # Pace the loop to match the source FPS for smooth preview
            elapsed = time.perf_counter() - t_start
            remaining = frame_delay - elapsed
            if remaining > 0.001:
                time.sleep(remaining)

        cam.release()

    # ------------------------------------------------------------------
    # Slot
    # ------------------------------------------------------------------

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._stop_requested = True
