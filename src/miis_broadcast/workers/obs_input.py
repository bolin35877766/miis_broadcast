# src/miis_broadcast/workers/obs_input.py

import time
from typing import Optional

import numpy as np
from PySide6 import QtCore

from ..core.io.obs_input import OBSVirtualCameraInput


class OBSCameraThread(QtCore.QThread):
    """
    QThread that reads frames from OBS Virtual Camera and emits them
    via signal_frame.

    Drop-in replacement for CameraThread when the source is OBS Studio.
    The signal interface is identical so the same on_camera_frame handler
    in MainWindow is reused without modification.
    """

    signal_frame = QtCore.Signal(np.ndarray)   # emits RGB ndarray
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

            self.signal_frame.emit(frame_rgb)

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
