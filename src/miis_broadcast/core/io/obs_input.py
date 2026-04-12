# src/miis_broadcast/core/io/obs_input.py

import platform
import cv2
import numpy as np

from .input import BaseInput


class OBSVirtualCameraInput(BaseInput):
    """
    Captures frames from OBS Virtual Camera.

    On Windows, OpenCV's DirectShow backend is used to open the named
    virtual device ("video=OBS Virtual Camera").  When the named device
    cannot be found (e.g. OBS not running, or non-Windows OS), it falls
    back to a numeric camera index supplied at construction time.
    """

    DEFAULT_DEVICE_NAME = "OBS Virtual Camera"

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        fallback_index: int = 1,
    ) -> None:
        super().__init__(device_name)
        self.device_name = device_name
        self.fallback_index = fallback_index

        # Attempt to open; raises if both methods fail
        if not self._try_open_named():
            if not self._try_open_index():
                raise RuntimeError(
                    f"[OBSVirtualCameraInput] Cannot open OBS Virtual Camera. "
                    f"Tried named device '{device_name}' and index {fallback_index}. "
                    f"Make sure OBS is running and Virtual Camera is started."
                )

        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.calculate_frame_delay()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_open_named(self) -> bool:
        """
        Try to open the virtual camera by its DirectShow device name.
        This only works on Windows because DirectShow is a Windows API.
        """
        if platform.system() != "Windows":
            return False

        cap = cv2.VideoCapture(f"video={self.device_name}", cv2.CAP_DSHOW)
        if cap.isOpened():
            self.capture = cap
            return True
        cap.release()
        return False

    def _try_open_index(self) -> bool:
        """
        Fallback: open the camera by a numeric index.
        Useful if OBS Virtual Camera appears as a numbered device.
        """
        cap = cv2.VideoCapture(self.fallback_index)
        if cap.isOpened():
            self.capture = cap
            return True
        cap.release()
        return False

    # ------------------------------------------------------------------
    # BaseInput interface
    # ------------------------------------------------------------------

    def verify_input_source(self) -> bool:
        # Verification is performed inside __init__ via _try_open_* methods
        return True

    def get_frame(self) -> np.ndarray:
        """Read one RGB frame from OBS Virtual Camera."""
        if self.capture is not None and self.capture.isOpened():
            status, img = self.capture.read()
            if status:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raise EOFError(
                "[OBSVirtualCameraInput] Failed to read frame — "
                "OBS may have stopped the Virtual Camera."
            )
        raise EOFError("[OBSVirtualCameraInput] Camera capture is not opened.")

    def goto_frame(self, frame_position: int) -> None:
        # Live stream: seeking is not applicable
        pass
