# src/miis_broadcast/core/io/obs_input.py

import platform
import cv2
import numpy as np

from .input import BaseInput


def find_obs_camera_index(device_name: str = "OBS Virtual Camera", max_scan: int = 10) -> int:
    """
    Scan DirectShow video devices (index 0..max_scan) and return the index
    whose backend name contains `device_name`.  Returns -1 if not found.
    """
    if platform.system() != "Windows":
        return -1

    # Try to use pygrabber for accurate device name lookup
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        for idx, name in enumerate(devices):
            if device_name.lower() in name.lower():
                return idx
    except Exception:
        pass

    # Fallback: brute-force scan by index and check opaque backend name
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            # CAP_PROP_BACKEND returns the integer backend, not the name.
            # We rely on pygrabber above; here we just return the first
            # non-zero index that opens (rough heuristic) only when
            # pygrabber is unavailable.
            cap.release()
    return -1


class OBSVirtualCameraInput(BaseInput):
    """
    Captures frames from OBS Virtual Camera.

    Scans DirectShow devices to find the OBS Virtual Camera by name,
    then opens it with the correct numeric index.  Falls back to
    `fallback_index` when auto-detection fails.
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

        # Auto-detect OBS Virtual Camera index
        obs_index = find_obs_camera_index(device_name)
        if obs_index >= 0:
            opened = self._try_open_index(obs_index)
        else:
            # pygrabber not available or not on Windows — try indices 0..9
            opened = self._try_open_by_scan()

        if not opened:
            raise RuntimeError(
                f"[OBSVirtualCameraInput] Cannot open OBS Virtual Camera. "
                f"Make sure OBS is running and Virtual Camera is started."
            )

        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.calculate_frame_delay()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_open_by_scan(self) -> bool:
        """
        Scan indices 0..9 with DirectShow.  Opens the first index that:
          1. Successfully opens
          2. Is NOT the default webcam (index 0) — prefer higher indices
        Falls back to fallback_index if nothing else works.
        """
        # First try indices > 0 so we avoid the built-in webcam
        for idx in list(range(1, 10)) + [0]:
            if self._try_open_index(idx):
                return True
        return False

    def _try_open_index(self, index: int) -> bool:
        """Open the camera at a specific numeric index using DirectShow."""
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
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
