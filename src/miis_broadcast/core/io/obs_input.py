# src/miis_broadcast/core/io/obs_input.py

import os
import platform
import cv2
import numpy as np

from .input import BaseInput


def list_all_cameras(max_scan: int = 10) -> dict:
    """
    Enumerate all available camera indices and return {index: opens_ok}.
    Prints a log line for each device found.
    """
    system = platform.system()
    backend = cv2.CAP_DSHOW if system == "Windows" else cv2.CAP_ANY
    found = {}
    print("[OBS] 掃描所有攝影機裝置：")
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx, backend)
        ok = cap.isOpened()
        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  [index {idx}] ✅ 開啟成功 — {w}x{h} @ {fps:.1f} fps")
        cap.release()
        found[idx] = ok
    return found


def find_obs_camera_index_linux(device_name: str = "OBS Virtual Camera") -> int:
    """
    On Linux, OBS Virtual Camera appears as a V4L2 loopback device.
    Read /sys/class/video4linux/videoN/name to find the matching device.
    """
    video_dir = "/sys/class/video4linux"
    if not os.path.isdir(video_dir):
        return -1

    entries = sorted(os.listdir(video_dir))
    print(f"[OBS] Linux V4L2 裝置列表：")
    for entry in entries:
        name_file = os.path.join(video_dir, entry, "name")
        try:
            with open(name_file) as f:
                name = f.read().strip()
            print(f"  {entry}: {name}")
            if device_name.lower() in name.lower():
                # Extract index number from "videoN"
                idx = int(entry.replace("video", ""))
                print(f"[OBS] ✅ 找到 OBS Virtual Camera → /dev/video{idx} (index {idx})")
                return idx
        except Exception:
            continue
    print(f"[OBS] ❌ 找不到名稱含 '{device_name}' 的 V4L2 裝置")
    return -1


def find_obs_camera_index_windows(device_name: str = "OBS Virtual Camera") -> int:
    """
    On Windows, use pygrabber to enumerate DirectShow devices by name.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        print(f"[OBS] Windows DirectShow 裝置列表：")
        for idx, name in enumerate(devices):
            print(f"  [index {idx}] {name}")
            if device_name.lower() in name.lower():
                print(f"[OBS] ✅ 找到 OBS Virtual Camera → index {idx}")
                return idx
        print(f"[OBS] ❌ 找不到名稱含 '{device_name}' 的 DirectShow 裝置")
    except Exception as e:
        print(f"[OBS] pygrabber 不可用 ({e})，將改用 index 掃描")
    return -1


def find_obs_camera_index(device_name: str = "OBS Virtual Camera") -> int:
    """Cross-platform OBS Virtual Camera index detection."""
    system = platform.system()
    if system == "Linux":
        return find_obs_camera_index_linux(device_name)
    elif system == "Windows":
        return find_obs_camera_index_windows(device_name)
    return -1


class OBSVirtualCameraInput(BaseInput):
    """
    Captures frames from OBS Virtual Camera.

    Scans DirectShow (Windows) or V4L2 (Linux) devices to find the OBS
    Virtual Camera by name, then opens it with the correct numeric index.
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

        # Always list all cameras first for debug visibility
        list_all_cameras()

        # Auto-detect OBS Virtual Camera index
        obs_index = find_obs_camera_index(device_name)
        if obs_index >= 0:
            print(f"[OBS] 嘗試開啟 index {obs_index}...")
            opened = self._try_open_index(obs_index)
            if opened:
                print(f"[OBS] ✅ 成功開啟 OBS Virtual Camera (index {obs_index})")
            else:
                print(f"[OBS] ❌ index {obs_index} 開啟失敗，改用 fallback index {fallback_index}")
                opened = self._try_open_index(fallback_index)
        else:
            print(f"[OBS] ⚠️  自動偵測失敗，使用 fallback index {fallback_index}")
            opened = self._try_open_index(fallback_index)

        if not opened:
            raise RuntimeError(
                f"[OBSVirtualCameraInput] Cannot open OBS Virtual Camera. "
                f"Make sure OBS is running and Virtual Camera is started.\n"
                f"On Linux, also ensure v4l2loopback is loaded: sudo modprobe v4l2loopback"
            )

        actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[OBS] 📷 串流規格：{actual_w}x{actual_h} @ {self.fps:.1f} fps")
        self.calculate_frame_delay()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_open_index(self, index: int) -> bool:
        """Open the camera at a specific numeric index."""
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
