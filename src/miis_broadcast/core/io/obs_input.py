# src/miis_broadcast/core/io/obs_input.py

import os
import platform
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .input import BaseInput

# Webcam selection: prefer external Logitech over built-in laptop cameras.
_LOGITECH_NAME_SUBSTR = "logitech"
_SKIP_VIRTUAL_KEYWORDS = (''
    "obs virtual",
    "meta quest",
    "vtubestudio",
)
_BUILTIN_LOW_PRIORITY_KEYWORDS = (
    "asus",
    "ir camera",
    "integrated",
    "facetime",
    "hd webcam",  # common built-in label on laptops
)


def list_dshow_devices(*, log: bool = False) -> List[Tuple[int, str]]:
    """Return DirectShow capture devices as (index, name) pairs."""
    if platform.system() != "Windows":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(enumerate(FilterGraph().get_input_devices()))
    except Exception:
        return []


def _windows_capture_backends() -> Tuple[int, ...]:
    return (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)


def probe_camera_index(idx: int) -> bool:
    """Return True when OpenCV can open idx and read at least one frame."""
    if platform.system() == "Windows":
        backends = _windows_capture_backends()
    else:
        backends = (cv2.CAP_ANY,)
    for backend in backends:
        cap = cv2.VideoCapture(idx, backend)
        try:
            if not cap.isOpened():
                continue
            ret, _ = cap.read()
            if ret:
                return True
        finally:
            cap.release()
    return False


def _is_virtual_camera_name(name: str) -> bool:
    lowered = name.lower()
    return any(kw in lowered for kw in _SKIP_VIRTUAL_KEYWORDS)


def _is_builtin_low_priority(name: str) -> bool:
    lowered = name.lower()
    return any(kw in lowered for kw in _BUILTIN_LOW_PRIORITY_KEYWORDS)


def _is_logitech_name(name: str) -> bool:
    return _LOGITECH_NAME_SUBSTR in name.lower()


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


def find_physical_camera_index(obs_device_name: str = "OBS Virtual Camera",
                               max_scan: int = 10) -> int:
    """
    Find a webcam index for the Webcam / Free Switch camera path.

    Priority:
      1. Logitech devices (e.g. StreamCam) when present and openable.
      2. Other external / non-virtual cameras.
      3. Built-in laptop cameras (ASUS IR/FHD, etc.) — last resort before index 0.
    """
    system = platform.system()

    # --- Linux: use V4L2 sysfs names ---
    if system == "Linux":
        video_dir = "/sys/class/video4linux"
        if os.path.isdir(video_dir):
            logitech_candidate = None
            other_candidate = None
            builtin_candidate = None
            for entry in sorted(os.listdir(video_dir)):
                name_file = os.path.join(video_dir, entry, "name")
                try:
                    with open(name_file) as f:
                        name = f.read().strip()
                    idx = int(entry.replace("video", ""))
                    if obs_device_name.lower() in name.lower():
                        print(f"[Camera] 跳過 OBS 裝置: /dev/video{idx} ({name})")
                        continue
                    if not probe_camera_index(idx):
                        continue
                    if _is_logitech_name(name):
                        logitech_candidate = idx
                        print(f"[Camera] ✅ 找到 Logitech 攝影機 /dev/video{idx} ({name})")
                        break
                    if _is_builtin_low_priority(name):
                        if builtin_candidate is None:
                            builtin_candidate = idx
                    elif other_candidate is None:
                        other_candidate = idx
                except Exception:
                    continue
            for pick, label in (
                (logitech_candidate, "Logitech"),
                (other_candidate, "實體"),
                (builtin_candidate, "內建"),
            ):
                if pick is not None:
                    if label != "Logitech":
                        print(f"[Camera] ✅ 使用{label}攝影機 /dev/video{pick}")
                    return pick

    obs_index = find_obs_camera_index(obs_device_name)
    devices = list_dshow_devices()

    if system == "Windows" and devices:
        print(f"[Camera] 掃描實體相機（優先 Logitech，跳過 OBS index {obs_index}）...")

        def _try_devices(predicate, label: str) -> Optional[int]:
            for idx, name in devices:
                if idx == obs_index or _is_virtual_camera_name(name):
                    continue
                if not predicate(name):
                    continue
                if probe_camera_index(idx):
                    print(f"[Camera] ✅ 使用 {label} index {idx} ({name})")
                    return idx
            return None

        picked = _try_devices(_is_logitech_name, "Logitech")
        if picked is not None:
            return picked

        picked = _try_devices(
            lambda n: not _is_builtin_low_priority(n),
            "外接/非內建",
        )
        if picked is not None:
            return picked

        picked = _try_devices(_is_builtin_low_priority, "內建")
        if picked is not None:
            return picked

    # --- Numeric fallback (non-Windows or pygrabber unavailable) ---
    backend = cv2.CAP_DSHOW if system == "Windows" else cv2.CAP_ANY
    print(f"[Camera] 掃描實體相機（跳過 OBS index {obs_index}）...")
    for idx in range(max_scan):
        if idx == obs_index:
            continue
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"[Camera] ✅ 找到實體攝影機 index {idx} — {w}x{h} @ {fps:.1f} fps")
            return idx
        cap.release()

    print("[Camera] ⚠️  找不到實體攝影機，使用 index 0 作為最後手段")
    return 0


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
    devices = list_dshow_devices()
    if devices:
        print("[OBS] Windows DirectShow 裝置列表：")
        for idx, name in devices:
            print(f"  [index {idx}] {name}")
            if device_name.lower() in name.lower():
                print(f"[OBS] ✅ 找到 OBS Virtual Camera → index {idx}")
                return idx
        print(f"[OBS] ❌ 找不到名稱含 '{device_name}' 的 DirectShow 裝置")
    else:
        print("[OBS] pygrabber 不可用，將改用 index 掃描")
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
        request_width: int | None = None,
        request_height: int | None = None,
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

        # Request native/HD resolution when the caller needs full quality
        # (e.g. Audience second screen). Without this, cv2 falls back to
        # whatever default the driver reports (often 640x480) even though
        # OBS itself may be rendering at 1080p.
        if request_width and request_height:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, request_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, request_height)

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
        if platform.system() == "Windows":
            for backend in _windows_capture_backends():
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    self.capture = cap
                    return True
                cap.release()
            return False
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
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
