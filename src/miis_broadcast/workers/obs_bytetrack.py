# src/miis_broadcast/workers/obs_bytetrack.py
"""
OBS Virtual Camera + ByteTrack tracking worker.

Reads frames from OBS Virtual Camera, runs YOLOX + BYTETracker on each
frame, then emits:
  - signal_frame         : annotated BGR frame for GUI preview  (np.ndarray)
  - signal_subject_frame : padded subject crop (RGB) for LiveCC (np.ndarray)
  - signal_error         : error message string

This thread is used when the user selects "OBS + 追蹤" mode.
"""

import time
from typing import Optional

import numpy as np
from PySide6 import QtCore

from ..core.io.obs_input import OBSVirtualCameraInput
from ..core.models.bytetrack_tracker import ByteTrackWrapper


class OBSByteTrackThread(QtCore.QThread):
    """
    QThread: OBS Virtual Camera → YOLOX + BYTETracker → LiveCC-ready crop.

    Two output signals
    ------------------
    signal_frame         — annotated frame (BGR ndarray) for the GUI preview
    signal_subject_frame — padded subject crop (RGB ndarray) forwarded to LiveCC
    signal_error         — fatal error message; thread stops after emitting this
    """

    signal_frame         = QtCore.Signal(np.ndarray)   # annotated BGR for preview
    signal_subject_frame = QtCore.Signal(np.ndarray)   # subject crop RGB for LiveCC
    signal_error         = QtCore.Signal(str)

    DEFAULT_FPS_FALLBACK = 30.0

    def __init__(
        self,
        # ByteTrack model arguments
        ckpt_path: str,
        exp_file: str,
        bytetrack_repo: Optional[str] = None,
        device: str = "cuda",
        fp16: bool = True,
        fuse: bool = True,
        # Tracker hyper-parameters
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        aspect_ratio_thresh: float = 1.6,
        min_box_area: float = 10.0,
        subject_only: bool = True,
        subject_pad: float = 0.15,
        min_subject_area_ratio: float = 0.03,
        preempt_ratio: float = 4.0,
        # OBS camera arguments
        device_name: str = OBSVirtualCameraInput.DEFAULT_DEVICE_NAME,
        fallback_index: int = 1,
        camera_index: Optional[int] = None,   # when set, bypass OBS detection and open this index directly
        # Pre-loaded tracker instance (skips model load if provided)
        preloaded_tracker: Optional["ByteTrackWrapper"] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)

        # Store init args so we can build objects inside run() on the worker thread
        self._ckpt_path         = ckpt_path
        self._exp_file          = exp_file
        self._bytetrack_repo    = bytetrack_repo
        self._device            = device
        self._fp16              = fp16
        self._fuse              = fuse
        self._track_thresh      = track_thresh
        self._match_thresh      = match_thresh
        self._track_buffer      = track_buffer
        self._aspect_ratio_thresh = aspect_ratio_thresh
        self._min_box_area           = min_box_area
        self._subject_only           = subject_only
        self._subject_pad            = subject_pad
        self._min_subject_area_ratio = min_subject_area_ratio
        self._preempt_ratio          = preempt_ratio

        self._device_name       = device_name
        self._fallback_index    = fallback_index
        self._camera_index      = camera_index   # None = use OBS detection
        self._preloaded_tracker = preloaded_tracker  # reuse if already loaded

        self._stop_requested    = False
        self._cam: Optional[OBSVirtualCameraInput] = None   # OBS input object (for release on stop)
        self._cap = None                                     # cv2.VideoCapture (direct index mode)

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        import cv2
        # ── Open camera source ──────────────────────────────────
        if self._camera_index is not None:
            # Direct camera index mode (e.g. webcam index 0)
            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                self.signal_error.emit(f"[ByteTrack] Cannot open camera index {self._camera_index}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            fps_raw = cap.get(cv2.CAP_PROP_FPS)
            fps = fps_raw if fps_raw and fps_raw > 0 else self.DEFAULT_FPS_FALLBACK
            frame_delay = 1.0 / fps
            use_obs_input = False
            self._cap = cap  # store ref so requestStop() can release to unblock read
            print(f"[ByteTrack] 直接開啟摄影機 index {self._camera_index} @ {fps:.1f} fps")
        else:
            # OBS Virtual Camera detection mode
            try:
                cam = OBSVirtualCameraInput(
                    device_name=self._device_name,
                    fallback_index=self._fallback_index,
                )
            except RuntimeError as e:
                self.signal_error.emit(f"[OBSByteTrack] Camera open failed: {e}")
                return
            fps = cam.fps if cam.fps > 0 else self.DEFAULT_FPS_FALLBACK
            frame_delay = 1.0 / fps
            use_obs_input = True
            self._cam = cam  # store ref so requestStop() can release to unblock read

        # ── Build ByteTrackWrapper (or reuse preloaded instance) ─────
        if self._preloaded_tracker is not None:
            # Reset tracker state so previous run does not affect the new session
            self._preloaded_tracker.reset()
            tracker = self._preloaded_tracker
            print("[OBSByteTrack] ✅ 使用預載 ByteTrackWrapper，跳過模型載入")
        else:
            try:
                tracker = ByteTrackWrapper(
                    ckpt_path             = self._ckpt_path,
                    exp_file              = self._exp_file,
                    bytetrack_repo        = self._bytetrack_repo,
                    device                = self._device,
                    fp16                  = self._fp16,
                    fuse                  = self._fuse,
                    track_thresh          = self._track_thresh,
                    match_thresh          = self._match_thresh,
                    track_buffer          = self._track_buffer,
                    aspect_ratio_thresh   = self._aspect_ratio_thresh,
                    min_box_area          = self._min_box_area,
                    subject_only          = self._subject_only,
                    subject_pad           = self._subject_pad,
                    fps                   = int(fps),
                    min_subject_area_ratio= self._min_subject_area_ratio,
                    preempt_ratio         = self._preempt_ratio,
                )
            except Exception as e:
                self.signal_error.emit(f"[OBSByteTrack] Tracker init failed: {e}")
                return

        frame_id = 0

        # ── Main loop ─────────────────────────────────────────────────
        while not self._stop_requested:
            t_start = time.perf_counter()

            # Read frame from selected source
            try:
                if use_obs_input:
                    frame_rgb = cam.get_frame()   # RGB ndarray
                else:
                    ret, frame_bgr_raw = cap.read()
                    if not ret:
                        self.signal_error.emit("[ByteTrack] Camera read failed")
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2RGB)
            except EOFError as e:
                self.signal_error.emit(f"[OBSByteTrack] Camera read error: {e}")
                break

            # Convert to BGR for YOLOX (OpenCV convention)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            frame_id += 1

            # Run ByteTrack
            try:
                annotated_bgr, subject_crop_rgb = tracker.process(frame_bgr, frame_id)
            except Exception as e:
                self.signal_error.emit(f"[OBSByteTrack] Tracking error on frame {frame_id}: {e}")
                break

            # Emit annotated preview (BGR) every frame for smooth GUI display
            # update_frame uses Format_BGR888 so no conversion needed here
            self.signal_frame.emit(annotated_bgr)

            # Push subject crop to LiveCC every frame (when a subject is detected).
            # LiveCCCameraWorker already handles its own infer_interval + sliding
            # window, so no extra rate-limiting is needed here.
            if subject_crop_rgb is not None:
                self.signal_subject_frame.emit(subject_crop_rgb)
                if frame_id % 60 == 0:  # log once every ~3 seconds
                    print(f"[OBS-ByteTrack] ✅ signal_subject_frame emitted | Frame: {frame_id}")

            # Pace loop to match source FPS
            elapsed = time.perf_counter() - t_start
            remaining = frame_delay - elapsed
            if remaining > 0.001:
                time.sleep(remaining)

        # Release camera if requestStop() has not already done so
        if use_obs_input:
            if self._cam is not None:
                self._cam.release()
                self._cam = None
        else:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    # ------------------------------------------------------------------
    # Slot
    # ------------------------------------------------------------------

    @QtCore.Slot()
    def requestStop(self) -> None:
        # Only set the flag here — do NOT release capture from another thread.
        # Cross-thread release causes DirectShow buffer corruption (garbled frames).
        # The GUI caller uses wait(timeout) + terminate() to handle blocked threads.
        self._stop_requested = True
