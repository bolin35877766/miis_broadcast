# src/miis_broadcast/workers/camera_bytetrack.py
"""
Webcam + ByteTrack tracking worker.

Reads frames from a physical camera by index, runs YOLOX + BYTETracker on
each frame, then emits:
  - signal_frame         : annotated BGR frame for GUI preview  (np.ndarray)
  - signal_subject_frame : padded subject crop (RGB) for LiveCC (np.ndarray)
  - signal_error         : error message string
"""

import time
from typing import Optional

import numpy as np
from PySide6 import QtCore

from ..core.models.bytetrack_tracker import ByteTrackWrapper


class CameraByteTrackThread(QtCore.QThread):
    """
    QThread: Physical Webcam → YOLOX + BYTETracker → LiveCC-ready crop.

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
        # Camera arguments
        camera_index: int = 0,
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

        self._camera_index      = camera_index
        self._preloaded_tracker = preloaded_tracker  # reuse if already loaded

        self._stop_requested    = False
        self._cap = None   # cv2.VideoCapture (released on stop)

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        import cv2
        # ── Open camera (Windows: try MSMF → DSHOW → any) ───────
        cap = None
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            _c = cv2.VideoCapture(self._camera_index, backend)
            if _c.isOpened():
                cap = _c
                break
            _c.release()
        if cap is None or not cap.isOpened():
            self.signal_error.emit(
                f"[ByteTrack] Cannot open camera index {self._camera_index}. "
                "Check that the webcam is connected and not in use by another app."
            )
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal capture latency
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = fps_raw if fps_raw and fps_raw > 0 else self.DEFAULT_FPS_FALLBACK
        frame_delay = 1.0 / fps
        self._cap = cap
        print(f"[ByteTrack] Camera index {self._camera_index} @ {fps:.1f} fps")

        # ── Build ByteTrackWrapper (or reuse preloaded instance) ─────
        if self._preloaded_tracker is not None:
            # Reset tracker state so previous run does not affect the new session
            self._preloaded_tracker.reset()
            tracker = self._preloaded_tracker
            print("[ByteTrack] ✅ Reusing preloaded ByteTrackWrapper")
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
                self.signal_error.emit(f"[ByteTrack] Tracker init failed: {e}")
                return

        frame_id = 0

        # ── Main loop ─────────────────────────────────────────────────
        while not self._stop_requested:
            t_start = time.perf_counter()

            # Read frame — OpenCV gives BGR directly; no RGB conversion needed
            ret, frame_bgr = cap.read()
            if not ret:
                self.signal_error.emit("[ByteTrack] Camera read failed")
                break

            frame_id += 1

            # Run ByteTrack
            try:
                annotated_bgr, subject_crop_rgb = tracker.process(frame_bgr, frame_id)
            except Exception as e:
                self.signal_error.emit(f"[ByteTrack] Tracking error on frame {frame_id}: {e}")
                break

            # Emit annotated preview (BGR) every frame for smooth GUI display
            # update_frame uses Format_BGR888 so no conversion needed here
            self.signal_frame.emit(annotated_bgr)

            # Push subject crop to LiveCC every frame (when a subject is detected).
            # LiveCCCameraWorker already handles its own infer_interval + sliding
            # window, so no extra rate-limiting is needed here.
            if subject_crop_rgb is not None:
                self.signal_subject_frame.emit(subject_crop_rgb)
                if frame_id % 60 == 0:
                    print(f"[ByteTrack] subject_frame emitted | frame={frame_id}")

            # Pace loop to match source FPS
            elapsed = time.perf_counter() - t_start
            remaining = frame_delay - elapsed
            if remaining > 0.001:
                time.sleep(remaining)

        # Release camera
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
