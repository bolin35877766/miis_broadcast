"""
ClientSession: handles one TCP client connection.

Lifecycle:
    HELLO  →  ACK
    START  →  [recv FRAME loop + inference thread running in parallel]
               for each inferred segment: send SEGMENT
    STOP   →  inference thread stops, wait for next START or close

ByteTrack (mode=obs_track):
    Server runs ByteTrack on raw client frames, extracts subject crop,
    feeds the crop into the LiveCC inference pipeline.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..network.protocol import (
    MSG_ACK, MSG_ERROR, MSG_FRAME, MSG_HELLO,
    MSG_PING, MSG_PONG, MSG_PREVIEW, MSG_SEGMENT, MSG_START,
    MSG_STATUS, MSG_STOP,
    PROTOCOL_VERSION, pack_message, read_message,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameItem: mirrors workers/livecc.py to avoid circular import
# ---------------------------------------------------------------------------

@dataclass
class _FrameItem:
    t: float
    frame: np.ndarray   # BGR


# ---------------------------------------------------------------------------
# ClientSession
# ---------------------------------------------------------------------------

class ClientSession:
    """Manages one connected client from HELLO to disconnect."""

    def __init__(
        self,
        sock: socket.socket,
        addr: tuple,
        livecc_model: Any,
        bytetrack_cfg: Dict[str, Any],
    ) -> None:
        self.sock = sock
        self.addr = addr
        self.livecc_model = livecc_model
        self.bytetrack_cfg = bytetrack_cfg

        self._stop = False
        self._mode = "camera"
        self._bt_frame_id: int = 0
        self._last_preview_mono: float = 0.0

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            self._handle_session()
        except Exception:
            log.exception("[Session %s] Unexpected error", self.addr)
        finally:
            try:
                self.sock.close()
            except Exception:
                pass
            log.info("[Session %s] Disconnected", self.addr)

    # ------------------------------------------------------------------ #
    # Session flow
    # ------------------------------------------------------------------ #

    def _handle_session(self) -> None:
        # Expect HELLO
        result = read_message(self.sock)
        if result is None:
            return
        hello, _ = result
        if hello.get("type") != MSG_HELLO:
            self._send({"type": MSG_ERROR, "msg": "Expected HELLO"})
            return

        client_ver = hello.get("protocol_version", 0)
        if client_ver != PROTOCOL_VERSION:
            self._send({
                "type": MSG_ERROR,
                "msg": f"Protocol version mismatch: got {client_ver}, expected {PROTOCOL_VERSION}",
            })
            return

        self._mode = hello.get("mode", "camera")
        self._send({"type": MSG_ACK, "protocol_version": PROTOCOL_VERSION})
        log.info("[Session %s] HELLO ok, mode=%s", self.addr, self._mode)

        # Wait for START messages
        while not self._stop:
            result = read_message(self.sock)
            if result is None:
                return
            msg, binary = result

            if msg.get("type") == MSG_START:
                mode  = msg.get("mode",  self._mode)
                query = msg.get("query", "")
                self._run_inference_session(mode, query)
            elif msg.get("type") == MSG_STOP:
                break
            elif msg.get("type") == MSG_PING:
                self._send({"type": MSG_PONG})

    # ------------------------------------------------------------------ #
    # Inference session (one START → STOP cycle)
    # ------------------------------------------------------------------ #

    def _run_inference_session(self, mode: str, query: str) -> None:
        log.info("[Session %s] Inference START mode=%s", self.addr, mode)
        self._send({"type": MSG_STATUS, "msg": f"Inference started (mode={mode})"})

        self._bt_frame_id = 0
        self._last_preview_mono = 0.0

        buffer: deque[_FrameItem] = deque(maxlen=180)
        stop_event = threading.Event()

        # Load ByteTrack if needed
        bt = self._maybe_load_bytetrack(mode)

        # Inference loop runs in a separate thread so the main thread can
        # keep receiving frames without blocking.
        infer_thread = threading.Thread(
            target=self._inference_loop,
            args=(buffer, query, stop_event),
            daemon=True,
        )
        infer_thread.start()

        # Frame-receive loop (main thread of this session)
        try:
            while not self._stop and not stop_event.is_set():
                result = read_message(self.sock)
                if result is None:
                    break
                msg, binary = result

                if msg.get("type") == MSG_STOP:
                    break
                elif msg.get("type") == MSG_FRAME:
                    self._handle_frame(msg, binary, buffer, bt)
                elif msg.get("type") == MSG_PING:
                    self._send({"type": MSG_PONG})
        finally:
            stop_event.set()
            infer_thread.join(timeout=5.0)
            log.info("[Session %s] Inference STOP", self.addr)

    # ------------------------------------------------------------------ #
    # Frame handling
    # ------------------------------------------------------------------ #

    def _handle_frame(
        self,
        msg: dict,
        binary: bytes,
        buffer: deque,
        bt: Any,
    ) -> None:
        if not binary:
            return
        t = float(msg.get("t", 0.0))

        nparr = np.frombuffer(binary, dtype=np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return

        if bt is not None:
            # ByteTrack: process() returns annotated BGR (boxes) + subject crop (RGB) for LiveCC
            try:
                self._bt_frame_id += 1
                annotated_bgr, subject_rgb = bt.process(frame_bgr, self._bt_frame_id)

                # Throttle preview to ~12 fps so the thin-client UI can show boxes
                _now = time.monotonic()
                if _now - self._last_preview_mono >= (1.0 / 12.0):
                    self._last_preview_mono = _now
                    ret, jbuf = cv2.imencode(
                        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 72]
                    )
                    if ret:
                        self._send({"type": MSG_PREVIEW, "t": t}, jbuf.tobytes())

                if subject_rgb is not None:
                    subject_bgr = cv2.cvtColor(
                        cv2.resize(subject_rgb, (640, 480)),
                        cv2.COLOR_RGB2BGR,
                    )
                    buffer.append(_FrameItem(t=t, frame=subject_bgr))
            except Exception as e:
                log.debug("[Session] ByteTrack error: %s", e)
        else:
            buffer.append(_FrameItem(t=t, frame=frame_bgr))

    # ------------------------------------------------------------------ #
    # Inference loop (background thread)
    # ------------------------------------------------------------------ #

    def _inference_loop(
        self,
        buffer: deque,
        query: str,
        stop_event: threading.Event,
    ) -> None:
        from ..workers.livecc import build_clip_from_buffer

        state: Dict[str, Any] = {}
        inference_count = 0
        infer_interval = 2.0
        last_infer_t = time.time()

        while not stop_event.is_set():
            now = time.time()
            if now - last_infer_t < infer_interval:
                time.sleep(0.1)
                continue

            if len(buffer) < 3:
                time.sleep(0.1)
                continue

            # Build clip from shared buffer (thread-safe read for deque)
            clip = build_clip_from_buffer(
                buffer, window_sec=2.0, target_fps=2.0
            )
            if clip is None:
                time.sleep(0.1)
                continue

            last_infer_t = now
            inference_count += 1

            # Periodic state reset to avoid repetition loops
            if inference_count % 5 == 0:
                state = {}
                log.debug("[Session] State reset (count=%d)", inference_count)

            try:
                for (start_ts, stop_ts), text, state in self.livecc_model.live_cc_from_frames(
                    clip=clip, query=query, state=state
                ):
                    if stop_event.is_set():
                        break
                    self._send({
                        "type":    MSG_SEGMENT,
                        "start_t": float(start_ts),
                        "stop_t":  float(stop_ts),
                        "text":    text,
                    })
            except Exception as e:
                log.exception("[Session] Inference error: %s", e)
                self._send({"type": MSG_ERROR, "msg": str(e)})
                stop_event.set()

    # ------------------------------------------------------------------ #
    # ByteTrack loader
    # ------------------------------------------------------------------ #

    def _maybe_load_bytetrack(self, mode: str) -> Optional[Any]:
        if mode != "obs_track" or not self.bytetrack_cfg:
            return None
        try:
            from ..core.models.bytetrack_tracker import ByteTrackWrapper
            cfg = self.bytetrack_cfg
            bt = ByteTrackWrapper(
                ckpt_path              = cfg.get("ckpt_path", ""),
                exp_file               = cfg.get("exp_file",  ""),
                bytetrack_repo         = cfg.get("bytetrack_repo") or None,
                device                 = cfg.get("device", "cuda"),
                fp16                   = bool(cfg.get("fp16", True)),
                fuse                   = bool(cfg.get("fuse", True)),
                track_thresh           = float(cfg.get("track_thresh", 0.5)),
                match_thresh           = float(cfg.get("match_thresh", 0.8)),
                track_buffer           = int(cfg.get("track_buffer", 30)),
                aspect_ratio_thresh    = float(cfg.get("aspect_ratio_thresh", 1.6)),
                min_box_area           = float(cfg.get("min_box_area", 10)),
                subject_only           = bool(cfg.get("subject_only", True)),
                subject_pad            = float(cfg.get("subject_pad", 0.15)),
                min_subject_area_ratio = float(cfg.get("min_subject_area_ratio", 0.03)),
                preempt_ratio          = float(cfg.get("preempt_ratio", 4.0)),
            )
            log.info("[Session] ByteTrack loaded for mode=obs_track")
            return bt
        except Exception as e:
            log.warning("[Session] ByteTrack load failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Socket send helper
    # ------------------------------------------------------------------ #

    def _send(self, msg: dict, binary: bytes = b"") -> None:
        try:
            self.sock.sendall(pack_message(msg, binary))
        except Exception as e:
            log.debug("[Session %s] Send failed: %s", self.addr, e)
