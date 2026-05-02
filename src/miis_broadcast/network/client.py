"""
SocketClientRunner: QThread-based TCP client that sends compressed frames to
the remote inference server and receives SEGMENT / STATUS / ERROR messages back.

Usage from main thread:
    runner = SocketClientRunner("127.0.0.1", 9000)
    runner.signal_connected.connect(...)
    runner.signal_segment.connect(on_segment_slot)
    runner.start()                          # connects and begins recv loop
    runner.send_frame(frame_bgr, t)         # thread-safe, call anytime
    runner.start_inference("camera", query) # tell server to begin
    runner.stop_inference()                 # tell server to stop
    runner.disconnect_and_quit()            # graceful shutdown
"""
from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore

from .protocol import (
    MSG_ACK, MSG_CLIENT_DIAG, MSG_ERROR, MSG_FRAME, MSG_HELLO,
    MSG_PING, MSG_PONG, MSG_PREVIEW, MSG_SEGMENT, MSG_START,
    MSG_STATUS, MSG_STOP,
    PROTOCOL_VERSION, pack_message, read_message,
)

log = logging.getLogger(__name__)

# Nominal upstream rate: GUI emits ~30 camera callbacks/sec (thin client encode path).
_REMOTE_INPUT_NOMINAL_FPS = 30.0
# Wired FRAME rate after deterministic sampling: ratio OUT/IN = 15/30 = 1/2 (phase accumulator).
_REMOTE_FRAME_SAMPLE_OUT_FPS = 15.0
_FRAME_SAMPLE_PHASE_STEP = (
    _REMOTE_FRAME_SAMPLE_OUT_FPS / _REMOTE_INPUT_NOMINAL_FPS
)
# Maximum pre-encoded blobs waiting for tcp sendall (different from FPS sampling).
_FRAME_QUEUE_MAX = 30


class SocketClientRunner(QtCore.QThread):
    """
    Connects to the inference server and runs the receive loop in a background
    QThread.  All outgoing sends (frames, control messages) are thread-safe.

    Frame sending is fully non-blocking from the caller's perspective: send_frame()
    applies deterministic **30 Hz → 15 Hz** downsampling via a phase accumulator, then enqueues JPEG
    wire bytes; a background thread calls sendall().

    Signals (emitted from the background thread, delivered via Qt queued
    connection to the main/GUI thread automatically):
        signal_connected        — TCP handshake + HELLO/ACK succeeded
        signal_disconnected(str)— server closed or error after connected
        signal_connect_error(str)— could not establish connection at all
        signal_segment(f, f, s) — start_t, stop_t, text from server
        signal_status(str)      — informational message from server
        signal_error(str)       — error message from server
        signal_preview(object)  — BGR ndarray from server tracking overlay
    """

    signal_connected     = QtCore.Signal()
    signal_disconnected  = QtCore.Signal(str)
    signal_connect_error = QtCore.Signal(str)
    signal_segment       = QtCore.Signal(float, float, str)
    signal_status        = QtCore.Signal(str)
    signal_error         = QtCore.Signal(str)
    # BGR numpy (HxWx3 uint8) — server-side tracking preview with boxes
    signal_preview       = QtCore.Signal(object)

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._host = host
        self._port = port
        self._sock: Optional[socket.socket] = None
        self._stop_requested = False
        self._frame_id = 0
        # Protects control-message sends (START / STOP / HELLO).
        # Frame sends go through the dedicated sender thread instead.
        self._ctrl_lock = threading.Lock()
        # Pre-encoded JPEG bytes (not raw numpy) so encoding is off the GUI thread.
        self._frame_queue: "queue.Queue[Optional[bytes]]" = queue.Queue(
            maxsize=_FRAME_QUEUE_MAX
        )
        # Phase accumulator for 30 -> 20 fps (emit when phase crosses 1.0): standard resampling.
        self._sample_phase: float = 0.0

    # ------------------------------------------------------------------ #
    # QThread entry point
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self._stop_requested = False
        self._sample_phase = 0.0
        # Drain any leftover frames from a previous run
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((self._host, self._port))
            sock.settimeout(None)
            self._sock = sock
        except Exception as e:
            self.signal_connect_error.emit(f"無法連線 {self._host}:{self._port} — {e}")
            return

        # HELLO handshake
        self._send_ctrl(pack_message({"type": MSG_HELLO, "protocol_version": PROTOCOL_VERSION}))

        self.signal_connected.emit()
        log.info("[SocketClient] Connected to %s:%d", self._host, self._port)

        # Start the dedicated frame-sender thread so send_frame() is never blocking
        sender_thread = threading.Thread(
            target=self._frame_sender_loop,
            daemon=True,
            name="socket-frame-sender",
        )
        sender_thread.start()

        # Receive loop (runs in the QThread's worker thread)
        while not self._stop_requested:
            try:
                result = read_message(self._sock)
            except Exception as e:
                if not self._stop_requested:
                    self.signal_disconnected.emit(str(e))
                break
            if result is None:
                if not self._stop_requested:
                    self.signal_disconnected.emit("伺服器關閉連線")
                break
            self._handle_message(*result)

        # Signal the frame sender to stop, then wait briefly
        self._stop_requested = True
        try:
            self._frame_queue.put_nowait(None)  # sentinel to unblock the sender
        except queue.Full:
            pass
        sender_thread.join(timeout=2.0)

        # Cleanup
        with self._ctrl_lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    # ------------------------------------------------------------------ #
    # Outgoing API (safe to call from any thread)
    # ------------------------------------------------------------------ #

    def send_frame(self, frame_bgr: np.ndarray, t: float) -> None:
        """
        Encode frame and enqueue for sending.  Returns immediately (non-blocking).

        **Sampling specification:** callers emit ~``_REMOTE_INPUT_NOMINAL_FPS`` callbacks/sec (30 Hz).
        We apply a **phase accumulator** step ``OUT/IN = 15/30`` per callback; whenever the
        accumulated phase reaches 1.0, we JPEG-encode **that** frame and enqueue one FRAME
        (~15/sec long-term average). Reproducible ratio **15/30 = 1/2**.

        If the outbound queue is full, the encoded blob may be dropped (TCP backlog); GUI never blocks.
        """
        if self._sock is None or self._stop_requested:
            return

        self._sample_phase += _FRAME_SAMPLE_PHASE_STEP
        if self._sample_phase < 1.0:
            return
        self._sample_phase -= 1.0

        try:
            ret, jpeg_buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if not ret:
                return
            msg = {
                "type": MSG_FRAME,
                "frame_id": self._frame_id,
                "t": float(t),
            }
            self._frame_id += 1
            wire = pack_message(msg, jpeg_buf.tobytes())
            self._frame_queue.put_nowait(wire)
        except queue.Full:
            pass  # TCP sender backlogged — drop encoded wire only
        except Exception as e:
            log.warning("[SocketClient] send_frame encode: %s", e)

    def get_frame_send_queue_levels(self) -> tuple[int, int]:
        """Return (current_qsize, maxsize) for the outgoing JPEG queue (telemetry)."""
        return self._frame_queue.qsize(), _FRAME_QUEUE_MAX

    def send_client_diagnostic(
        self,
        rss_mib: float,
        jpeg_q_used: int,
        jpeg_q_max: int,
        sys_ram_pct: float,
        gpu_vram_used_mib: Optional[float] = None,
        gpu_vram_total_mib: Optional[float] = None,
        gpu_torch_alloc_mib: Optional[float] = None,
    ) -> None:
        """
        Send thin-client RAM / JPEG queue / optional CUDA VRAM stats for server stdout.
        """
        if self._sock is None or self._stop_requested:
            return
        payload: dict = {
            "type": MSG_CLIENT_DIAG,
            "rss_mib": float(rss_mib),
            "jpeg_q_used": int(jpeg_q_used),
            "jpeg_q_max": int(jpeg_q_max),
            "sys_ram_pct": float(sys_ram_pct),
        }
        if gpu_vram_used_mib is not None:
            payload["gpu_vram_used_mib"] = float(gpu_vram_used_mib)
        if gpu_vram_total_mib is not None:
            payload["gpu_vram_total_mib"] = float(gpu_vram_total_mib)
        if gpu_torch_alloc_mib is not None:
            payload["gpu_torch_alloc_mib"] = float(gpu_torch_alloc_mib)
        self._send_ctrl(pack_message(payload))

    def start_inference(self, mode: str, query: str) -> None:
        """Tell server to begin inference with given mode and query prompt."""
        self._send_ctrl(
            pack_message({"type": MSG_START, "mode": mode, "query": query})
        )

    def stop_inference(self) -> None:
        """Tell server to stop current inference session."""
        self._send_ctrl(pack_message({"type": MSG_STOP}))

    def disconnect_and_quit(self) -> None:
        """Gracefully close socket and stop the QThread."""
        self._stop_requested = True
        try:
            self._frame_queue.put_nowait(None)  # sentinel for the sender thread
        except queue.Full:
            pass
        with self._ctrl_lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
        self.quit()
        self.wait(2000)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _frame_sender_loop(self) -> None:
        """Background thread: drains the frame queue and does the actual sendall."""
        while not self._stop_requested:
            try:
                wire = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if wire is None:
                break  # sentinel — time to stop
            with self._ctrl_lock:
                sock = self._sock
            if sock is None:
                continue
            try:
                sock.sendall(wire)
            except Exception as e:
                if not self._stop_requested:
                    log.warning("[SocketClient] frame send failed: %s", e)
                break

    def _send_ctrl(self, data: bytes) -> None:
        """Send a control message (HELLO / START / STOP).  Thread-safe."""
        with self._ctrl_lock:
            if self._sock is None:
                return
            try:
                self._sock.sendall(data)
            except Exception as e:
                log.warning("[SocketClient] ctrl send failed: %s", e)

    def _handle_message(self, msg: dict, binary: bytes) -> None:
        t = msg.get("type")
        if t == MSG_SEGMENT:
            self.signal_segment.emit(
                float(msg.get("start_t", 0.0)),
                float(msg.get("stop_t", 0.0)),
                str(msg.get("text", "")),
            )
        elif t == MSG_ACK:
            self.signal_status.emit("伺服器就緒 (Server Ready)")
        elif t == MSG_STATUS:
            self.signal_status.emit(str(msg.get("msg", "")))
        elif t == MSG_ERROR:
            self.signal_error.emit(str(msg.get("msg", "")))
        elif t == MSG_PING:
            self._send_ctrl(pack_message({"type": MSG_PONG}))
        elif t == MSG_PREVIEW:
            if not binary:
                return
            nparr = np.frombuffer(binary, dtype=np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is not None:
                # copy(): recv buffer may be reused; GUI thread must own the array
                self.signal_preview.emit(bgr.copy())
