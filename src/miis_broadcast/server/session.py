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

import difflib
import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from ..network.protocol import (
    MSG_ACK, MSG_CLIENT_DIAG, MSG_ERROR, MSG_FRAME, MSG_HELLO,
    MSG_PING, MSG_PONG, MSG_PREVIEW, MSG_SEGMENT, MSG_START,
    MSG_STATUS, MSG_STOP,
    PROTOCOL_VERSION, pack_message, read_message,
)

log = logging.getLogger(__name__)

# PREVIEW(MSG_PREVIEW back to thin client): max rate (aligned with thin-client FRAME sample; 15 Hz).
_PREVIEW_SAMPLE_OUT_FPS = 15.0


def _commentary_too_similar(prev: str, cur: str, *, ratio: float = 0.86) -> bool:
    """True if the new segment is a near-duplicate of the last (loop / stuck phrasing)."""
    a = (prev or "").strip().lower()
    b = (cur or "").strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(b) < 20:
        return a == b
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio


def _is_cuda_or_oom(exc: BaseException) -> bool:
    """Treat CUDA OOM errors from LiveCC.generate (often deep inside Qwen layers)."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower()


def _is_cuda_recoverable_inference_error(exc: BaseException) -> bool:
    """
    OOM, cuBLAS/cuDNN faults, device-side assert, etc.: clear KV and continue when possible.

    PyTorch emits ``RuntimeError: CUDA error: CUBLAS_STATUS_*`` inside ``model.generate``;
    treating these as fatal disconnects users unnecessarily (often recover after KV reset).
    After a corrupted context errors may repeat until process restart — log explains that.
    """
    if _is_cuda_or_oom(exc):
        return True
    msg = str(exc).lower()
    # cuBLAS BF16 GEMM (Qwen VL language_model), async kernel failures, etc.
    if "cublas" in msg or "cudnn" in msg:
        return True
    if "device-side assert" in msg:
        return True
    if "assert triggered" in msg and "cuda" in msg:
        return True
    if "indexSelectLargeIndex" in str(exc):
        return True
    return False


def _is_oob_vocab_value_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and "input_ids out of vocab" in str(exc).lower()



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
        self.bytetrack_cfg = bytetrack_cfg or {}

        self._stop = False
        self._mode = "camera"
        self._bt_frame_id: int = 0
        self._infer_mode: str = "camera"
        self._rx_frames: int = 0
        self._tx_previews: int = 0
        self._tx_segments: int = 0
        self._infer_cycles: int = 0
        # Drop near-duplicate LiveCC lines (overlapping 2s clips + KV tend to echo wording)
        self._last_segment_text: str = ""
        # Throttle server-side RSS print (~2s, aligned with thin-client QTimer)
        self._server_ram_last_mono: float = 0.0

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
                log.info(
                    "[Session %s] MSG_START mode=%s query_len=%d",
                    self.addr, mode, len(query or ""),
                )
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
        self._infer_mode = mode
        self._rx_frames = 0
        self._tx_previews = 0
        self._tx_segments = 0
        self._infer_cycles = 0
        self._last_segment_text = ""
        self._server_ram_last_mono = time.monotonic() - 2.01

        buffer: deque[_FrameItem] = deque(maxlen=180)
        stop_event = threading.Event()
        # Load ByteTrack before frame worker (worker needs bt reference).
        bt = self._maybe_load_bytetrack(mode)

        # Single-slot pending JPEG: recv overwrites with the latest FRAME (temporal sampling —
        # track always the freshest frame, not a multi-frame FIFO that discards by order).
        self._frame_queue: Queue[tuple[float, bytes]] = Queue(maxsize=1)
        self._logged_first_frame_decode = False
        self._preview_sample_last_emit_mono = 0.0

        frame_thread = threading.Thread(
            target=self._frame_processor_loop,
            args=(buffer, bt, stop_event),
            name="session-frame-gpu",
            daemon=True,
        )
        frame_thread.start()

        if mode == "obs_track" and bt is None:
            log.warning(
                "[Session %s] obs_track but ByteTrack not loaded — "
                "check configs/models.yml bytetrack paths; preview will be raw frames only",
                self.addr,
            )
            self._send({
                "type": MSG_STATUS,
                "msg": "ByteTrack unavailable: using full frame (no boxes). Check server bytetrack config.",
            })

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
                    log.info(
                        "[Session %s] MSG_STOP  rx_frames=%d tx_previews=%d tx_segments=%d",
                        self.addr,
                        self._rx_frames,
                        self._tx_previews,
                        self._tx_segments,
                    )
                    break
                elif msg.get("type") == MSG_FRAME:
                    self._handle_frame(msg, binary, buffer, bt)
                elif msg.get("type") == MSG_CLIENT_DIAG:
                    self._handle_client_diag(msg)
                elif msg.get("type") == MSG_PING:
                    self._send({"type": MSG_PONG})
        finally:
            stop_event.set()
            frame_thread.join(timeout=5.0)
            infer_thread.join(timeout=5.0)
            log.info(
                "[Session %s] Inference STOP  rx_frames=%d tx_previews=%d tx_segments=%d infer_cycles=%d",
                self.addr,
                self._rx_frames,
                self._tx_previews,
                self._tx_segments,
                self._infer_cycles,
            )

    # ------------------------------------------------------------------ #
    # Thin-client diagnostic (same stdout as ByteTrack FPS prints in bytetrack_tracker)
    # ------------------------------------------------------------------ #

    def _handle_client_diag(self, msg: dict) -> None:
        """
        Stats from the thin client's machine (JPEG encode + tcp send queue), not from this server.
        """
        try:
            rss = float(msg.get("rss_mib", 0.0))
            qu = int(msg.get("jpeg_q_used", 0))
            qm = int(msg.get("jpeg_q_max", 0))
            sp = float(msg.get("sys_ram_pct", 0.0))
        except (TypeError, ValueError):
            return
        print(
            f"[Client] RSS={rss:.1f} MiB | JPEG send_queue={qu}/{qm} | "
            f"system_RAM_used={sp:.0f}%"
        )

    @staticmethod
    def _print_server_process_ram(buffer_len: int) -> None:
        """
        RSS of this server's Python process (inference host: LiveCC + JPEG decode).
        """
        try:
            import psutil

            rss_mib = psutil.Process().memory_info().rss / (1024.0**2)
            sys_pct = psutil.virtual_memory().percent
        except Exception:
            return
        print(
            f"[Server] RSS={rss_mib:.1f} MiB | livecc_buffer={buffer_len} | "
            f"system_RAM_used={sys_pct:.0f}%"
        )

    # ------------------------------------------------------------------ #
    # Frame handling (recv: enqueue JPEG only; GPU in _frame_processor_loop)
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
        self._rx_frames += 1
        if self._rx_frames == 1:
            log.info(
                "[Session %s] First FRAME recv  jpeg_bytes=%d t=%.3f",
                self.addr,
                len(binary),
                t,
            )
        elif self._rx_frames % 120 == 0:
            log.info(
                "[Session %s] FRAME stats  rx=%d tx_previews=%d buffer_len=%d mode=%s",
                self.addr,
                self._rx_frames,
                self._tx_previews,
                len(buffer),
                self._infer_mode,
            )

        q = getattr(self, "_frame_queue", None)
        if q is None:
            return
        try:
            q.put_nowait((t, binary))
        except Full:
            # Slot occupied — replace with latest frame (sample), do not accumulate lag.
            try:
                q.get_nowait()
                q.put_nowait((t, binary))
            except (Empty, Full):
                pass

    # ------------------------------------------------------------------ #
    # JPEG decode + ByteTrack (+ optional PREVIEW/buffer): dedicated thread.
    # ------------------------------------------------------------------ #

    def _frame_processor_loop(
        self,
        buffer: deque,
        bt: Any,
        stop_event: threading.Event,
    ) -> None:
        q = self._frame_queue
        while not stop_event.is_set():
            try:
                item = q.get(timeout=0.2)
            except Empty:
                continue
            t, binary = item
            nparr = np.frombuffer(binary, dtype=np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            if not self._logged_first_frame_decode:
                self._logged_first_frame_decode = True
                log.info(
                    "[Session %s] First FRAME decoded  shape=%s t=%.3f",
                    self.addr,
                    getattr(frame_bgr, "shape", "?"),
                    t,
                )

            if bt is not None:
                try:
                    self._bt_frame_id += 1
                    annotated_bgr, subject_rgb = bt.process(
                        frame_bgr, self._bt_frame_id
                    )
                except Exception as e:
                    log.warning("[Session] ByteTrack error: %s", e)
                    if _is_cuda_recoverable_inference_error(e):
                        try:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue

                # PREVIEW return path: at most `_PREVIEW_SAMPLE_OUT_FPS` Hz (consistent 15fps cap).
                gap = 1.0 / _PREVIEW_SAMPLE_OUT_FPS
                pn = time.monotonic()
                if pn - self._preview_sample_last_emit_mono >= gap:
                    self._preview_sample_last_emit_mono = pn
                    ret, jbuf = cv2.imencode(
                        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 78]
                    )
                    if ret:
                        self._tx_previews += 1
                        self._send({"type": MSG_PREVIEW, "t": t}, jbuf.tobytes())

                if subject_rgb is not None:
                    subject_bgr = cv2.cvtColor(
                        cv2.resize(subject_rgb, (640, 480)),
                        cv2.COLOR_RGB2BGR,
                    )
                    buffer.append(_FrameItem(t=t, frame=subject_bgr))
            else:
                buffer.append(_FrameItem(t=t, frame=frame_bgr))
                if self._infer_mode == "obs_track":
                    _now = time.monotonic()
                    _gap = 1.0 / _PREVIEW_SAMPLE_OUT_FPS
                    if _now - self._preview_sample_last_emit_mono >= _gap:
                        self._preview_sample_last_emit_mono = _now
                        ret, jbuf = cv2.imencode(
                            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75]
                        )
                        if ret:
                            self._tx_previews += 1
                            self._send({"type": MSG_PREVIEW, "t": t}, jbuf.tobytes())

            _ram_now = time.monotonic()
            if _ram_now - self._server_ram_last_mono >= 2.0:
                self._server_ram_last_mono = _ram_now
                self._print_server_process_ram(len(buffer))

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

            inference_count += 1
            self._infer_cycles += 1
            log.info(
                "[Session %s] LiveCC run #%d  buffer_size=%d clip_ok",
                self.addr,
                self._infer_cycles,
                len(buffer),
            )

            # Periodic state reset: overlapping 2s clips + KV make echo outputs; clear more often on server
            if inference_count % 3 == 0:
                state = {}
                log.debug("[Session] State reset (count=%d)", inference_count)

            try:
                batch = list(
                    self.livecc_model.live_cc_from_frames(
                        clip=clip, query=query, state=state
                    )
                )
                last_infer_t = time.time()
                for (start_ts, stop_ts), text, state in batch:
                    if stop_event.is_set():
                        break
                    if _commentary_too_similar(self._last_segment_text, text or ""):
                        log.debug(
                            "[Session] Skip near-duplicate segment (t=[%.2f,%.2f])",
                            float(start_ts),
                            float(stop_ts),
                        )
                        continue
                    self._last_segment_text = text or ""
                    self._tx_segments += 1
                    tprev = (text or "").replace("\n", " ")[:100]
                    nseg = self._tx_segments
                    if nseg == 1 or nseg % 10 == 0:
                        log.info(
                            "[Session %s] SEGMENT out #%d  t=[%.2f,%.2f]  %s%s",
                            self.addr,
                            nseg,
                            float(start_ts),
                            float(stop_ts),
                            tprev,
                            "…" if (text and len(text) > 100) else "",
                        )
                    else:
                        log.debug(
                            "[Session %s] SEGMENT out #%d  t=[%.2f,%.2f]  (truncated log)",
                            self.addr,
                            nseg,
                            float(start_ts),
                            float(stop_ts),
                        )
                    self._send({
                        "type":    MSG_SEGMENT,
                        "start_t": float(start_ts),
                        "stop_t":  float(stop_ts),
                        "text":    text,
                    })
            except RuntimeError as e:
                if _is_cuda_recoverable_inference_error(e):
                    # KV clear + bump timer: avoids a tight retry loop while respecting
                    # infer_interval. Covers CUBLAS / device-side asserts / OOM transient faults.
                    log.warning("[Session] LiveCC recoverable GPU error — resetting KV: %s", e)
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    state = {}
                    last_infer_t = time.time()
                    continue
                log.exception("[Session] Inference RuntimeError: %s", e)
                self._send({"type": MSG_ERROR, "msg": str(e)})
                stop_event.set()
            except ValueError as e:
                if _is_oob_vocab_value_error(e):
                    log.warning("[Session] %s — resetting KV.", e)
                    state = {}
                    continue
                log.exception("[Session] Inference ValueError: %s", e)
                self._send({"type": MSG_ERROR, "msg": str(e)})
                stop_event.set()
            except Exception as e:
                log.exception("[Session] Inference error: %s", e)
                self._send({"type": MSG_ERROR, "msg": str(e)})
                stop_event.set()

    # ------------------------------------------------------------------ #
    # ByteTrack loader
    # ------------------------------------------------------------------ #

    def _maybe_load_bytetrack(self, mode: str) -> Optional[Any]:
        if mode != "obs_track":
            return None
        if not self.bytetrack_cfg:
            log.warning(
                "[Session] bytetrack section missing or empty in model config — "
                "cannot load ByteTrack for obs_track"
            )
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
            log.warning("[Session] ByteTrack load failed: %s", e, exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Socket send helper
    # ------------------------------------------------------------------ #

    def _send(self, msg: dict, binary: bytes = b"") -> None:
        try:
            self.sock.sendall(pack_message(msg, binary))
        except Exception as e:
            log.debug("[Session %s] Send failed: %s", self.addr, e)
