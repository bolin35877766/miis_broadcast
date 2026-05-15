"""
ClientSession: handles one TCP client connection.

Lifecycle:
    HELLO  →  ACK
    START  →  [recv FRAME loop + inference thread running in parallel]
               for each inferred segment: send SEGMENT
    STOP   →  inference thread stops, wait for next START or close

The server runs LiveCC on decoded BGR frames from JPEG only. ByteTrack and subject
cropping run on the thin client when needed; the wire carries full frames or
pre-cropped images per ``MSG_START`` mode. There is no server-side tracking and
no ``MSG_PREVIEW`` to the client.
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
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
from ..network.protocol import (
    MSG_ACK, MSG_CLIENT_DIAG, MSG_ERROR, MSG_FRAME, MSG_HELLO,
    MSG_PING, MSG_PONG, MSG_SEGMENT, MSG_START,
    MSG_STATUS, MSG_STOP,
    PROTOCOL_VERSION, pack_message, read_message,
)
from ..core.utils.gpu_telemetry import (
    cuda_vram_snapshot,
    vram_log_suffix,
    vram_log_suffix_from_wire,
)
log = logging.getLogger(__name__)


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


def _mean_float(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _mean_optional(vals: List[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in vals if x is not None]
    return sum(xs) / len(xs) if xs else None
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
    ) -> None:
        self.sock = sock
        self.addr = addr
        self.livecc_model = livecc_model
        self._stop = False
        self._mode = "camera"
        self._infer_mode: str = "camera"
        self._rx_frames: int = 0
        self._tx_segments: int = 0
        self._infer_cycles: int = 0
        # Drop near-duplicate LiveCC lines (overlapping 2s clips + KV tend to echo wording)
        self._last_segment_text: str = ""
        # Throttle server-side RSS print (~2s, aligned with thin-client QTimer)
        self._server_ram_last_mono: float = 0.0
        # One inference session (START→STOP): rolling numeric samples for end-of-session AVG
        self._telemetry_server_samples: List[Dict[str, Any]] = []
        self._telemetry_client_samples: List[Dict[str, Any]] = []
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
        self._infer_mode = mode
        self._rx_frames = 0
        self._tx_segments = 0
        self._infer_cycles = 0
        self._last_segment_text = ""
        self._server_ram_last_mono = time.monotonic() - 2.01
        self._telemetry_server_samples = []
        self._telemetry_client_samples = []
        buffer: deque[_FrameItem] = deque(maxlen=180)
        stop_event = threading.Event()
        # Single-slot pending JPEG: recv overwrites with the latest FRAME (temporal sampling —
        # track always the freshest frame, not a multi-frame FIFO that discards by order).
        self._frame_queue: Queue[tuple[float, bytes]] = Queue(maxsize=1)
        self._logged_first_frame_decode = False
        frame_thread = threading.Thread(
            target=self._frame_processor_loop,
            args=(buffer, stop_event),
            name="session-frame-decode",
            daemon=True,
        )
        frame_thread.start()
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
                        "[Session %s] MSG_STOP  rx_frames=%d tx_segments=%d",
                        self.addr,
                        self._rx_frames,
                        self._tx_segments,
                    )
                    break
                elif msg.get("type") == MSG_FRAME:
                    self._handle_frame(msg, binary)
                elif msg.get("type") == MSG_CLIENT_DIAG:
                    self._handle_client_diag(msg)
                elif msg.get("type") == MSG_PING:
                    self._send({"type": MSG_PONG})
        finally:
            stop_event.set()
            frame_thread.join(timeout=5.0)
            infer_thread.join(timeout=5.0)
            self._print_session_telemetry_averages()
            log.info(
                "[Session %s] Inference STOP  rx_frames=%d tx_segments=%d infer_cycles=%d",
                self.addr,
                self._rx_frames,
                self._tx_segments,
                self._infer_cycles,
            )
    # ------------------------------------------------------------------ #
    # Thin-client CLIENT_DIAG → printed on server stdout as [Client] lines
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
        gu = msg.get("gpu_vram_used_mib")
        gt = msg.get("gpu_vram_total_mib")
        gto = msg.get("gpu_torch_alloc_mib")
        try:
            gu_f = float(gu) if gu is not None else None
            gt_f = float(gt) if gt is not None else None
            gto_f = float(gto) if gto is not None else None
        except (TypeError, ValueError):
            gu_f = gt_f = gto_f = None
        gpu_sfx = vram_log_suffix_from_wire(gu_f, gt_f, gto_f)
        print(
            f"[Client] RSS={rss:.1f} MiB | JPEG send_queue={qu}/{qm} | "
            f"system_RAM_used={sp:.0f}%{gpu_sfx}"
        )
        self._telemetry_client_samples.append(
            {
                "rss_mib": rss,
                "jpeg_q_used": qu,
                "jpeg_q_max": qm,
                "sys_ram_pct": sp,
                "gpu_vram_used_mib": gu_f,
                "gpu_vram_total_mib": gt_f,
                "gpu_torch_alloc_mib": gto_f,
            }
        )

    def _print_server_process_ram(self, buffer_len: int) -> None:
        """
        RSS of this server's Python process (inference host: LiveCC + JPEG decode).
        Records a snapshot for end-of-session [SESSION AVG] on MSG_STOP path.
        """
        try:
            import psutil
            rss_mib = psutil.Process().memory_info().rss / (1024.0**2)
            sys_pct = psutil.virtual_memory().percent
        except Exception:
            return
        snap = cuda_vram_snapshot(0)
        gpu_sfx = vram_log_suffix(snap)

        gpu_used = snap.used_mib if snap is not None else None
        gpu_total = snap.total_mib if snap is not None else None
        gpu_torch = snap.torch_alloc_mib if snap is not None else None

        self._telemetry_server_samples.append(
            {
                "rss_mib": rss_mib,
                "livecc_buffer": float(buffer_len),
                "sys_ram_pct": sys_pct,
                "gpu_used_mib": gpu_used,
                "gpu_total_mib": gpu_total,
                "gpu_torch_alloc_mib": gpu_torch,
            }
        )
        print(
            f"[Server] RSS={rss_mib:.1f} MiB | livecc_buffer={buffer_len} | "
            f"system_RAM_used={sys_pct:.0f}%{gpu_sfx}"
        )

    def _print_session_telemetry_averages(self) -> None:
        """Pretty-print averages for Server + Client telemetry samples (same fields as periodic lines)."""
        srv_rows = getattr(self, "_telemetry_server_samples", None) or []
        cli_rows = getattr(self, "_telemetry_client_samples", None) or []

        if srv_rows:
            rss = _mean_float([float(r["rss_mib"]) for r in srv_rows])
            buf = _mean_float([float(r["livecc_buffer"]) for r in srv_rows])
            sysp = _mean_float([float(r["sys_ram_pct"]) for r in srv_rows])

            gpu_used_avg = _mean_optional(
                [r.get("gpu_used_mib") for r in srv_rows]
            )
            gpu_tot_ref: Optional[float] = None
            for r in reversed(srv_rows):
                gt = r.get("gpu_total_mib")
                if gt is not None and isinstance(gt, (int, float)) and float(gt) > 0:
                    gpu_tot_ref = float(gt)
                    break

            sfx: str
            if (
                gpu_used_avg is None
                or gpu_tot_ref is None
                or gpu_tot_ref <= 0
            ):
                sfx = " | GPU_VRAM=n/a"
            else:
                pct = 100.0 * float(gpu_used_avg) / float(gpu_tot_ref)
                ga = _mean_optional(
                    [r.get("gpu_torch_alloc_mib") for r in srv_rows]
                )
                torch_part = f"{ga:.0f}" if ga is not None else "0"
                sfx = (
                    f" | GPU_VRAM={gpu_used_avg:.0f}/{gpu_tot_ref:.0f} MiB "
                    f"({pct:.0f}%) torch_alloc={torch_part} MiB"
                )

            print(
                f"[SESSION AVG] Server (n={len(srv_rows)})  "
                f"RSS={rss:.1f} MiB | livecc_buffer={buf:.1f} | "
                f"system_RAM_used={sysp:.0f}%{sfx}"
            )

        if cli_rows:
            rss = _mean_float([float(r["rss_mib"]) for r in cli_rows])
            qu_avg = _mean_float([float(r["jpeg_q_used"]) for r in cli_rows])
            qm = int(cli_rows[-1].get("jpeg_q_max", 0)) if cli_rows else 0
            sysp = _mean_float([float(r["sys_ram_pct"]) for r in cli_rows])

            gu_f = _mean_optional([r.get("gpu_vram_used_mib") for r in cli_rows])
            gt_ref: Optional[float] = None
            for r in reversed(cli_rows):
                tmi = r.get("gpu_vram_total_mib")
                if tmi is not None and isinstance(tmi, (int, float)) and float(tmi) > 0:
                    gt_ref = float(tmi)
                    break
            gto_avg = _mean_optional([r.get("gpu_torch_alloc_mib") for r in cli_rows])
            gpu_sfx = vram_log_suffix_from_wire(gu_f, gt_ref, gto_avg)

            print(
                f"[SESSION AVG] Client (n={len(cli_rows)})  "
                f"RSS={rss:.1f} MiB | JPEG send_queue={qu_avg:.2f}/{qm} | "
                f"system_RAM_used={sysp:.0f}%{gpu_sfx}"
            )

    # ------------------------------------------------------------------ #
    # Frame handling (recv: enqueue JPEG only; decode in _frame_processor_loop)
    # ------------------------------------------------------------------ #
    def _handle_frame(
        self,
        msg: dict,
        binary: bytes,
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
            q = getattr(self, "_frame_queue", None)
            pending = q.qsize() if q is not None else 0
            log.info(
                "[Session %s] FRAME stats  rx=%d pending_jpeg=%d mode=%s",
                self.addr,
                self._rx_frames,
                pending,
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
    # JPEG decode → LiveCC buffer (dedicated thread; no server-side ByteTrack)
    # ------------------------------------------------------------------ #
    def _frame_processor_loop(
        self,
        buffer: deque,
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
            buffer.append(_FrameItem(t=t, frame=frame_bgr))
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
    # Socket send helper
    # ------------------------------------------------------------------ #
    def _send(self, msg: dict, binary: bytes = b"") -> None:
        try:
            self.sock.sendall(pack_message(msg, binary))
        except Exception as e:
            log.debug("[Session %s] Send failed: %s", self.addr, e)
