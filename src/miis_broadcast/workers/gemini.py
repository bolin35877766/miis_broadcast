# src/miis_broadcast/workers/gemini.py

from __future__ import annotations
from collections import deque
from typing import Optional
from PySide6 import QtCore
import logging
import time

# Pre-import in main thread so GeminiWorker and GeminiBackgroundWorker threads
# don't race on the import lock when both call initialize() simultaneously.
from ..core.models.gemini_broadcaster import _get_client as _gemini_get_client  # noqa: F401


class GeminiWorker(QtCore.QObject):
    """
    Qt worker that enriches LiveCC visual event dicts via Gemini API.
    Runs in a dedicated QThread.

    Queue design:
      P3 (normal):  process_segment() Slot via QueuedConnection → appends to deque tail
      P2 (urgent):  enqueue_front() direct call from GUI thread → appendleft
      P1 (critical): flush_and_abort() + enqueue_front() direct calls from GUI thread
                      → clears deque, sets _abort_current, adds P1 task

    Threading: flush_and_abort() and enqueue_front() are called directly from the GUI
    thread. CPython GIL makes bool writes and deque.clear()/appendleft() safe without
    an explicit lock. _drain_deque() always runs in the GeminiWorker thread.

    Signal order per segment:
    1. signal_priority  — emitted as soon as P-line is parsed
    2. signal_broadcast — emitted immediately after (text already available)
    """
    # (start_t, stop_t, priority, should_speak)
    signal_priority = QtCore.Signal(float, float, int, bool)
    # (start_t, stop_t, result_dict)
    signal_broadcast = QtCore.Signal(float, float, object)
    signal_error = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._initialized = False
        self._busy = False
        self._task_deque: deque = deque()
        self._abort_current: bool = False

    @QtCore.Slot()
    def initialize(self) -> None:
        """Initialize the Gemini client (call once after moving to thread)."""
        try:
            from ..core.models.gemini_broadcaster import _get_client
            _get_client()
            self._initialized = True
            logging.info("[GeminiWorker] Ready")
        except Exception as e:
            logging.exception("[GeminiWorker] Initialization failed")
            self.signal_error.emit(str(e))

    # ------------------------------------------------------------------ #
    #  Public interface — P3 path (via Signal / QueuedConnection)          #
    # ------------------------------------------------------------------ #

    @QtCore.Slot(float, float, object)
    def process_segment(self, start_t: float, stop_t: float, data: object) -> None:
        """P3 normal path — called via QueuedConnection, runs in worker thread."""
        if not self._initialized:
            self.signal_error.emit("GeminiWorker not initialized")
            return

        if not isinstance(data, dict):
            enqueue_ts = time.time()
            self.signal_priority.emit(start_t, stop_t, 1, False)
            result = {
                "priority": 1,
                "broadcast_text": str(data),
                "action_label": "raw",
                "should_speak": False,
                "_enqueue_ts": enqueue_ts,
            }
            self.signal_broadcast.emit(start_t, stop_t, result)
            return

        enqueue_ts = time.time()
        self._task_deque.append((start_t, stop_t, data, enqueue_ts))
        depth = len(self._task_deque)
        if depth > 1:
            logging.warning(
                "[GeminiWorker] Queue depth=%d — segment %.2f-%.2f is waiting",
                depth, start_t, stop_t,
            )
        if not self._busy:
            self._drain_deque()

    # ------------------------------------------------------------------ #
    #  Public interface — P1/P2 paths (direct call from GUI thread)        #
    # ------------------------------------------------------------------ #

    def flush_and_abort(self) -> None:
        """P1 — called directly from GUI thread. Clears all pending tasks and
        signals the in-progress stream to abort at its next yield point."""
        dropped = len(self._task_deque)
        self._task_deque.clear()
        self._abort_current = True
        logging.info("[GeminiWorker] Flushed deque (dropped=%d) and set abort flag (P1 event)", dropped)

    def enqueue_front(self, start_t: float, stop_t: float, data: object) -> None:
        """P1/P2 — called directly from GUI thread. Inserts task at deque front.
        If drain is not running, schedules it via QueuedConnection so it executes
        in the worker thread (not the GUI thread)."""
        enqueue_ts = time.time()
        self._task_deque.appendleft((start_t, stop_t, data, enqueue_ts))
        if not self._busy:
            QtCore.QMetaObject.invokeMethod(
                self, "_start_drain", QtCore.Qt.QueuedConnection
            )

    # ------------------------------------------------------------------ #
    #  Internal drain machinery — runs in worker thread only              #
    # ------------------------------------------------------------------ #

    @QtCore.Slot()
    def _start_drain(self) -> None:
        """Queued-connection trampoline so _drain_deque always runs in worker thread."""
        if not self._busy:
            self._drain_deque()

    def _drain_deque(self) -> None:
        """Consume all queued tasks. Runs entirely in the worker QThread."""
        self._busy = True
        try:
            while self._task_deque:
                start_t, stop_t, data, enqueue_ts = self._task_deque.popleft()
                self._abort_current = False
                self._process_one(start_t, stop_t, data, enqueue_ts)
        finally:
            self._busy = False

    def _process_one(self, start_t: float, stop_t: float, data: dict, enqueue_ts: float) -> None:
        """Stream one segment through Gemini, emitting signals as fields arrive.
        Checks _abort_current at every yield point for fast P1 cancellation."""
        priority_emitted = False
        broadcast_emitted = False
        wait_time = time.time() - enqueue_ts
        if wait_time > 0.5:
            logging.warning(
                "[GeminiWorker] segment %.2f-%.2f waited %.2fs in queue before processing",
                start_t, stop_t, wait_time,
            )
        api_start = time.time()
        raw_visual = data.get("metadata", {}).get("raw") or data.get("event", "")
        try:
            from ..core.models.gemini_broadcaster import get_language, stream_gemini
            from ..core.models.broadcast_grounding import ground_broadcast_text
            for ev in stream_gemini(data):
                if self._abort_current:
                    logging.info(
                        "[GeminiWorker] Aborted mid-stream %.2f-%.2f", start_t, stop_t
                    )
                    return

                if not priority_emitted and ev.priority is not None:
                    ev.should_speak = ev.priority <= 4
                    self.signal_priority.emit(
                        start_t, stop_t, ev.priority, bool(ev.should_speak)
                    )
                    priority_emitted = True

                if not broadcast_emitted and ev.broadcast_text is not None:
                    ev.broadcast_text = ground_broadcast_text(
                        ev.broadcast_text, raw_visual, language=get_language()
                    )
                    result = ev.to_dict()
                    result["should_speak"] = bool(ev.should_speak)
                    result["_enqueue_ts"] = enqueue_ts
                    self.signal_broadcast.emit(start_t, stop_t, result)
                    broadcast_emitted = True

            api_time = time.time() - api_start
            total_latency = time.time() - enqueue_ts
            logging.info(
                "[GeminiWorker] segment %.2f-%.2f done — wait=%.2fs api=%.2fs total=%.2fs | priority=%s broadcast=%s",
                start_t, stop_t, wait_time, api_time, total_latency,
                priority_emitted, broadcast_emitted,
            )
        except Exception as e:
            logging.exception("[GeminiWorker] stream_gemini failed")
            self.signal_error.emit(str(e))


class GeminiBackgroundWorker(QtCore.QObject):
    """
    Continuously generates background broadcast commentary (slow blade).

    Backpressure control: fires a new Gemini call only when TTS queue remaining
    time drops below WATERMARK_SEC (1.0s). Prevents unbounded queue growth when
    Gemini API (~300-500ms/call) outpaces TTS playback (~3-4s/sentence).

    Pause/resume: P1 events call pause() to halt the loop; after the mandatory
    0.5s post-interrupt silence, resume() re-enables it.

    Thread safety: _paused and _abort_current are bool flags (CPython GIL-safe
    for single assignment). _context_pool writes arrive via update_context() Slot
    through signal_livecc_context (QueuedConnection) — no lock needed.
    """

    signal_broadcast = QtCore.Signal(float, float, object)
    signal_error = QtCore.Signal(str)

    WATERMARK_SEC = 1.0       # trigger next call when TTS remaining < this
    POLL_INTERVAL_MS = 200    # polling interval while watermark not reached
    INTER_SENTENCE_MS = 500   # silence injected between consecutive sentences
    MIN_FIRE_INTERVAL_SEC = 8.0  # aggregate enough temporal context; avoids repetitive play calls

    def __init__(self, get_remaining_sec_fn, parent=None) -> None:
        super().__init__(parent)
        # Callable resolving the *currently active* TTS engine's remaining queue
        # time — the active engine can change at runtime (tts_mode dropdown), so
        # a fixed worker reference would watch the wrong queue and mis-pace firing.
        self._get_remaining_sec = get_remaining_sec_fn
        self._stop_requested: bool = False
        self._abort_current: bool = False
        self._paused: bool = False
        self._initialized: bool = False
        self._context_pool: deque = deque(maxlen=4)
        self._context_version: int = 0
        self._last_fired_context_version: int = -1
        self._last_fire_t: float = 0.0
        self._poll_timer: Optional[QtCore.QTimer] = None

    @QtCore.Slot()
    def initialize(self) -> None:
        try:
            from ..core.models.gemini_broadcaster import _get_client
            _get_client()
            self._initialized = True
            logging.info("[GeminiBackgroundWorker] Ready")
        except Exception as e:
            logging.exception("[GeminiBackgroundWorker] Initialization failed")
            self.signal_error.emit(str(e))

    @QtCore.Slot()
    def run_background_loop(self) -> None:
        """Start non-blocking polling in the worker thread.

        A permanent while/msleep loop here used to starve this object's queued
        update_context(), pause(), and resume() slots.  It also fired once with
        empty "Game in progress" context and consumed the first 8-second
        throttle window.  QTimer keeps the thread event loop available.
        """
        self._stop_requested = False
        self._paused = False
        self._last_fire_t = 0.0
        self._last_fired_context_version = -1
        if self._poll_timer is None:
            self._poll_timer = QtCore.QTimer(self)
            self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
            self._poll_timer.timeout.connect(self._poll_once)
        else:
            self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
        self._poll_timer.start()
        logging.info("[GeminiBackgroundWorker] Background loop started")

    @QtCore.Slot()
    def _poll_once(self) -> None:
        if self._stop_requested:
            if self._poll_timer is not None:
                self._poll_timer.stop()
            logging.info("[GeminiBackgroundWorker] Background loop stopped")
            return
        if self._paused or not self._initialized:
            return
        # Never spend the first throttle window on synthetic empty context.
        if not self._context_pool:
            return
        # Do not rebroadcast unchanged context every eight seconds.
        if self._context_version == self._last_fired_context_version:
            return

        now = time.time()
        if now - self._last_fire_t < self.MIN_FIRE_INTERVAL_SEC:
            return
        remaining = self._get_remaining_sec()
        if remaining > self.WATERMARK_SEC:
            return

        self._abort_current = False
        context = self._build_context()
        t_now = time.time()
        self._last_fire_t = t_now
        self._last_fired_context_version = self._context_version

        final_ev = None
        try:
            from ..core.models.gemini_broadcaster import stream_gemini
            for ev in stream_gemini(context):
                if self._abort_current or self._stop_requested:
                    logging.info("[GeminiBackgroundWorker] Stream aborted mid-way")
                    break
                if ev.broadcast_text:
                    final_ev = ev
            if final_ev and not self._abort_current and not self._stop_requested:
                from ..core.models.broadcast_grounding import ground_broadcast_text
                from ..core.models.gemini_broadcaster import get_language
                final_ev.broadcast_text = ground_broadcast_text(
                    final_ev.broadcast_text or "",
                    context.get("event", ""),
                    language=get_language(),
                )
                result = final_ev.to_dict()
                result["_enqueue_ts"] = t_now
                result["should_speak"] = bool(final_ev.should_speak)
                result["_background"] = True
                self.signal_broadcast.emit(t_now, t_now, result)
        except Exception as e:
            logging.exception("[GeminiBackgroundWorker] stream_gemini error")
            self.signal_error.emit(str(e))

    def _build_context(self) -> dict:
        from ..core.models.gemini_broadcaster import _get_match_state
        recent = list(self._context_pool)
        descriptions: list[str] = []
        latest_frames: list[bytes] = []
        for item in recent:
            if isinstance(item, dict):
                metadata = item.get("metadata", {})
                description = metadata.get("raw") or item.get("event", "")
                frames = metadata.get("actor_frames_jpeg") or []
                if isinstance(frames, (list, tuple)) and frames:
                    latest_frames = [frame for frame in frames if isinstance(frame, bytes)]
            else:
                description = str(item)
            if description:
                descriptions.append(description)
        event_text = " ".join(descriptions) if descriptions else "Game in progress."
        context = {
            "event": event_text,
            "metadata": {"raw": event_text},
            # Shared 0:0-opening suppression (see _get_match_state) so background
            # commentary doesn't recite a bogus scoreline before anyone scores.
            "match_state": _get_match_state(),
        }
        if latest_frames:
            context["metadata"]["actor_frames_jpeg"] = latest_frames[:3]
        return context

    @QtCore.Slot()
    def pause(self) -> None:
        self._abort_current = True
        self._paused = True
        logging.info("[GeminiBackgroundWorker] Paused by P1 interrupt")

    @QtCore.Slot()
    def resume(self) -> None:
        self._paused = False
        logging.info("[GeminiBackgroundWorker] Resumed after P1 silence")

    @QtCore.Slot(object)
    def update_context(self, description: object) -> None:
        """Receive LiveCC P3 description via signal_livecc_context (QueuedConnection)."""
        self._context_pool.append(description)
        self._context_version += 1

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._abort_current = True
        self._stop_requested = True
        self._context_pool.clear()
        if (
            self._poll_timer is not None
            and QtCore.QThread.currentThread() is self.thread()
        ):
            self._poll_timer.stop()
