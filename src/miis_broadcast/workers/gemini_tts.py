# src/miis_broadcast/workers/gemini_tts.py
"""
Qt wrapper for the Gemini TTS backend.
Same public interface as OpenAITTSWorker:
  - signal_tts_done   (only on natural completion)
  - speak / interrupt / start / stop / apply_settings
  - get_queue_remaining_sec()  (for GeminiBackgroundWorker backpressure)
"""
import threading
import time

from PySide6 import QtCore

from ..core.models.gemini_tts import (
    start_tts_system,
    stop_tts_system,
    enqueue_tts_text,
    interrupt_tts,
    soft_interrupt_tts,
    set_tts_voice,
    set_natural_completion_callback,
    set_playback_start_callback,
    print_tts_stats,
)


class GeminiTTSWorker(QtCore.QObject):
    signal_tts_done = QtCore.Signal()  # emitted ONLY on natural completion
    signal_playback_start = QtCore.Signal(object)  # dict payload when first audio chunk plays

    def __init__(self, parent=None):
        super().__init__(parent)
        self._started = False
        self._interrupted: bool = False

        self._queue_remaining_lock = threading.Lock()
        self._queue_remaining_sec: float = 0.0
        self._last_decay_wall: float = 0.0

    @staticmethod
    def _estimate_tts_duration(text: str, speed: float = 1.0) -> float:
        """CJK-aware duration estimate. Gemini TTS doesn't expose speed, uses 1.0."""
        cjk_count = sum(
            1 for c in text
            if '一' <= c <= '鿿' or '぀' <= c <= 'ヿ'
        )
        ascii_only = ''.join(' ' if '一' <= c <= '鿿' else c for c in text)
        other_words = len(ascii_only.split())
        return max((cjk_count / 4.0 + other_words / 2.5) / max(speed, 0.1), 0.5)

    def _decay_locked(self) -> None:
        """Drain the tracked backlog by real elapsed time — ffplay plays audio
        1:1 with the wall clock, so this models the queue draining even though
        we don't track each item's playback individually."""
        now = time.time()
        if self._last_decay_wall > 0.0:
            elapsed = now - self._last_decay_wall
            self._queue_remaining_sec = max(0.0, self._queue_remaining_sec - elapsed)
        self._last_decay_wall = now

    def get_queue_remaining_sec(self) -> float:
        """Thread-safe estimate of total queued+playing audio for
        GeminiBackgroundWorker backpressure. Sums across the bounded FIFO
        (not just the most-recently-enqueued item), so long sentences and a
        non-empty backlog are both reflected."""
        with self._queue_remaining_lock:
            self._decay_locked()
            return self._queue_remaining_sec

    def _on_core_tts_complete(self) -> None:
        """Called from the core TTS thread on natural completion."""
        if not self._interrupted:
            self.signal_tts_done.emit()
        self._interrupted = False

    @QtCore.Slot(str, float)
    def apply_settings(self, voice: str, speed: float) -> None:
        set_tts_voice(voice)

    @QtCore.Slot()
    def start(self) -> None:
        if not self._started:
            start_tts_system()
            set_natural_completion_callback(self._on_core_tts_complete)
            set_playback_start_callback(self._on_core_playback_start)
            self._started = True

    def _on_core_playback_start(self, meta: dict) -> None:
        self.signal_playback_start.emit(meta)

    @QtCore.Slot(str, int, float, float, float, object)
    def speak(
        self,
        text: str,
        priority: int = 5,
        ref_ts: float = 0.0,
        start_t: float = 0.0,
        stop_t: float = 0.0,
        log_meta: object = None,
    ) -> None:
        self._interrupted = False
        est = self._estimate_tts_duration(text)
        with self._queue_remaining_lock:
            self._decay_locked()
            if priority <= 2:
                self._queue_remaining_sec = est
            else:
                self._queue_remaining_sec += est
        self._speak_start_wall = time.time()
        meta = log_meta if isinstance(log_meta, dict) else {}
        enqueue_tts_text(
            text,
            ref_ts=ref_ts,
            drop_outdated=(priority <= 2),
            priority=priority,
            start_t=start_t,
            stop_t=stop_t,
            log_meta=meta,
        )

    @QtCore.Slot()
    def interrupt(self) -> None:
        self._interrupted = True
        with self._queue_remaining_lock:
            self._queue_remaining_sec = 0.0
            self._last_decay_wall = 0.0
        interrupt_tts()

    @QtCore.Slot()
    def soft_interrupt(self) -> None:
        """P1 priority bump: drop pending TTS, keep current sentence playing."""
        soft_interrupt_tts()

    @QtCore.Slot()
    def stop(self) -> None:
        stop_tts_system()
        print_tts_stats()
        self._started = False
