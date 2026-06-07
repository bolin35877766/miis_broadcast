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
    set_tts_voice,
    set_natural_completion_callback,
)


class GeminiTTSWorker(QtCore.QObject):
    signal_tts_done = QtCore.Signal()  # emitted ONLY on natural completion

    def __init__(self, parent=None):
        super().__init__(parent)
        self._started = False
        self._interrupted: bool = False

        self._queue_remaining_lock = threading.Lock()
        self._queue_remaining_sec: float = 0.0
        self._speak_start_wall: float = 0.0

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

    def get_queue_remaining_sec(self) -> float:
        """Thread-safe remaining TTS time for GeminiBackgroundWorker backpressure."""
        with self._queue_remaining_lock:
            est = self._queue_remaining_sec
        if est <= 0.0 or self._speak_start_wall <= 0.0:
            return 0.0
        elapsed = time.time() - self._speak_start_wall
        return max(0.0, est - elapsed)

    def _on_core_tts_complete(self) -> None:
        """Called from the core TTS thread on natural completion."""
        with self._queue_remaining_lock:
            self._queue_remaining_sec = 0.0
        self._speak_start_wall = 0.0
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
            self._started = True

    @QtCore.Slot(str, int, float, float)
    def speak(self, text: str, priority: int = 5, ref_ts: float = 0.0, start_t: float = 0.0) -> None:
        self._interrupted = False
        est = self._estimate_tts_duration(text)
        with self._queue_remaining_lock:
            self._queue_remaining_sec = est
        self._speak_start_wall = time.time()
        # P1/P2 (urgent): replace queue immediately. P3-P5 (routine commentary):
        # bounded FIFO so continuous LiveCC-driven broadcast keeps flowing to TTS
        # instead of being discarded by the next arrival before it's ever spoken.
        enqueue_tts_text(text, ref_ts=ref_ts, drop_outdated=(priority <= 2), priority=priority, start_t=start_t)

    @QtCore.Slot()
    def interrupt(self) -> None:
        self._interrupted = True
        with self._queue_remaining_lock:
            self._queue_remaining_sec = 0.0
        self._speak_start_wall = 0.0
        interrupt_tts()

    @QtCore.Slot()
    def stop(self) -> None:
        stop_tts_system()
        self._started = False
