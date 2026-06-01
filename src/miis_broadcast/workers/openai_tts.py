# src/miis_broadcast/workers/openai_tts.py
import threading
import time

from PySide6 import QtCore

from ..core.models.openai_tts import (
    start_tts_system,
    stop_tts_system,
    enqueue_tts_text,
    print_tts_stats,
    set_tts_voice,
    set_tts_speed,
    interrupt_tts,
    set_natural_completion_callback,
    _tts_cfg,
    _tts_cfg_lock,
)


class OpenAITTSWorker(QtCore.QObject):
    signal_tts_done = QtCore.Signal()  # emitted ONLY on natural completion, never on interrupt

    def __init__(self, parent=None):
        super().__init__(parent)
        self._started = False
        self._interrupted: bool = False          # guards against spurious signal_tts_done on forced cut

        self._queue_remaining_lock = threading.Lock()
        self._queue_remaining_sec: float = 0.0
        self._speak_start_wall: float = 0.0
        self._speak_est_duration: float = 0.0

    @staticmethod
    def _estimate_tts_duration(text: str, speed: float) -> float:
        """
        CJK-aware TTS duration estimate for backpressure watermark.
        Chinese/Japanese chars: ~4 chars/sec at 1.0x speed.
        English words: ~2.5 words/sec at 1.0x speed.
        """
        cjk_count = sum(
            1 for c in text
            if '一' <= c <= '鿿'   # CJK Unified Ideographs
            or '぀' <= c <= 'ヿ'   # Hiragana / Katakana
        )
        ascii_only = ''.join(' ' if '一' <= c <= '鿿' else c for c in text)
        other_words = len(ascii_only.split())
        estimated = (cjk_count / 4.0 + other_words / 2.5) / max(speed, 0.1)
        return max(estimated, 0.5)  # floor at 0.5s to prevent instant re-fire

    def get_queue_remaining_sec(self) -> float:
        """Thread-safe remaining TTS time estimate for GeminiBackgroundWorker backpressure."""
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
        set_tts_speed(speed)

    @QtCore.Slot()
    def start(self) -> None:
        if not self._started:
            start_tts_system()
            set_natural_completion_callback(self._on_core_tts_complete)
            self._started = True

<<<<<<< HEAD
    @QtCore.Slot(str, int, float, float)
    def speak(self, text: str, priority: int = 5, ref_ts: float = 0.0, start_t: float = 0.0) -> None:
        self._interrupted = False
        with _tts_cfg_lock:
            speed = _tts_cfg.get("speed", 1.0)
        est = self._estimate_tts_duration(text, speed)
        with self._queue_remaining_lock:
            self._queue_remaining_sec = est
        self._speak_start_wall = time.time()
        enqueue_tts_text(text, ref_ts=ref_ts, drop_outdated=True, priority=priority, start_t=start_t)
=======
    @QtCore.Slot(str)
    def speak(self, text: str) -> None:
        # Preserve order vs rapid SEGMENT bursts; see openai_tts realtime worker FIFO path.
        enqueue_tts_text(text, drop_outdated=False)
>>>>>>> Multi-API

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
        print_tts_stats()
        self._started = False
