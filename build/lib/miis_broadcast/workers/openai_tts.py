# src/miis_broadcast/workers/openai_tts.py
from PySide6 import QtCore

from ..core.models.openai_tts import (
    start_tts_system,
    stop_tts_system,
    enqueue_tts_text,
    print_tts_stats,
    set_tts_voice,
    set_tts_speed,
    interrupt_tts,
)

class OpenAITTSWorker(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._started = False

    @QtCore.Slot(str, float)
    def apply_settings(self, voice: str, speed: float) -> None:
        set_tts_voice(voice)
        set_tts_speed(speed)

    @QtCore.Slot()
    def start(self) -> None:
        if not self._started:
            start_tts_system()
            self._started = True

    @QtCore.Slot(str)
    def speak(self, text: str) -> None:
        enqueue_tts_text(text, drop_outdated=True)

    @QtCore.Slot()
    def interrupt(self) -> None:
        interrupt_tts()

    @QtCore.Slot()
    def stop(self) -> None:
        stop_tts_system()
        print_tts_stats()
        self._started = False
