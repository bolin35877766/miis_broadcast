# src/miis_broadcast/workers/local_tts.py
from PySide6 import QtCore

from ..core.models.chatterbox_tts import (
    start_local_tts_system,
    stop_local_tts_system,
    enqueue_local_tts_text,
    set_local_tts_params,
    interrupt_local_tts,
)

class ChatterboxTTSWorker(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._started = False

    @QtCore.Slot(float, float)
    def apply_settings(self, exaggeration: float, cfg: float) -> None:
        # 將 GUI 的 Exaggeration 對應到 temperature
        # 將 GUI 的 CFG 對應到 cfg_weight
        set_local_tts_params(exaggeration, cfg)

    @QtCore.Slot()
    def start(self) -> None:
        if not self._started:
            start_local_tts_system()
            self._started = True

    @QtCore.Slot(str)
    def speak(self, text: str) -> None:
        enqueue_local_tts_text(text)

    @QtCore.Slot()
    def interrupt(self) -> None:
        interrupt_local_tts()

    @QtCore.Slot()
    def stop(self) -> None:
        stop_local_tts_system()
        self._started = False