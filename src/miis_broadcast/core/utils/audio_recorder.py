import time
import wave
import threading
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16

# Maximum silence to pad in one shot (avoid huge gaps if recorder was idle)
_MAX_SILENCE_SEC = 5.0


class AudioRecorder:
    """Thread-safe WAV recorder.

    Hooks into TTS output to capture audio. Silence is automatically padded
    between speech chunks so the WAV duration matches the actual broadcast time.
    """

    def __init__(self, recordings_dir: str = "logs/recordings"):
        self._dir = Path(recordings_dir)
        self._lock = threading.Lock()
        self._wav: Optional[wave.Wave_write] = None
        self._path: Optional[str] = None
        self._start_time: float = 0.0
        self._samples_written: int = 0

    def start(self, mode: str = "") -> str:
        """Open a new WAV file and begin recording. Returns the file path."""
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = mode.replace(" ", "_").replace("+", "plus") if mode else "session"
        path = self._dir / f"{safe_mode}_{ts}.wav"
        with self._lock:
            if self._wav is not None:
                self._finalize_locked()
            w = wave.open(str(path), "wb")
            w.setnchannels(CHANNELS)
            w.setsampwidth(SAMPLE_WIDTH)
            w.setframerate(SAMPLE_RATE)
            self._wav = w
            self._path = str(path)
            self._start_time = time.monotonic()
            self._samples_written = 0
        print(f"[AudioRecorder] 開始錄音: {path}")
        return str(path)

    def write_chunk(self, data) -> None:
        """Write a chunk of audio, padding silence to match real elapsed time.

        Accepts np.ndarray (int16) or bytes.
        """
        now = time.monotonic()
        with self._lock:
            if self._wav is None:
                return

            # Pad silence for the gap since recording started / last chunk
            self._pad_silence_locked(now)

            if isinstance(data, np.ndarray):
                audio_bytes = np.ascontiguousarray(data, dtype=np.int16).tobytes()
            elif isinstance(data, (bytes, bytearray)):
                audio_bytes = bytes(data)
            else:
                return

            try:
                self._wav.writeframes(audio_bytes)
                self._samples_written += len(audio_bytes) // SAMPLE_WIDTH
            except Exception:
                pass

    def stop(self) -> Optional[str]:
        """Finalize and close the WAV file. Returns the saved file path."""
        now = time.monotonic()
        with self._lock:
            if self._wav is None:
                return None
            # Pad trailing silence so duration matches broadcast length
            self._pad_silence_locked(now)
            return self._finalize_locked()

    def _pad_silence_locked(self, now: float) -> None:
        """Write silence to cover elapsed time not yet filled with audio."""
        elapsed = now - self._start_time
        expected = int(elapsed * SAMPLE_RATE)
        gap = expected - self._samples_written
        if gap <= 0:
            return
        # Cap to avoid huge files if recorder was left idle
        gap = min(gap, int(_MAX_SILENCE_SEC * SAMPLE_RATE))
        try:
            self._wav.writeframes(bytes(gap * SAMPLE_WIDTH))
            self._samples_written += gap
        except Exception:
            pass

    def _finalize_locked(self) -> Optional[str]:
        if self._wav is None:
            return None
        path = self._path
        try:
            self._wav.close()
        except Exception:
            pass
        self._wav = None
        self._path = None
        print(f"[AudioRecorder] 錄音結束，儲存至: {path}")
        return path

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._wav is not None
