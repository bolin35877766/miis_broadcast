import os
import datetime
from pathlib import Path
from typing import Optional

class SessionLogger:
    def __init__(self, log_dir="logs/sessions"):
        """
        Initialize the session logger.
        
        Args:
            log_dir (str): Directory where session logs will be stored.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = None

    def start_new_session(
        self,
        mode_name: str,
        inference_backend: Optional[str] = None,
    ):
        """
        Start a new logging session with a specific mode name.

        Args:
            mode_name: GUI input source mode (e.g. camera, obs, obs_track, file, dual_sync).
            inference_backend: "remote" | "local" — where LiveCC runs (thin client vs local model).
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure filename is safe and unique
        safe_mode = mode_name.replace(" ", "_").replace("+", "plus")
        self.current_log_file = self.log_dir / f"{safe_mode}_{timestamp}.log"
        self._write_header(mode_name, inference_backend)
        return str(self.current_log_file)

    def _write_header(self, mode_name: str, inference_backend: Optional[str] = None):
        """Write session start header."""
        if not self.current_log_file:
            return
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(f"Session Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Mode: {mode_name}\n")
            if inference_backend:
                f.write(f"Inference: {inference_backend}\n")
            f.write("="*50 + "\n\n")

    def log_commentary(self, text: str):
        """
        Log the commentary (TTS output).
        """
        if not self.current_log_file:
            return
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        log_entry = f"{timestamp} [COMMENTARY] {text}\n"
        
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
            
    def log_system(self, source: str, level: str, message: str):
        """
        Log system events.
        
        Args:
            source (str): GUI, LiveCC, TTS, etc.
            level (str): INFO, WARNING, ERROR, etc.
            message (str): The message to log.
        """
        if not self.current_log_file:
            return
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        log_entry = f"{timestamp} [{source}] [{level}] {message}\n"
        
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def get_log_path(self):
        return str(self.current_log_file)
