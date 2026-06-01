# src/miis_broadcast/core/match_tracker.py

import threading
import logging
from typing import Optional


class MatchTracker:
    """Single source of truth for match state. LLMs read it; only Python writes it.

    Thread-safe: all mutations go through _lock. The singleton is module-level so
    it can be imported anywhere without dependency injection.
    """

    _lock: threading.Lock
    red_score: int
    blue_score: int
    period: int
    last_event: str

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.red_score = 0
        self.blue_score = 0
        self.period = 1
        self.last_event = ""

    def add_score(self, team: str) -> None:
        """Increment score for `team`. team must be 'red' or 'blue'."""
        with self._lock:
            if team == "red":
                self.red_score += 1
                logging.info("[MatchTracker] Red scores — Red %d : Blue %d",
                             self.red_score, self.blue_score)
            elif team == "blue":
                self.blue_score += 1
                logging.info("[MatchTracker] Blue scores — Red %d : Blue %d",
                             self.red_score, self.blue_score)
            else:
                logging.warning("[MatchTracker] Unknown team: %r — score unchanged", team)


    def set_period(self, n: int) -> None:
        with self._lock:
            self.period = max(1, int(n))
        logging.info("[MatchTracker] Period set to %d", self.period)

    def set_last_event(self, event: str) -> None:
        with self._lock:
            self.last_event = event

    def get_state_string(self) -> str:
        """Return a compact string suitable for injection into Gemini prompt."""
        with self._lock:
            event_part = f", last event: {self.last_event}" if self.last_event else ""
            return f"Red {self.red_score} : Blue {self.blue_score}, Q{self.period}{event_part}"

    def get_scores(self) -> tuple[int, int]:
        """Return (red_score, blue_score) snapshot."""
        with self._lock:
            return self.red_score, self.blue_score

    def reset(self) -> None:
        with self._lock:
            self.red_score = 0
            self.blue_score = 0
            self.period = 1
            self.last_event = ""
        logging.info("[MatchTracker] Reset")


# Module-level singleton — import this directly everywhere
match_tracker = MatchTracker()
