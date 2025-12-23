import time
from collections import deque

import numpy as np

class FPSCounter:
    def __init__(self, max_frametimes=30) -> None:
        self.max_frametimes = max_frametimes
        self.frametimes = deque(maxlen=self.max_frametimes)
        self.sum = 0 # sum of all values
        self.val = 0 # last value

    def update(self, val: float) -> None:
        """ Add the latest frametime. 
        Will pop the queue if it exceeded max_frametimes.
        
        Args:
            val (float): The latest frametime.
        """
        val = float(val)
        val = val if val > 0 else 1e-3
        if len(self.frametimes) >= self.max_frametimes:
            self.sum -= self.frametimes.popleft()
        if len(self.frametimes) < self.max_frametimes:
            self.sum += val
            self.frametimes.append(val)
        self.val = val

    def calculateFPS(self) -> float:
        return len(self.frametimes) / (self.sum + 1e-8)

    def calculateMean(self) -> float:
        """ Calculate the simple moving average.
        """
        if len(self.frametimes):
            return self.sum / len(self.frametimes)
        else:
            return 0.

    def calculateWeightedMean(self) -> float:
        """ Calculate the weighted moving average.
        The weight is linear with `step = 1` starting from 1 to the
        length of the queue + 1, where the latest value has the largest
        weight.
        """
        if len(self.frametimes) > 1:
            weight = np.arange(1, len(self.frametimes) + 1)
            return (self.frametimes * weight).sum() / weight.sum()
        elif len(self.frametimes) == 1:
            return self.val
        else:
            return 0.

    def reset(self) -> None:
        self.frametimes.clear()
        self.sum = 0

class FPSCounterTimestamp(FPSCounter):
    def __init__(self, max_frametimes=20) -> None:
        super().__init__(max_frametimes=max_frametimes)
        self.last_timestamp = None

    def update(self):
        if self.last_timestamp is None:
            self.last_timestamp = time.perf_counter()
        
        current_timestamp = time.perf_counter()
        val = current_timestamp - self.last_timestamp
        super().update(val)
        self.last_timestamp = current_timestamp

    def reset(self) -> None:
        self.last_timestamp = None
        super().reset()