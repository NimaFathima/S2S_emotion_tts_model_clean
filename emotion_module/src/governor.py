# src/governor.py
import time
import logging
from typing import Generator, Any
from config.settings import TARGET_FPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class FrameGovernor:
    def __init__(self, target_fps: float = TARGET_FPS):
        if target_fps <= 0:
            raise ValueError("Target FPS must be greater than zero.")
        self.target_interval: float = 1.0 / target_fps
        self.next_frame_time: float = time.monotonic()
        # Throttle "falling behind" warnings to avoid flooding the log.
        # The timing anchor ALWAYS resets when behind — only the log message
        # is suppressed to reduce noise when hardware can't sustain target FPS.
        self._last_behind_warn: float = 0.0
        self._behind_warn_interval: float = 5.0   # warn at most every 5 seconds

    def regulate(self, stream: Generator[Any, None, None]) -> Generator[Any, None, None]:
        self.next_frame_time = time.monotonic()
        for frame in stream:
            now = time.monotonic()
            if now > self.next_frame_time + self.target_interval:
                # Always reset the anchor to prevent drift accumulation,
                # but only log the warning once per _behind_warn_interval
                # to keep the console readable.
                if now - self._last_behind_warn > self._behind_warn_interval:
                    logging.warning("Frame governor falling behind. Resetting timing anchor.")
                    self._last_behind_warn = now
                self.next_frame_time = now

            sleep_time = self.next_frame_time - now
            if sleep_time > 0:
                if sleep_time > 0.003:
                    time.sleep(sleep_time - 0.002)
                # Short benign spin for the last ~2 ms to hit frame time accurately
                # without burning a full core for the entire interval.
                remaining = self.next_frame_time - time.monotonic()
                if remaining > 0:
                    time.sleep(max(0.0, remaining))

            yield frame
            self.next_frame_time += self.target_interval

