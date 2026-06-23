# src/shared_memory.py
import time
import logging
from multiprocessing import Lock, Array
from typing import Tuple

class AtomicSharedMemory:
    def __init__(self, shared_array: Array, lock: Lock, min_write_interval: float = 0.010):
        self._shared_array = shared_array
        self._lock = lock
        self.min_write_interval = min_write_interval  # Throttling gate (10ms)
        self.last_write_time = 0.0

    def write_metrics(self, valence: float, arousal: float, detection_confidence: float, tracking_lost: bool,
                      is_yn_question: bool = False, is_wh_question: bool = False, is_negation: bool = False,
                      force: bool = False) -> bool:
        now = time.monotonic()

        # Throttle writes unless explicit critical flags (like tracking_lost changes) force it
        if not force and (now - self.last_write_time < self.min_write_interval):
            return False  # Skip write to prevent lock starvation

        try:
            tracking_flag_float = 1.0 if tracking_lost else 0.0
            yn_float = 1.0 if is_yn_question else 0.0
            wh_float = 1.0 if is_wh_question else 0.0
            neg_float = 1.0 if is_negation else 0.0
            with self._lock:
                self._shared_array[0] = valence
                self._shared_array[1] = arousal
                self._shared_array[2] = detection_confidence
                self._shared_array[3] = tracking_flag_float
                self._shared_array[4] = yn_float
                self._shared_array[5] = wh_float
                self._shared_array[6] = neg_float

            self.last_write_time = now
            return True
        except Exception as e:
            logging.error(f"Failed atomic write: {e}")
            return False

    def read_metrics(self) -> Tuple[float, float, float, bool, bool, bool, bool]:
        with self._lock:
            val = self._shared_array[0]
            ars = self._shared_array[1]
            cnf = self._shared_array[2]
            lst = self._shared_array[3] == 1.0
            yn  = self._shared_array[4] == 1.0
            wh  = self._shared_array[5] == 1.0
            neg = self._shared_array[6] == 1.0
        return val, ars, cnf, lst, yn, wh, neg
