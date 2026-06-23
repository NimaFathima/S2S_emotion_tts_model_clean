"""
coasting_matrix.py
Nima — Signet Aid Vision Producer, Component 2

Responsibilities:
  - Track frames since last valid detection
  - When confidence < 0.65: begin exponential decay toward (0.0, 0.0)
  - When confidence >= 0.65: reset decay counter, return raw VA
  - Output: (effective_valence, effective_arousal, tracking_lost: bool)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Tuple
from config.settings import (
    DETECTION_CONFIDENCE_THRESHOLD,
    COASTING_DECAY_FRAMES,
    TRACKING_LOST_HOLD_FRAMES,
)

log = logging.getLogger(__name__)

# Decay constant: solve e^(-k * COASTING_DECAY_FRAMES) ≈ 0.05
# k = -ln(0.05) / 15 ≈ 0.1996
_DECAY_K = -math.log(0.05) / COASTING_DECAY_FRAMES


@dataclass
class CoastingState:
    frames_since_valid: int   = 0
    last_valid_valence: float = 0.0
    last_valid_arousal: float = 0.0
    is_coasting:        bool  = False


@dataclass
class CoastingResult:
    valence:       float
    arousal:       float
    tracking_lost: bool    # True only after TRACKING_LOST_HOLD_FRAMES exceeded


class CoastingMatrix:
    """
    Stateful coasting filter. One instance lives for the entire session.
    Call update() every frame with the raw detection result.
    """

    def __init__(self):
        self._state = CoastingState()

    def update(self,
               confidence: float,
               raw_valence: float,
               raw_arousal: float) -> CoastingResult:
        """
        Process one frame's detection result and return effective VA.

        Args:
            confidence:  RetinaFace detection confidence (0.0 to 1.0).
                         Pass 0.0 if no face was detected at all.
            raw_valence: Valence from HSEmotion this frame (or 0.0 if no face).
            raw_arousal: Arousal from HSEmotion this frame (or 0.0 if no face).

        Returns:
            CoastingResult with effective VA and tracking_lost flag.
        """
        s = self._state

        if confidence >= DETECTION_CONFIDENCE_THRESHOLD:
            # Valid detection — reset coasting, update last known values
            s.frames_since_valid = 0
            s.last_valid_valence = raw_valence
            s.last_valid_arousal = raw_arousal
            s.is_coasting        = False
            return CoastingResult(
                valence=raw_valence,
                arousal=raw_arousal,
                tracking_lost=False
            )

        # Confidence below threshold — begin or continue coasting
        s.frames_since_valid += 1
        s.is_coasting = True

        # Exponential decay toward (0.0, 0.0)
        # decay_factor goes from ~1.0 at frame 0 to ~0.05 at frame 15
        decay_factor = math.exp(-_DECAY_K * s.frames_since_valid)
        decay_factor = max(0.0, decay_factor)   # clamp, never go negative

        eff_valence = s.last_valid_valence * decay_factor
        eff_arousal = s.last_valid_arousal * decay_factor

        tracking_lost = s.frames_since_valid >= TRACKING_LOST_HOLD_FRAMES

        return CoastingResult(
            valence=eff_valence,
            arousal=eff_arousal,
            tracking_lost=tracking_lost
        )

    def reset(self) -> None:
        """Hard reset — call at session start."""
        self._state = CoastingState()


if __name__ == "__main__":
    cm = CoastingMatrix()

    # Simulate 5 valid frames
    for i in range(5):
        r = cm.update(confidence=0.92, raw_valence=0.6, raw_arousal=0.3)
        assert r.tracking_lost == False
        assert abs(r.valence - 0.6) < 0.01

    # Simulate 30 frames of lost tracking
    for i in range(30):
        r = cm.update(confidence=0.0, raw_valence=0.0, raw_arousal=0.0)
        # Optionally print every 5th frame to see progress
        if (i+1) % 5 == 0:
            print(f"  frame {i+1}: V={r.valence:.4f} A={r.arousal:.4f} lost={r.tracking_lost}")

    # After 30 frames lost, tracking_lost should be True
    assert r.tracking_lost == True
    # Valence and arousal should be very close to zero
    assert abs(r.valence) < 0.01   # after 30 frames, decay factor is 0.05, so 0.6*0.05=0.03 -> we use 0.01 to be safe
    assert abs(r.arousal) < 0.01
    print("[coasting_matrix] All unit tests passed.")
