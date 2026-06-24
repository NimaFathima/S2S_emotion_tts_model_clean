"""
nmm_classifier.py
Nima — Signet Aid Vision Producer, Component 4

Responsibilities:
  - Load MediaPipe FaceLandmarker (face module only, NOT Holistic)
  - Extract 478-point face mesh from the full CLAHE-enhanced frame
  - Compute geometric ratios for eyebrow raise, eyebrow furrow, negation
  - Return NMMContext flags: is_yn_question, is_wh_question, is_negation
  - Apply NMM dampening to raw VA scores

NOTE: MediaPipe may log "Failed to send to clearcut: Status_ConnectFailed"
  This is harmless internal telemetry attempting to reach Google services.
  It has NO impact on inference accuracy or performance.
  Safe to ignore in offline / air-gapped environments.
  There is no officially supported suppression mechanism.
"""

import os
import sys
import cv2
import numpy as np
import logging
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import Optional
import urllib.request
from pathlib import Path
from config.settings import (
    BROW_RAISE_THRESHOLD,
    BROW_FURROW_THRESHOLD,
    HEAD_YAW_THRESHOLD,
    NMM_DAMPEN_ALPHA,
    NMM_CALIBRATION_FRAMES,
    AFFECT_CONFOUND_THRESHOLD,
    TEMPORAL_GATE,
    TEMPORAL_GATE_STABLE_FRAMES,
)
from processes.brow_gate import BrowTemporalGate

# Use CPU-only MediaPipe by default; can be overridden via env var before import.
if os.getenv("MEDIAPIPE_DISABLE_GPU") is None:
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

log = logging.getLogger(__name__)

# ── Landmark Indices ──────────────────────────────────────────────────────────
# UPPER ARC ONLY — 5 points furthest from eye, highest vertical displacement
RIGHT_BROW_UPPER  = [46, 53, 52, 65, 55]
LEFT_BROW_UPPER   = [276, 283, 282, 295, 285]

# Eye centre landmarks for normalisation reference
RIGHT_EYE_CENTRE  = 468   # or use average of [33, 133]
LEFT_EYE_CENTRE   = 473   # or use average of [362, 263]
RIGHT_EYE_REF     = [33, 133]    # inner and outer right eye corners
LEFT_EYE_REF      = [362, 263]   # inner and outer left eye corners

# Nose tip for head pose reference (yaw estimation)
NOSE_TIP          = 4
NOSE_BRIDGE       = 168


@dataclass
class NMMContext:
    is_yn_question:  bool = False   # raised brows (brow-isolated) → Y/N question
    is_wh_question:  bool = False   # furrowed brows (brow-isolated) → who/what/where
    is_negation:     bool = False   # furrowed + head shake → negation
    any_active:      bool = False   # convenience flag
    brow_affective:  bool = False   # brow crossed threshold but whole face moved
                                    # → treat as EMOTION, not grammar


class NMMClassifier:
    """
    Stateful NMM geometry classifier.
    Maintains a small history of nose positions for head shake detection.

    Per-session baseline calibration (Issue 2):
      For the first NMM_CALIBRATION_FRAMES valid frames, collects neutral
      eyebrow raise ratios to compute a per-user baseline. During this
      calibration window, all NMM flags are suppressed (returns empty
      NMMContext). After calibration, thresholds are applied *relative*
      to the baseline, eliminating false positives from users whose
      natural resting brow position exceeds the absolute threshold.
    """

    def __init__(self):
        self.dummy_mode = False

        # ── Per-session baseline calibration state (Issue 2) ──────────────
        self._baseline_raise = None             # calibrated neutral brow position
        self._calibration_buf = []              # buffer collecting raise ratios
        self._calibration_frames = NMM_CALIBRATION_FRAMES

        # Warn only once if blendshapes are unavailable (concordance gate then
        # degrades to the old geometry-only behaviour).
        self._blendshape_warned = False

        # Grammatical-vs-affective brow decision (pure/testable). temporal=False
        # reproduces the single-frame gate exactly; True adds flicker-suppressing
        # hysteresis. See processes/brow_gate.py and config TEMPORAL_GATE.
        self._gate = BrowTemporalGate(
            BROW_RAISE_THRESHOLD, BROW_FURROW_THRESHOLD, AFFECT_CONFOUND_THRESHOLD,
            temporal=TEMPORAL_GATE, stable_frames=TEMPORAL_GATE_STABLE_FRAMES,
        )

        try:
            _download_model_if_missing()

            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                # Blendshapes are required for the grammatical-vs-affective gate:
                # they expose browInnerUp/jawOpen/eyeWide/etc. so we can measure
                # whether a brow movement is brow-isolated (grammatical) or part
                # of a whole-face emotional expression (affective).
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=mp_vision.RunningMode.IMAGE
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._nose_x_history = []   # rolling nose x for head shake detection
            log.info("MediaPipe FaceLandmarker loaded.")
        except Exception as e:
            log.error(f"Failed to load MediaPipe model: {e}")
            self.dummy_mode = True
            log.warning("Using dummy NMM classifier (no model available)")

    def _get_landmark_ys(self, landmarks, indices: list) -> np.ndarray:
        """Return y-coordinates (normalised 0-1) for a list of landmark indices."""
        return np.array([landmarks[i].y for i in indices])

    def _get_landmark_xs(self, landmarks, indices: list) -> np.ndarray:
        return np.array([landmarks[i].x for i in indices])

    def _confound_scores(self, result) -> Optional[tuple]:
        """
        Measure the EMOTION-specific co-markers that disambiguate a grammatical
        brow movement from an affective one with the same brow position.

        A Y/N raise and SURPRISE/FEAR share the raised brow, but surprise/fear
        additionally widen the eyes and drop the jaw. A WH furrow and ANGER
        share the lowered brow, but anger additionally sneers the nose, presses
        the lips and squints the eyes. Grammar recruits none of these; a smile
        recruits none of the surprise markers (so a smiling Y/N is preserved).

        Returns:
            (surprise_activity, anger_activity) in [0, 1], or None if
            blendshapes are unavailable (caller falls back to geometry-only).
        """
        if not getattr(result, "face_blendshapes", None):
            if not self._blendshape_warned:
                log.warning(
                    "Blendshapes unavailable — grammatical/affective brow gate "
                    "degraded to geometry-only. Ensure output_face_blendshapes=True "
                    "and the float16 face_landmarker.task bundle is in use."
                )
                self._blendshape_warned = True
            return None

        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        surprise = np.mean([
            bs.get("eyeWideLeft", 0.0),
            bs.get("eyeWideRight", 0.0),
            bs.get("jawOpen", 0.0),
        ])
        anger = np.mean([
            bs.get("noseSneerLeft", 0.0),
            bs.get("noseSneerRight", 0.0),
            bs.get("mouthPressLeft", 0.0),
            bs.get("mouthPressRight", 0.0),
            bs.get("eyeSquintLeft", 0.0),
            bs.get("eyeSquintRight", 0.0),
        ])
        return float(surprise), float(anger)

    def _get_raise_ratio(self, landmarks, face_height_ref: float) -> float:
        """
        Compute the average normalised eyebrow raise ratio.

        In MediaPipe normalised coords, y increases DOWNWARD.
        A raised brow has SMALLER y than the eye centre.
        brow_raise_ratio = (eye_y - brow_y) / face_height_ref
        Larger positive value = more raised.

        Args:
            landmarks: MediaPipe face landmarks (478-point mesh).
            face_height_ref: Reference face height for scale-invariant normalisation.

        Returns:
            Average raise ratio across both brows.
        """
        right_eye_y = np.mean(self._get_landmark_ys(landmarks, RIGHT_EYE_REF))
        left_eye_y  = np.mean(self._get_landmark_ys(landmarks, LEFT_EYE_REF))

        right_brow_y = np.mean(self._get_landmark_ys(landmarks, RIGHT_BROW_UPPER))
        left_brow_y  = np.mean(self._get_landmark_ys(landmarks, LEFT_BROW_UPPER))

        right_raise = (right_eye_y - right_brow_y) / face_height_ref
        left_raise  = (left_eye_y  - left_brow_y)  / face_height_ref

        return (right_raise + left_raise) / 2.0

    def classify(self, clahe_frame_bgr: np.ndarray) -> NMMContext:
        """
        Detect NMM grammar flags from a full CLAHE-enhanced BGR frame.

        Args:
            clahe_frame_bgr: Full-resolution BGR frame, CLAHE-normalised.

        Returns:
            NMMContext with grammar flags.
        """
        if self.dummy_mode:
            # Return no NMM flags in dummy mode
            return NMMContext()

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(clahe_frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            # No face detected — return no flags
            # Do NOT add to calibration buffer (invalid frame)
            self._gate.reset()   # clear hysteresis so no stale decision carries over
            return NMMContext()

        landmarks = result.face_landmarks[0]   # first (and only) face
        h, w = clahe_frame_bgr.shape[:2]

        # ── Compute face height reference (scale-invariant normalisation) ─────
        face_height_ref = abs(landmarks[NOSE_TIP].y - landmarks[10].y)  # nose to forehead
        face_height_ref = max(face_height_ref, 0.01)   # avoid division by zero

        # ── Compute raise ratio via extracted method ──────────────────────────
        avg_raise = self._get_raise_ratio(landmarks, face_height_ref)

        # ── Calibration phase (Issue 2) ───────────────────────────────────────
        # For the first N valid frames, collect neutral brow positions.
        # Suppress all NMM outputs during this window to prevent false positives
        # from contaminating the initial emotion readings.
        if self._baseline_raise is None:
            self._calibration_buf.append(avg_raise)

            if len(self._calibration_buf) >= self._calibration_frames:
                self._baseline_raise = float(np.mean(self._calibration_buf))
                log.info(
                    f"NMM baseline calibrated: raise_ratio={self._baseline_raise:.4f} "
                    f"({len(self._calibration_buf)} frames)"
                )
                # Free calibration buffer memory
                self._calibration_buf = []

            # During calibration: return no NMM flags
            return NMMContext()

        # ── Confound gate: grammatical vs affective brow movement ────────────
        # The decision lives in BrowTemporalGate (pure/testable). It disambiguates
        # by the emotion-specific co-markers, per direction:
        #   raise  is grammatical unless SURPRISE/FEAR markers are present
        #   furrow is grammatical unless ANGER markers are present
        # (surprise/anger None → blendshapes unavailable → assume grammatical).
        scores = self._confound_scores(result)
        surprise_act, anger_act = (None, None) if scores is None else scores

        # RELATIVE raise: subtract calibrated baseline so only intentional brow
        # movements trigger, not the signer's natural resting position.
        relative_raise = avg_raise - self._baseline_raise
        is_yn_question, is_wh_question, brow_affective = self._gate.update(
            relative_raise, surprise_act, anger_act)

        # ── 3. Head shake detection (negation) ───────────────────────────────
        nose_x = landmarks[NOSE_TIP].x
        self._nose_x_history.append(nose_x)
        if len(self._nose_x_history) > 15:
            self._nose_x_history.pop(0)

        is_negation = False
        if len(self._nose_x_history) >= 6:
            # A head shake is lateral oscillation of the nose x position:
            # multiple direction reversals plus sufficient lateral range.
            diffs = np.diff(self._nose_x_history)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            nose_x_range = max(self._nose_x_history) - min(self._nose_x_history)
            head_shake = (sign_changes >= 2 and
                          nose_x_range > (HEAD_YAW_THRESHOLD / 100.0))
            # Lexical negation ("not/never/no/cannot") is owned by the text
            # channel; this face signal is a CONFIRMATION. We only require a
            # clean head shake that is not part of a Y/N affirmation brow-raise.
            # (Previous logic `is_wh_question or avg_raise < 0` conflated WH
            # furrows and absolute brow position with negation.)
            is_negation = head_shake and not is_yn_question

        any_active = is_yn_question or is_wh_question or is_negation

        return NMMContext(
            is_yn_question=is_yn_question,
            is_wh_question=is_wh_question,
            is_negation=is_negation,
            any_active=any_active,
            brow_affective=brow_affective
        )

    def apply_dampening(self,
                        valence: float,
                        arousal: float,
                        nmm: NMMContext) -> tuple:
        """
        Conditionally dampen VA based on the grammatical-vs-affective decision.

        - brow_affective: the brow moved AS PART OF a whole-face emotion. Keep
          the emotion intact — it is the signer's tone, not grammar. (This is
          what lets an angry WH-question or excited Y/N keep its affect.)
        - any_active (and not affective): a genuine, brow-ISOLATED grammatical
          marker. The lower face is quiet, so HSEmotion's whole-face reading is
          driven mainly by the grammatical brow itself — dampen it toward
          neutral so the marker is not mistaken for emotion.
        - otherwise: no NMM, pass VA through unchanged.

        Formula (grammatical case):
            effective = value * (1 - NMM_DAMPEN_ALPHA)

        Returns:
            (effective_valence, effective_arousal)
        """
        # Affective brow → preserve emotion (brow_affective and any_active are
        # mutually exclusive by construction; this is explicit and defensive).
        if nmm.brow_affective:
            return valence, arousal

        if not nmm.any_active:
            return valence, arousal

        dampen = 1.0 - NMM_DAMPEN_ALPHA
        return valence * dampen, arousal * dampen

    def close(self) -> None:
        """Release MediaPipe resources. Call at process shutdown."""
        if not self.dummy_mode:
            self._landmarker.close()


def _download_model_if_missing() -> None:
    if os.path.exists(MODEL_PATH):
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading FaceLandmarker model ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info(f"Model saved to {MODEL_PATH}")


# ── Model path ───────────────────────────────────────────────────────────────
# Canonical model directory is the repo-root `models/` (alongside
# enet_b0_8_va_mtl.onnx etc.), the same dir emotion_inference uses. Resolved
# from __file__ so it is independent of the current working directory:
#   processes/nmm_classifier.py -> processes -> emotion_module -> <repo root>
# (Previously pointed at the non-existent emotion_module/models/, which forced
#  a re-download even though the file was already committed.)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = str(MODEL_DIR / "face_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


if __name__ == "__main__":
    classifier = NMMClassifier()

    # Test with a blank frame — should return no flags
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx = classifier.classify(blank)
    assert not ctx.any_active, "Blank frame should produce no NMM flags"

    # Test dampening
    v, a = classifier.apply_dampening(0.7, 0.5, NMMContext(is_yn_question=True, any_active=True))
    assert abs(v - 0.7 * 0.25) < 0.01, f"Dampening wrong: {v}"
    print(f"Dampened VA: V={v:.4f} A={a:.4f}")
    print("[nmm_classifier] Smoke tests passed.")
    classifier.close()
