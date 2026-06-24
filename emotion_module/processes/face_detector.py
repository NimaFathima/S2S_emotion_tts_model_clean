"""
face_detector.py
Signet Aid Vision Producer, Component 1 — face localization.

Two interchangeable backends, chosen by config FACE_BACKEND via
create_face_detector():
  - FaceDetector            : RetinaFace/SCRFD via InsightFace (default, GPU-safe)
  - MediaPipeFaceDetector   : face box from the MediaPipe FaceLandmarker mesh
                              (unified — same model the NMM stage uses)

Both share the same crop pipeline:
  - Run face detection on CLAHE-enhanced full frames
  - Return bounding box (x1, y1, x2, y2) and confidence float
  - Crop face region with 20% spatial padding
  - Apply a second CLAHE pass (LAB L-channel) and resize to 224x224 for HSEmotion
"""

import cv2
import numpy as np
import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from config.settings import (
    FACE_CROP_SIZE as CROP_SIZE,
    PADDING_RATIO,
    CLAHE_CLIP_LIMIT as CLAHE_CLIP,
    CLAHE_TILE_GRID_SIZE as CLAHE_TILE,
    DETECTION_CONFIDENCE_THRESHOLD,
    FACE_BACKEND,
)

log = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    bbox: Optional[Tuple[int, int, int, int]]   # (x1, y1, x2, y2) in pixels
    confidence: float                            # 0.0 to 1.0
    face_crop: Optional[np.ndarray]             # 224x224 BGR, CLAHE-normalised


def create_face_detector(backend: str = None):
    """
    Factory: return the configured face-detection backend.

    Both backends implement the same interface — detect(clahe_frame) ->
    DetectionResult, and close() — and produce the crop via the SAME pad/CLAHE/
    resize pipeline, so they are drop-in interchangeable. Only the bounding-box
    source differs (MediaPipe BlazeFace vs InsightFace SCRFD).
    """
    backend = (backend or FACE_BACKEND).lower()
    if backend == "insightface":
        return FaceDetector()
    if backend == "mediapipe":
        return MediaPipeFaceDetector()
    log.warning(f"Unknown FACE_BACKEND '{backend}', defaulting to mediapipe.")
    return MediaPipeFaceDetector()


class FaceDetector:
    """
    Wraps the RetinaFace MobileNet-0.25 model using the insightface library.
    Instantiate once at module startup — do not create per-frame.
    """

    def __init__(self):
        self.dummy_mode = False
        self._released = False   # guard for close() to prevent multiple-release
        self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

        try:
            from insightface.app import FaceAnalysis
            # 'buffalo_sc' contains the det_500m.onnx model, which is the
            # MobileNet architecture required by the assignment.
            self.app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection'])

            # Try GPU (ctx_id=0), fall back to CPU (ctx_id=-1)
            try:
                # det_size=(640,640) forces the 640x640 preprocessing required by the model
                self.app.prepare(ctx_id=0, det_thresh=DETECTION_CONFIDENCE_THRESHOLD, det_size=(640, 640))
                log.info("Loaded RetinaFace (MobileNet) on GPU")
            except Exception as e:
                log.warning(f"GPU prepare failed, falling back to CPU: {e}")
                self.app.prepare(ctx_id=-1, det_thresh=DETECTION_CONFIDENCE_THRESHOLD, det_size=(640, 640))
                log.info("Loaded RetinaFace (MobileNet) on CPU")

        except Exception as e:
            log.error(f"Failed to load RetinaFace via FaceAnalysis: {e}")
            self.dummy_mode = True
            log.warning("Using dummy face detector (no model available)")

    def close(self) -> None:
        """
        Release InsightFace resources explicitly.
        Call at process shutdown to prevent the model being re-loaded
        during Python's non-deterministic object finalization order.

        Guarded by _released flag to prevent multiple-release logging spam
        when both explicit close() and Python's GC/__del__ trigger cleanup.
        """
        if self._released:
            return
        self._released = True

        if not self.dummy_mode and hasattr(self, 'app') and self.app is not None:
            log.info("FaceDetector releasing InsightFace resources.")
            try:
                del self.app
            except Exception:
                pass
            self.app = None
        self.dummy_mode = True  # prevent any further detect() calls from using the model

    def __del__(self):
        """Ensure cleanup on GC — delegates to guarded close()."""
        self.close()

    def detect(self, clahe_frame: np.ndarray) -> DetectionResult:
        """
        Run face detection on a CLAHE-enhanced BGR frame.
        """
        h, w = clahe_frame.shape[:2]

        if self.dummy_mode:
            # Return a dummy detection in the center of the frame
            x1, y1, x2, y2 = int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face_crop = cv2.resize(clahe_frame[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
            return DetectionResult(bbox=(x1, y1, x2, y2), confidence=0.0, face_crop=face_crop)

        # Insightface FaceAnalysis expects BGR images natively, so we pass clahe_frame directly
        faces = self.app.get(clahe_frame)

        if not faces:
            return DetectionResult(bbox=None, confidence=0.0, face_crop=None)

        # Faces are already filtered by the det_thresh set in prepare()
        # Take the face with the highest confidence
        best_face = max(faces, key=lambda f: f.det_score)

        x1, y1, x2, y2 = best_face.bbox.astype(int)
        score = best_face.det_score

        # Clamp to frame just in case
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        # Pad and crop
        x1p, y1p, x2p, y2p = _pad_bbox(x1, y1, x2, y2, h, w)
        crop = clahe_frame[y1p:y2p, x1p:x2p]

        if crop.size == 0:
            return DetectionResult(bbox=None, confidence=0.0, face_crop=None)

        # Second CLAHE pass on crop
        crop_clahe = _apply_clahe_to_crop(crop, self._clahe)
        face_crop = cv2.resize(crop_clahe, (CROP_SIZE, CROP_SIZE))

        return DetectionResult(
            bbox=(x1p, y1p, x2p, y2p),
            confidence=float(score),
            face_crop=face_crop
        )


class MediaPipeFaceDetector:
    """
    Face detection via the MediaPipe Tasks FaceLandmarker, exposing the SAME
    interface and crop pipeline as FaceDetector. This unifies the pipeline on a
    single face system: the face box is taken from the 478 face-mesh landmarks
    produced by the *same* `face_landmarker.task` model the NMM stage already
    uses — no extra model download, removing the separate heavier InsightFace
    detector.

    The crop is produced by the same _pad_bbox / _apply_clahe_to_crop / resize
    path as the InsightFace backend, so only the bounding-box source differs.
    """

    def __init__(self):
        self.dummy_mode = False
        self._released = False
        self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            # Reuse the NMM model path + downloader so there is a single source
            # of truth for the face_landmarker.task location.
            from processes.nmm_classifier import MODEL_PATH, _download_model_if_missing

            self._mp = mp
            _download_model_if_missing()
            base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base,
                output_face_blendshapes=False,             # only need geometry here
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=DETECTION_CONFIDENCE_THRESHOLD,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            log.info("Loaded MediaPipe FaceLandmarker as unified face detector.")
        except Exception as e:
            log.error(f"Failed to load MediaPipe face detector: {e}")
            self.dummy_mode = True
            log.warning("Using dummy face detector (no model available)")

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        if not self.dummy_mode and getattr(self, "_landmarker", None) is not None:
            log.info("MediaPipeFaceDetector releasing resources.")
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        self.dummy_mode = True

    def __del__(self):
        self.close()

    def detect(self, clahe_frame: np.ndarray) -> DetectionResult:
        h, w = clahe_frame.shape[:2]

        if self.dummy_mode:
            x1, y1, x2, y2 = int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face_crop = cv2.resize(clahe_frame[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
            return DetectionResult(bbox=(x1, y1, x2, y2), confidence=0.0, face_crop=face_crop)

        rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return DetectionResult(bbox=None, confidence=0.0, face_crop=None)

        # Bounding box = extent of the face-mesh landmarks (normalised → pixels).
        lms = result.face_landmarks[0]
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        x1 = int(min(xs) * w); x2 = int(max(xs) * w)
        y1 = int(min(ys) * h); y2 = int(max(ys) * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return DetectionResult(bbox=None, confidence=0.0, face_crop=None)

        # Identical pad / CLAHE / resize pipeline to the InsightFace backend.
        x1p, y1p, x2p, y2p = _pad_bbox(x1, y1, x2, y2, h, w)
        crop = clahe_frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return DetectionResult(bbox=None, confidence=0.0, face_crop=None)

        crop_clahe = _apply_clahe_to_crop(crop, self._clahe)
        face_crop = cv2.resize(crop_clahe, (CROP_SIZE, CROP_SIZE))
        # FaceLandmarker does not expose a detection score; face-present == high
        # confidence (the model's own min_face_detection_confidence gate already
        # rejected weak detections). Tracking-loss is driven by no-face frames.
        return DetectionResult(
            bbox=(x1p, y1p, x2p, y2p),
            confidence=1.0,
            face_crop=face_crop,
        )


def _apply_clahe_to_crop(bgr_crop: np.ndarray, clahe) -> np.ndarray:
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _pad_bbox(x1: int, y1: int, x2: int, y2: int,
              frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * PADDING_RATIO)
    pad_y = int(bh * PADDING_RATIO)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(frame_w, x2 + pad_x)
    y2p = min(frame_h, y2 + pad_y)
    return x1p, y1p, x2p, y2p
