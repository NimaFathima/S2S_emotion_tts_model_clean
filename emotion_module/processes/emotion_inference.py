"""
emotion_inference.py
Nima — Signet Aid Vision Producer, Component 3

Responsibilities:
  - Load HSEmotion enet_b0_8_va_mtl ONNX model
  - Preprocess 224x224 BGR face crop for inference
  - Run ONNX session on GPU (CUDA/DirectML/CPU fallback)
  - Return (valence: float, arousal: float) in range [-1.0, +1.0]
"""

import cv2
import numpy as np
import onnxruntime as rt
import logging
from dataclasses import dataclass
from typing import Tuple
import os
import shutil
from pathlib import Path
from config.settings import FACE_CROP_SIZE as CROP_SIZE

log = logging.getLogger(__name__)

# Canonical model directory is the repo-root `models/` (same dir the NMM
# classifier uses), resolved from __file__ so it is independent of the current
# working directory:  processes/ -> emotion_module/ -> <repo root> / models
_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "models" / "enet_b0_8_va_mtl.onnx"
)

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


@dataclass
class EmotionResult:
    valence: float    # in [-1.0, +1.0], positive = pleasant
    arousal: float    # in [-1.0, +1.0], positive = activated/energetic


class HSEmotionInference:
    """
    Wraps the HSEmotion enet_b0_8_va_mtl ONNX session.
    Instantiate once at process startup — model loading is expensive.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self._model_path = model_path
        self._session    = self._load_session()
        self._input_name = self._session.get_inputs()[0].name
        # Discover output names — model outputs valence and arousal
        self._output_names = [o.name for o in self._session.get_outputs()]
        log.info(f"HSEmotion loaded on {self._session.get_providers()[0]}")
        log.info(f"Output nodes: {self._output_names}")
        self._verify_output_layout()

    def _verify_output_layout(self) -> None:
        """
        One-time structural check of the model's output layout.

        enet_b0_8_va_mtl emits 10 values: 8 discrete-emotion logits followed by
        valence and arousal. infer() reads valence/arousal from [-2]/[-1]. If a
        different model is swapped in and the output size changes, surface it
        loudly here rather than silently producing wrong (or inverted) emotions.
        """
        EXPECTED = 10
        try:
            dummy = np.full((CROP_SIZE, CROP_SIZE, 3), 128, dtype=np.uint8)
            out = np.squeeze(self._session.run(
                self._output_names, {self._input_name: self._preprocess(dummy)}
            )[0])
            n = int(out.size)
            if n != EXPECTED:
                log.warning(
                    f"HSEmotion output has {n} values (expected {EXPECTED} = "
                    f"8 emotion logits + valence + arousal). Verify the valence/"
                    f"arousal indices [-2]/[-1] are correct for this model."
                )
            else:
                log.info(
                    "HSEmotion output layout OK: 10 values; "
                    "using [-2]=valence, [-1]=arousal."
                )
        except Exception as e:
            log.warning(f"Could not verify HSEmotion output layout: {e}")

    def _load_session(self) -> rt.InferenceSession:
        if not os.path.exists(self._model_path):
            self._download_model()

        providers = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider"
        ]
        session = rt.InferenceSession(self._model_path, providers=providers)
        active = session.get_providers()[0]
        if active == "CPUExecutionProvider":
            log.warning("HSEmotion running on CPU. GPU execution preferred for 30 FPS target.")
        return session

    def _download_model(self) -> None:
        """
        Download HSEmotion enet_b0_8_va_mtl ONNX weights.
        Uses the hsemotion-onnx package which bundles the weights.
        """
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        try:
            from hsemotion_onnx.facial_emotions import get_model_path
            # The package downloads weights on first use to a cache directory
            # Copy from cache to our models/ directory for explicit path control
            src = get_model_path("enet_b0_8_va_mtl")
            shutil.copy(src, self._model_path)
            log.info(f"Model copied to {self._model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Could not obtain HSEmotion model: {e}\n"
                "Run: pip install hsemotion-onnx"
            )

    def _preprocess(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Prepare 224x224 BGR face crop for ONNX inference.

        Pipeline:
          BGR uint8 → RGB float32 → ImageNet normalisation → NCHW tensor
        """
        # Ensure correct size
        if face_crop_bgr.shape[:2] != (CROP_SIZE, CROP_SIZE):
            face_crop_bgr = cv2.resize(face_crop_bgr, (CROP_SIZE, CROP_SIZE))

        # BGR → RGB
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # ImageNet normalisation: (pixel - mean) / std
        rgb = (rgb - _MEAN) / _STD

        # HWC → CHW → NCHW (batch=1)
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return tensor.astype(np.float32)

    def infer(self, face_crop_bgr: np.ndarray) -> EmotionResult:
        """
        Run HSEmotion inference on a prepared 224x224 face crop.

        Args:
            face_crop_bgr: 224x224 BGR numpy array, CLAHE-normalised.
                           Must come from FaceDetector.detect().face_crop.

        Returns:
            EmotionResult with valence and arousal in [-1.0, +1.0].
            Returns (0.0, 0.0) with a warning if inference fails.
        """
        try:
            tensor  = self._preprocess(face_crop_bgr)
            outputs = self._session.run(
                self._output_names,
                {self._input_name: tensor}
            )
            # HSEmotion enet_b0_8_va_mtl outputs:
            # A single array of shape (batch, 10).
            # Indices [0:8] are logits for 8 discrete emotions.
            # Index [8] is Valence, Index [9] is Arousal.
            out_array = np.squeeze(outputs[0])

            # Extract the last two elements for Valence and Arousal
            if out_array.size >= 2:
                valence = float(out_array[-2])
                arousal = float(out_array[-1])
            else:
                valence, arousal = 0.0, 0.0

            # Clamp to valid range
            valence = float(np.clip(valence, -1.0, 1.0))
            arousal = float(np.clip(arousal, -1.0, 1.0))

            return EmotionResult(valence=valence, arousal=arousal)

        except Exception as e:
            log.error(f"Inference error: {e}", exc_info=True)
            return EmotionResult(valence=0.0, arousal=0.0)


if __name__ == "__main__":
    engine = HSEmotionInference()

    # Feed a blank grey face crop — should return values near (0, 0)
    dummy_crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    result = engine.infer(dummy_crop)
    print(f"Dummy crop → V={result.valence:.4f}, A={result.arousal:.4f}")
    assert -1.0 <= result.valence <= 1.0, "Valence out of range"
    assert -1.0 <= result.arousal <= 1.0, "Arousal out of range"
    print("[emotion_inference] Smoke test passed.")
