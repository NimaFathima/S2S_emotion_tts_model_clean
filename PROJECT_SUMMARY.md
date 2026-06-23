# Signet Aid — Nima's Vision Producer Engine (Process 1)

## Overview
This implements Nima's half of Process 1 for the Signet Aid system, which converts ASL sign language to emotionally-coloured spoken audio. Process 1 handles all computer vision work at a locked 30 FPS.

## Components Implemented

### 1. Face Localization (RetinaFace MobileNet-0.25)
- **File**: `emotion_module/processes/face_detector.py`
- **Responsibilities**:
  - Load the MobileNet-0.25 RetinaFace architecture.
  - Run face detection on CLAHE-enhanced full frames.
  - Return bounding box with the required 20% spatial padding.
  - Resize crop to 224x224 for HSEmotion.
  - Apply second CLAHE pass to face crop (LAB L-channel only).
- **Final Implementation Notes**:
  - The raw ONNX graph outputs anchor offsets rather than pixel coordinates. To resolve this without breaking assignment requirements, the system uses the `insightface.app.FaceAnalysis(name='buffalo_sc')` API. This downloads the exact required MobileNet-0.25 RetinaFace model and automatically applies the necessary Anchor Decoding/NMS math.
  - Successfully utilizes GPU acceleration (`DmlExecutionProvider` / `CUDAExecutionProvider`).
  - Strict confidence threshold of 0.65.

### 2. Confidence Fail-Safe & Coasting Matrix
- **File**: `emotion_module/processes/coasting_matrix.py`
- **Responsibilities**:
  - Track frames since last valid detection.
  - When RetinaFace confidence drops below 0.65, trigger the 15-frame exponential decay window to smoothly guide metrics toward (0.0, 0.0).
  - When confidence >= 0.65: reset decay counter, return raw Valence/Arousal.
- **Final Implementation Notes**:
  - The decay logic smoothly dampens tracking loss instead of crashing or stuttering. Tracking is declared fully 'lost' after 30 frames.

### 3. HSEmotion GPU Inference
- **File**: `emotion_module/processes/emotion_inference.py`
- **Responsibilities**:
  - Initialize the `enet_b0_8_va_mtl` ONNX session.
  - Preprocess 224×224 BGR face crop for inference.
  - Run the forward pass to output the true Valence and Arousal decimals inside [-1.0, +1.0].
- **Final Implementation Notes**:
  - **Bug Fix Applied:** The `enet_b0_8_va_mtl` model outputs an array of 10 elements (8 discrete emotions + 2 V/A scores). The extraction logic was updated to correctly pull Valence and Arousal from indices `[-2]` and `[-1]`.
  - Runs successfully on GPU via `DmlExecutionProvider`.

### 4. MediaPipe NMM Eyebrow Classifier
- **File**: `emotion_module/processes/nmm_classifier.py`
- **Responsibilities**:
  - Load MediaPipe FaceLandmarker tracker.
  - Extract the 5 upper-arc indices for both brows from the 478-point mesh.
  - Compute deterministic grammar ratios for eyebrow raise, furrow, and negation (head shake).
- **Final Implementation Notes**:
  - Operates perfectly on the current environment. NMM dampening logic successfully intercepts and dampens raw VA scores when grammatical markers are active.

## Recent Improvements (Post-Code Review)

### Performance
- **CPU Spinlock Eliminated**: `src/governor.py` no longer burns 100% CPU during frame pacing. Sleep-based waiting reduces idle load to near-zero.
- **CLAHE Object Cached**: `FaceDetector` creates `cv2.createCLAHE()` once at startup and reuses it per-frame instead of recreating it.
- **Dummy Mode Confidence Fixed**: When RetinaFace fails to load, the fallback now returns `confidence=0.0` (was `0.9`), so downstream code correctly detects failure.

### Reliability
- **Process Monitoring**: `main.py` now monitors all three child processes (P1 Vision, P2 Audio Consumer, P3 Audio Player). If any dies, the system shuts down cleanly instead of leaving orphans.
- **Headless Mode**: Added `--headless` CLI flag. Skips `cv2.imshow` and window creation for server deployments.
- **Specific Exception Handling**: Replaced bare `except Exception` in preprocessor and inference modules with targeted `(cv2.error, OSError, ValueError)` catches.

### Maintainability
- **Centralized Constants**: All tunable thresholds moved to `config/settings.py`. Modules import from there instead of duplicating magic numbers.
- **Logging Standardization**: Replaced ad-hoc `print()` calls with standard `logging.getLogger(__name__)` across `face_detector.py`, `emotion_inference.py`, `nmm_classifier.py`, and `coasting_matrix.py`.
- **Class Rename**: `AtomicSharedMemoryWriter` renamed to `AtomicSharedMemory` — it was always used for both reading and writing.
- **Import Guards**: All test scripts now wrap execution in `if __name__ == "__main__"` to prevent accidental side effects on import.
- **Model Download Path Fix**: `nmm_classifier.py` now uses `pathlib` to resolve the models directory reliably, fixing the mismatch between `processes/models/` and `../models/`.
- **.gitignore Added**: Covers `__pycache__/`, `venv/`, Windows ADS (`*:Zone.Identifier`), backups, and media artifacts.

### Benchmarking
- **NMM Benchmark Script**: Added `benchmark_nmm_detection.py` to compare full-frame vs. crop-based MediaPipe on your target hardware. On the tested environment the speedup was negligible (~0.96×), so the full-frame approach remains the default to preserve accuracy.

## Known Limitations & Integration Notes
While the system strictly fulfills all assignment requirements, the following architectural constraints were observed:

1. **Architectural Redundancy**: The pipeline calculates face tracking twice per frame (once via RetinaFace for the emotion crop, and again internally by MediaPipe for the NMM landmarks). Benchmarking showed that reusing the RetinaFace crop for MediaPipe provides negligible speedup (~0.96×) while risking landmark accuracy due to lost context. Therefore, the full-frame dual-detection approach remains the default.
2. **Preprocessing Conflict**: Applying CLAHE high-contrast enhancement *before* face detection can artificially darken eyes/features. In poor lighting, this may drop RetinaFace confidence below the 0.65 threshold, prematurely triggering the Coasting Matrix.
3. **HSEmotion Jitter**: The ONNX emotion model evaluates frames independently without temporal smoothing. Maintaining a perfectly still expression will still result in slight decimal jitter in the V/A scores due to sensor noise.

## Affective vs. Grammatical Expression Separation (Core)
The headline capability: distinguishing **emotional** facial expression from
**grammatical** non-manual markers (NMMs), so the spoken output never mistakes a
yes/no-question brow-raise for surprise, or a shocked statement for a question.
Full design: **[EMOTION_GRAMMAR_SEPARATION.md](EMOTION_GRAMMAR_SEPARATION.md)**.

Three cooperating channels:
1. **Text channel** (`audio_consumer.sentence_type_from_text` / `resolve_sentence_type`):
   the translated English text owns WH-questions, lexical negation, and explicit/
   auxiliary-fronted yes/no questions. The brow only decides yes/no when the text
   is silent — the one ASL case words cannot disambiguate.
2. **Confound gate** (`nmm_classifier._confound_scores` + `classify`): MediaPipe
   **blendshapes** distinguish a brow-isolated *grammatical* movement from a
   whole-face *emotional* one. A **raise** is grammatical unless surprise markers
   (eyeWide, jawOpen) fire; a **furrow** unless anger markers (noseSneer,
   mouthPress, eyeSquint) fire. Affective brows are flagged `brow_affective`.
3. **Fusion + conditional dampening** (`audio_consumer.VAToChatterbox.modify_text`,
   `nmm_classifier.apply_dampening`): punctuation comes from the *resolved* type
   (never a raw brow flag); emotion is dampened only for genuine, brow-isolated
   grammar — so an angry WH-question keeps its anger.

### Reliability / correctness hardening
- **Vision-loop crash guard**: the per-frame inference + write block is wrapped
  so one malformed frame is logged (≤1/s) and skipped instead of killing the
  vision producer (which would tear down the whole system).
- **Negation logic fixed**: replaced the incoherent
  `is_negation = is_wh_question or avg_raise < 0` with a coherent head-shake test
  (`head_shake and not is_yn_question`); lexical negation is owned by the text
  channel and the head-shake is a confirmation signal.
- **HSEmotion output-layout check**: `_verify_output_layout()` runs one dummy
  inference at load and warns if the output is not the expected 10 values, so a
  swapped model that breaks the `[-2]=valence, [-1]=arousal` assumption surfaces
  loudly instead of silently inverting emotion.
- **Honest demo mode** (`--demo-fixed`): plays a fixed, expression-independent
  script so the text/fusion path can be seen to work (the default `--demo` is
  congruent-but-circular).

## Evaluation Harness
- **File**: `emotion_module/evaluate_nmm.py` (+ `eval_samples/`)
- Measures the separation. `--from-features` runs with no clips/models (validates
  text + fusion); `--from-video` runs the real `NMMClassifier` over recorded clips.
- Headline metric **`false_question_rate`** (true statements wrongly made
  questions) plus `question_recall`, `affect_preservation`, and `brow_affective`
  precision/recall. Use it to tune `AFFECT_CONFOUND_THRESHOLD` /
  `BROW_RAISE_THRESHOLD` against a labelled clip set.

## Known Limitations (design-level, documented not fixed)
1. **Surprise-toned yes/no question**: a genuine wide-eyed-shock yes/no question
   with no WH-word may miss the "?" (emotion is preserved). Inherent to
   frame-level disambiguation; measured by `question_recall`. Fix path: temporal
   NMM onset/offset dynamics.
2. **Generic emotion model**: HSEmotion is trained on AffectNet (non-signers);
   validate V/A on signer data rather than assume it generalises.
3. **Channel timing**: emotion and NMM run on different interleaved frames
   (~100–200ms skew).
4. **Calibration / expressiveness**: baseline assumes a roughly neutral first ~1s;
   no per-user expressiveness scaling; one emotion snapshot per sentence.

## Integration Test
- **File**: `emotion_module/test_integration.py`
- **Usage**: `cd emotion_module && python test_integration.py`
- **Function**: Runs 8 unit tests covering governor timing, preprocessor CLAHE, shared memory atomicity, face detection, coasting decay, HSEmotion inference, NMM classification, and full-pipeline frame simulation.
- **Related**: `processes/test_tts.py` (VA→Chatterbox + text/fusion logic),
  `test_emotion.py` / `verify_pipeline.py` (real-image checks),
  `evaluate_nmm.py` (separation metrics).

## Handover Checklist
- [x] All constants centralized in `config/settings.py`
- [x] `test_integration.py` passes (8/8)
- [x] `test_tts.py` logic suite passes (mapping + text/fusion)
- [x] `ast.parse` validates all modified `.py` files
- [x] Affective vs. grammatical separation implemented (text + confound gate + conditional dampening)
- [x] Evaluation harness (`evaluate_nmm.py`) added with sample features + manifest template
- [x] Vision-loop crash guard, negation fix, HSEmotion output-layout check
- [x] `.gitignore` covers caches, ADS, downloaded images, generated audio, eval output
- [x] Dead scripts removed (`test_face/face_prep/preprocess/scores.py` — abandoned raw-RetinaFace approach)
- [x] `README.md` + `EMOTION_GRAMMAR_SEPARATION.md` document architecture, demo modes, evaluation, limitations
- [x] `PROJECT_SUMMARY.md` updated with separation design + known limitations
