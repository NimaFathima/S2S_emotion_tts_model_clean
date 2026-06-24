# Signet Aid — Emotion + Grammar + TTS Module

## Overview
The Signet Aid emotion module converts ASL facial expression into emotionally-
coloured spoken audio. It began as the Process-1 vision engine (face detection +
emotion + NMM at a locked 30 FPS) and now also covers the **affective-vs-
grammatical separation**, the **emotion→TTS fusion**, evaluation/benchmark
tooling, and per-run logging. Architecture, design rationale, measured changes,
and roadmap are in this file plus [README](README.md),
[EMOTION_GRAMMAR_SEPARATION](EMOTION_GRAMMAR_SEPARATION.md), and
[BENCHMARKS](BENCHMARKS.md).

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
  - **Pluggable backend** (`create_face_detector()` + `FACE_BACKEND`): the detector
    is now selectable between `"insightface"` (default, GPU-safe) and `"mediapipe"`
    (unified — face box from the same FaceLandmarker the NMM stage uses). Both share
    the identical pad/CLAHE/resize crop pipeline. See Change 1 below and BENCHMARKS.md.

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

## Architecture Improvements (measured & flag-gated)
Each change was made one at a time against a locked baseline, with a
"no-degradation gate." Full before/after numbers: **[BENCHMARKS.md](BENCHMARKS.md)**.

### Change 1 — Unify on one face system (`FACE_BACKEND`)
- New `MediaPipeFaceDetector` derives the face box from the **same**
  `face_landmarker.task` model the NMM stage already loads — no extra download,
  one face system. Selectable via `FACE_BACKEND`; crop math identical to InsightFace.
- **Measured (CPU):** detect stage 54→23 ms, cycle −31.7%; separation accuracy
  unchanged; emotion-crop bbox IoU 0.83 vs InsightFace.
- **GPU caveat:** MediaPipe's Python API has no GPU delegate on Windows, while
  InsightFace runs on CUDA — so the CPU win may not hold on GPU. **Default is
  `"insightface"`** (proven-on-hardware, showcase-safe). MediaPipe is opt-in,
  adopt only after benchmarking on the target GPU.

### Change 2 — Temporal stability for the brow gate (`TEMPORAL_GATE`)
- Gate logic extracted to a pure, testable unit `processes/brow_gate.py`
  (`BrowTemporalGate`). `temporal=True` adds hysteresis (3-frame) that suppresses
  single-frame flicker (14→0 flips in tests) and provably cannot turn a stable
  statement into a question. **Default `False`** = exact prior behaviour.
- Honest scope: a *stability* win, not a fix for the surprise-toned-Y/N recall
  miss (that needs clause timing from the gloss stream). Verified by
  `processes/test_brow_gate.py`.

### Measurement tooling
- `benchmark_pipeline.py` — per-stage latency / regression guard.
- `evaluate_nmm.py` — separation-accuracy metrics (false-question rate, recall…).
- `processes/test_brow_gate.py` — proves the gate refactor is behaviour-preserving.

## Roadmap / Planned Next
- **Change 3 — TTS prosody:** map valence/arousal to real prosody (pitch, rate,
  intensity, pausing) instead of Chatterbox's two opaque knobs. Options: SSML
  (Azure/Google, transparent) or a local controllable model (ECE-TTS/EmoSphere++).
  Will be **flag-gated with Chatterbox as the fallback default**. Biggest quality
  win; deferred until after the live showcase to protect demo stability.
- **Validation clips (data, not code):** record ~30–60 labelled ASL clips and run
  `evaluate_nmm.py --from-video` to tune `AFFECT_CONFOUND_THRESHOLD` /
  `BROW_RAISE_THRESHOLD` and report a measured false-question rate. Highest
  competition value.
- **North-star (research):** a joint multimodal model (hands + face + pose →
  translation *and* affect/prosody together), per the EASLT line of work. Needs a
  signer corpus; long-term.

## Known Limitations & Integration Notes
While the system strictly fulfills all assignment requirements, the following architectural constraints were observed:

1. **Architectural Redundancy** *(addressed, opt-in)*: The pipeline detects the face twice per frame (InsightFace for the emotion crop + MediaPipe for the NMM landmarks). Change 1 adds a unified `FACE_BACKEND="mediapipe"` that removes the separate InsightFace stage (−31.7% cycle on CPU). It is **not** the default because the win is hardware-dependent (no MediaPipe GPU delegate on Windows) — the dual-detection InsightFace path remains the showcase-safe default.
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
- [x] Change 1: unified face backend (`FACE_BACKEND`) + latency benchmark (`benchmark_pipeline.py`)
- [x] Change 2: temporal brow-gate stability (`brow_gate.py`, `TEMPORAL_GATE`) + `test_brow_gate.py`
- [x] Per-run file logging (`emotion_module/logs/`); before/after numbers tracked in `BENCHMARKS.md`
- [x] `.gitignore` covers caches, ADS, downloaded images, generated audio, eval output
- [x] Dead scripts removed (`test_face/face_prep/preprocess/scores.py` — abandoned raw-RetinaFace approach)
- [x] `README.md` + `EMOTION_GRAMMAR_SEPARATION.md` document architecture, demo modes, evaluation, limitations
- [x] `PROJECT_SUMMARY.md` updated with separation design + known limitations
