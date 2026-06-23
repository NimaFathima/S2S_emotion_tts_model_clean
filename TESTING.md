# Signet Aid — Testing & Verification Guide (Handover)

This guide takes you from a clean machine to a fully verified system. Work
through the tiers in order — each is faster/safer than the next and proves a
different layer:

| Tier | Needs | Proves | Time |
|------|-------|--------|------|
| 1 | Python only | Core logic: emotion↔grammar separation, TTS mapping | ~2 min |
| 2 | + internet + models | Models load, inference works on still images | ~10 min |
| 3 | + webcam (+ GPU ideal) | Full real-time pipeline + spoken output | ~15 min |

> **The headline thing to verify** is that the system separates *emotional*
> facial expression from *grammatical* eyebrow movement (a raised brow can mean
> "is this a question?" OR "I'm surprised"). Tier 1 proves the logic; Tier 3
> lets you see it live. Background: `EMOTION_GRAMMAR_SEPARATION.md`.

---

## 0. Prerequisites
- **Python 3.10+** (`python --version`)
- **Webcam** (Tier 3 only)
- **Internet** on first run (downloads models)
- **GPU optional**: NVIDIA+CUDA → fastest; Windows non-NVIDIA → DirectML; else CPU
  (works, just slower than 30 FPS).

## 1. Setup
```bash
# from the project root (the folder containing README.md)
python -m venv venv
.\venv\Scripts\activate          # Windows PowerShell/CMD
# source venv/bin/activate       # Linux/macOS

pip install -r requirements.txt        # Linux/macOS
# pip install -r requirements_win.txt  # Windows (use this on Windows)
```
If `chatterbox-tts` is not installed, the system still runs — Process 2 prints
"running in stub mode (no speech synthesis)". That's fine for Tiers 1–2.

**All commands below are run from the `emotion_module/` folder:**
```bash
cd emotion_module
```

---

## TIER 1 — Offline logic tests (no camera, no models)

These prove the emotion-vs-grammar fusion and the TTS parameter mapping with
zero hardware. **Run these first** — if they pass, the core logic is sound.

### 1a. Fusion / TTS mapping logic
```bash
python processes/test_tts.py
```
**PASS:** ends with `All logic tests (no model required) passed successfully!`
Look in the "Text Modification Test" block for these lines (all `[OK]`):
- `'That just happened' (yn_face=False) -> [statement] 'That just happened'`
  → a surprised statement is **NOT** turned into a question.
- `'Where is the exit' -> [wh] 'Where is the exit?'` → WH question gets `?`.

### 1b. Separation metrics (the headline verification)
```bash
python evaluate_nmm.py --from-features eval_samples/sample_features.csv
```
**PASS criteria** (printed under "Headline metrics"):
- `FALSE-QUESTION RATE : 0.0%` ← **most important** (no statement became a question)
- `affect preservation : 100.0%` (emotional statements kept as statements)
- `brow_affective precision/recall : 100.0%`
- `question recall : 83.3%` (one emotional Y/N is intentionally missed — this is
  the documented trade-off, not a bug)

---

## TIER 2 — Model + integration tests (needs internet + models, no camera)

First run downloads models to `models/` (InsightFace `buffalo_sc`, MediaPipe
`face_landmarker.task`, HSEmotion `enet_b0_8_va_mtl.onnx`). Allow a few minutes.

### 2a. Integration smoke suite (8 tests)
```bash
python test_integration.py
```
**PASS:** all 8 tests report OK (governor timing, CLAHE, shared-memory atomicity,
face detection, coasting decay, HSEmotion inference, NMM classification, full
frame simulation).

### 2b. Emotion model sanity on real faces
```bash
python test_emotion.py
```
Downloads a happy and an angry face and prints Valence/Arousal.
**PASS:** happy image → **valence positive**; angry image → **valence negative**.
(This confirms the V/A output order is correct — i.e. emotion isn't inverted.)

### 2c. Full vision pipeline on a still image
```bash
python verify_pipeline.py
```
**PASS:** prints a face detection, V/A scores, and NMM flags without errors.

---

## TIER 3 — Live end-to-end (webcam)

### 3a. Vision only, watch the live overlay
```bash
python main.py --headless        # console only, OR:
python main.py                   # opens "Signet Aid — Emotion Monitor" window
```
A preview window shows: green/orange face box, Valence & Arousal bars, and an
**NMM line**. Press **`q`** to quit (or Ctrl+C in the console).

**Verify the emotion-vs-grammar separation by performing these expressions** and
watching the NMM line + bars:

| Do this | Expected overlay | Meaning |
|---------|------------------|---------|
| Sit neutral ~2 s at start | `NMM: —`, bars near 0 | baseline calibration |
| Raise eyebrows **only** (calm face) | `NMM: Y/N?` in **cyan** | grammatical raise detected |
| Make a **surprised** face (brows up, **eyes wide, mouth open**) | `NMM: AFFECT` in **violet**, *not* Y/N; Arousal bar jumps | brow read as **emotion**, not a question ✅ |
| Furrow brows **only** | `NMM: WH?` in cyan | grammatical furrow |
| Make an **angry** face (furrow + tense mouth/eyes) | `NMM: AFFECT` violet; Valence negative | anger kept as emotion ✅ |
| Smile | bars: Valence positive | emotion tracked |

The key win: **surprise/anger show `AFFECT` (violet), not `Y/N?`/`WH?`** — that's
the separation working. A naive system would flag those as questions.

### 3b. Full pipeline with spoken output (honest demo)
Requires `chatterbox-tts` installed and speakers.
```bash
python main.py --demo --demo-fixed
```
This feeds a fixed, expression-independent script through the TTS every ~10 s
(first sentence after ~15 s while the model loads). Watch the **AudioConsumer**
log line: `type=… | '…text…'`.

**Verify:**
- `"What happened today"` → `type=wh`, spoken text becomes `"What happened today?"`
  (rising question intonation).
- `"The bus is arriving now"` → `type=statement`, spoken flat (no `?`).
- `"I do not agree with that"` → `type=neg`, no `?`.
- Now **hold a raised-brow** while `"You go to the store"` is spoken → `type=yn`,
  text becomes `"You go to the store?"`. Hold a **surprised** face instead → it
  stays `type=statement` (no false question). ✅

> Use `--demo` (without `--demo-fixed`) only for a "looks-nice" showcase — it
> picks text to match your expression, so it can't be seen to be wrong. Prefer
> `--demo-fixed` for honest verification.

---

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `Cannot open webcam` | Close other apps using the camera; check OS camera privacy permission. |
| `running in stub mode (no speech synthesis)` | `pip install chatterbox-tts` (Tier 3b only; Tiers 1–2 don't need it). |
| `Frame governor falling behind` | Hardware can't hold 30 FPS — expected on CPU; functionality still correct. |
| `Failed to send to clearcut: Status_ConnectFailed` | Harmless MediaPipe offline telemetry — ignore. |
| `Blendshapes unavailable …` warning | The grammar/affect gate falls back to geometry-only; ensure the MediaPipe `face_landmarker.task` downloaded to `models/`. |
| First run slow / network errors | Models download on first run — needs internet once. |
| Overlay always `TRACKING LOST` | Improve lighting / face the camera; detection confidence threshold is 0.65. |

## Logs (saved automatically — check or share these later)
Every run of `python main.py` writes a combined log of **all three processes** to:
```
emotion_module/logs/signet_aid_<YYYYMMDD_HHMMSS>.log
```
The path is also printed at startup (`Logging to …`). The same lines appear on
the console live. To capture Tier 1/2 test output too, redirect it:
```bash
python test_integration.py            > logs/test_integration.txt 2>&1
python evaluate_nmm.py --from-features eval_samples/sample_features.csv --out results.csv | tee logs/eval.txt
```
To control where the pipeline log goes, set an env var before launching:
```bash
# Windows PowerShell
$env:SIGNET_LOG_FILE="C:\path\to\my_run.log"; python main.py
# Linux/macOS
SIGNET_LOG_FILE=/path/to/my_run.log python main.py
```
**When reporting a problem, attach the latest `logs/signet_aid_*.log`** — it
contains the device used (CUDA/DML/CPU), model load lines, per-sentence
`type=…` decisions, and any errors.

## What to report back
1. Which tiers passed (1 / 2 / 3).
2. Tier 1b numbers: **false-question rate** and **affect preservation**.
3. Tier 2b: did happy=positive / angry=negative valence hold?
4. Tier 3a: did **surprise/anger show `AFFECT` (violet)** rather than `Y/N?`/`WH?`?
5. Any errors (copy the full traceback) + your OS, Python version, and whether
   GPU/DirectML/CPU was used (printed in the startup logs).

## Optional: measure on your own clips
To validate on real ASL (recommended for the competition): record short clips,
label them per `eval_samples/manifest_template.csv`, then:
```bash
python evaluate_nmm.py --from-video your_manifest.csv --out results.csv
```
Tune `AFFECT_CONFOUND_THRESHOLD` / `BROW_RAISE_THRESHOLD` in `config/settings.py`
and re-run; report the before/after **false-question rate**.
