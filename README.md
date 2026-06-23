# Signet Aid — Emotion + Grammar Module

Real-time ASL → emotionally-coloured speech. This module converts a signer's
facial expression into Valence/Arousal, **separates affective expression from
grammatical non-manual markers (NMMs)**, and maps the result to expressive TTS
so the spoken output conveys both *what* was signed and *how* it was felt —
without mistaking grammar for emotion (or vice-versa).

> **Why this is hard:** in sign languages the face is *layered* — the same brow
> movement can mark a yes/no question **or** signal surprise. Getting this wrong
> means adding a question intonation to a statement, or flattening real emotion
> as if it were grammar. The separation strategy is the heart of this project —
> see **[EMOTION_GRAMMAR_SEPARATION.md](EMOTION_GRAMMAR_SEPARATION.md)**.

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**
* **NVIDIA GPU (recommended):** CUDA 11.x/12.x + cuDNN. Falls back to DirectML
  (Windows non-NVIDIA) or CPU.
* **Webcam.**

### Installation
```bash
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt        # or requirements_win.txt on Windows
```

### Running
```bash
cd emotion_module
python main.py                  # full pipeline + live preview window
python main.py --headless       # no GUI (servers)
python main.py --demo           # also feed sample text to the TTS every 10s
python main.py --demo --demo-fixed   # honest demo: fixed script, all sentence types
```

**`--demo` vs `--demo --demo-fixed`:** plain `--demo` picks text that *matches*
the detected emotion (a congruent "happy path" showcase — but circular, so the
fusion can never be seen to be wrong). `--demo-fixed` plays a fixed,
expression-independent script across every sentence type, so you can actually
watch the text→sentence-type→prosody path handle each case. **Use `--demo-fixed`
for honest demos and debugging.**

## 🏗️ Architecture

Three OS processes communicate over shared memory + queues:

* **Process 1 — Vision Producer** (`main.py::vision_producer`): captures 30 FPS,
  CLAHE preprocessing, face detection (RetinaFace via InsightFace), emotion
  inference (HSEmotion V/A), and NMM/grammar detection (MediaPipe). Writes
  effective V/A + grammar flags to shared memory. Heavy models are **interleaved**
  round-robin (one per frame) to hold the frame budget. The per-frame inference
  block is **guarded** so one bad frame is skipped, not fatal.
* **Process 2 — Audio Consumer** (`processes/audio_consumer.py`): EMA-smooths V/A,
  **fuses the text channel with the grammar flags** to resolve sentence type,
  applies punctuation, maps emotion → Chatterbox TTS parameters, and synthesises
  speech.
* **Process 3 — Audio Player** (`processes/audio_player.py`): non-blocking
  playback via sounddevice.

The main orchestrator monitors all three children and shuts down cleanly if any
dies.

### Emotion vs. Grammar separation (the core)
Three cooperating channels (full detail in
[EMOTION_GRAMMAR_SEPARATION.md](EMOTION_GRAMMAR_SEPARATION.md)):

1. **Text channel** (`sentence_type_from_text`) — owns WH-questions, lexical
   negation, and explicit/auxiliary-fronted yes/no questions from the translated
   English text.
2. **Confound gate** (`nmm_classifier._confound_scores`) — uses MediaPipe
   blendshapes to tell a *grammatical* brow movement (brow-isolated) from an
   *emotional* one (whole-face): a **raise** is grammatical unless surprise
   markers fire (eyeWide, jawOpen); a **furrow** unless anger markers fire
   (noseSneer, mouthPress, eyeSquint).
3. **Fusion + conditional dampening** — punctuation is driven by the *resolved*
   sentence type (never a raw brow flag), and emotion is only dampened for
   genuine, brow-isolated grammar — so an angry question keeps its anger.

## 📊 Evaluation

`evaluate_nmm.py` measures whether the separation actually works and turns the
two tunable thresholds into defensible numbers.

```bash
cd emotion_module
# No clips / no models needed — validate the text + fusion layer:
python evaluate_nmm.py --from-features eval_samples/sample_features.csv

# With recorded clips (needs MediaPipe): runs the real NMMClassifier per clip:
python evaluate_nmm.py --from-video eval_samples/manifest_template.csv --out results.csv
```

Headline metrics:
* **`false_question_rate`** — true statements wrongly spoken as questions (**the
  core safety number**; lower is better).
* **`question_recall`** — true questions emitted with the right type.
* **`affect_preservation`** — emotional statements kept as statements.
* **`brow_affective` precision/recall** — does the gate flag emotional brows.

Record ~30–60 ASL clips (load up on the trap cases — *surprised statements,
emotional questions, smiling yes/no questions*), label them per
`eval_samples/manifest_template.csv`, then tune `AFFECT_CONFOUND_THRESHOLD` and
`BROW_RAISE_THRESHOLD`. Report the before/after `false_question_rate` as evidence.

## 🧪 Tests
```bash
cd emotion_module
python test_integration.py            # 8-test smoke suite (governor, CLAHE, shared mem, …)
python processes/test_tts.py          # VA→Chatterbox mapping + text/fusion logic (no model)
python processes/test_tts.py --model  # + actual Chatterbox synthesis to .wav
python test_emotion.py                # HSEmotion V/A on real happy/angry images
python verify_pipeline.py             # full face+emotion+NMM on sample images
python benchmark_nmm_detection.py     # full-frame vs crop MediaPipe on your hardware
```

## ⚙️ Configuration
All tunable constants live in `config/settings.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `TARGET_FPS` | 30.0 | Locked frame rate |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.65 | Min face-detection confidence |
| `COASTING_DECAY_FRAMES` | 15 | Frames for V/A decay on tracking loss |
| `TRACKING_LOST_HOLD_FRAMES` | 30 | Frames before tracking declared lost |
| `BROW_RAISE_THRESHOLD` | 0.08 | Relative brow raise → Y/N candidate |
| `BROW_FURROW_THRESHOLD` | -0.05 | Relative brow furrow → WH candidate |
| `AFFECT_CONFOUND_THRESHOLD` | 0.30 | Below = grammatical brow; above = affective (see separation doc) |
| `NMM_DAMPEN_ALPHA` | 0.75 | Emotion dampening for genuine grammatical brows |
| `NMM_CALIBRATION_FRAMES` | 30 | Per-session neutral-brow baseline window |
| `EMA_WARMUP_FRAMES` | 15 | Frames before TTS trusts smoothed V/A |
| `SUSTAIN_FRAMES_REQUIRED` | 6 | Frames an NMM flag must hold to count |
| `CHATTERBOX_DEVICE` | "cuda" | TTS device ("cpu" if VRAM < 4GB) |

## 📝 Logs
Each `python main.py` run is logged (console **and** file) to
`emotion_module/logs/signet_aid_<timestamp>.log`, combining all three processes.
The path prints at startup. Override with the `SIGNET_LOG_FILE` env var. Logs are
git-ignored. See [TESTING.md](TESTING.md#logs) for capturing test output too.

## 💡 GPU notes
* Auto-uses **CUDA** if available; otherwise **DirectML** (DML), else CPU.
* "Frame governor falling behind" warnings = hardware can't hold 30 FPS.
* The MediaPipe `Failed to send to clearcut` log is harmless offline telemetry.

## ⚠️ Known limitations (design-level)
* **Surprise-toned yes/no question** (genuine shock *and* a real question, no
  WH-word in text): the frame-level gate may miss the "?" (emotion is still
  preserved). Measured by `question_recall`. Fix path: temporal NMM dynamics.
* **Emotion model is generic.** HSEmotion (`enet_b0_8_va_mtl`) is trained on
  AffectNet (non-signers); validate V/A on signer clips, don't assume it
  generalises. `_verify_output_layout()` guards the V/A index assumption.
* **Channel timing.** Emotion and NMM are computed on different interleaved
  frames (~100–200ms skew); usually fine, but a known seam.
* **Calibration** assumes a roughly neutral first ~1s; **no per-user
  expressiveness scaling** for limited-mobility signers; **one emotion snapshot
  per sentence** (intra-sentence affect not tracked).

## 📁 Repository
* `.gitignore` excludes venvs, caches, Windows ADS, downloaded test images,
  generated `.wav`/eval CSV.
* All test scripts are guarded with `if __name__ == "__main__"`.
