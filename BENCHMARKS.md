# Benchmarks — measured before/after each change

We change one thing at a time and re-run both harnesses. A change is kept only
if it **improves or holds** every number below (the "no-degradation gate").

Reproduce:
```bash
cd emotion_module
python evaluate_nmm.py --from-features eval_samples/sample_features.csv   # accuracy
python benchmark_pipeline.py --iters 80                                   # latency
```
Latency numbers below are **CPU** (this dev machine); absolute values differ on
GPU/DirectML, but the *relative* change is what the gate checks.

## Baseline (before any architecture change)

**Accuracy** (grammar/affect separation — `evaluate_nmm.py`):
| metric | value |
|---|---|
| sentence-type accuracy | 92.3% |
| **false-question rate** (core safety) | **0.0%** |
| question recall | 83.3% (5/6) |
| affect preservation | 100% |
| brow_affective precision / recall | 100% / 100% |

**Latency** (`benchmark_pipeline.py`, real face, per stage):
| stage | ms |
|---|---|
| detect (InsightFace) | 54.3 |
| landmarks (MediaPipe) | 18.0 |
| emotion (HSEmotion) | 26.7 |
| **cycle (detect+landmarks+emotion)** | **99.0** |

---

## Change 1 — Unify on one face system (MediaPipe), drop InsightFace

**What:** face box now comes from the MediaPipe FaceLandmarker mesh (the same
model the NMM stage already loads) instead of a separate InsightFace SCRFD
detector. Crop pipeline (pad 20% → CLAHE → resize 224) unchanged. Selectable via
`FACE_BACKEND` ("mediapipe" default, "insightface" rollback).

**No-degradation gate:**
| check | baseline | after | verdict |
|---|---|---|---|
| false-question rate | 0.0% | 0.0% | ✅ hold |
| accuracy / recall / affect | 92.3% / 83.3% / 100% | 92.3% / 83.3% / 100% | ✅ hold (separation logic untouched) |
| emotion crop geometry | — | bbox IoU 0.83 vs InsightFace | ✅ crop barely moves |
| **detect stage** | 54.3 ms | **22.9 ms** | ✅ **2.4× faster** |
| **cycle** | 99.0 ms | **67.7 ms** | ✅ **−31.7%** |
| new model download | — | none (reuses face_landmarker.task) | ✅ |
| rollback | — | one setting | ✅ |

**Behaviour note:** detection confidence is now 1.0 when a face is present
(FaceLandmarker exposes no score); tracking-loss still fires on no-face frames.

**⚠️ GPU caveat (important):** the −32% number is **CPU**. MediaPipe's Python
Tasks API has **no GPU delegate on Windows**, so it stays on CPU, while
InsightFace runs on **CUDA** on a GPU machine. On a GPU laptop InsightFace detect
may therefore be *faster* than MediaPipe, and switching could slightly *increase*
latency. The CPU win does **not** transfer to GPU.

**Default is `FACE_BACKEND="insightface"`** — the proven-on-hardware,
showcase-safe path. The MediaPipe unification is validated on CPU and available
(`FACE_BACKEND="mediapipe"`), but should only be made default after running
`python benchmark_pipeline.py` on the target GPU machine confirms it is not
slower there.

**Needs on-device confirmation before adopting MediaPipe on GPU:**
1. `python benchmark_pipeline.py` on the GPU laptop → compare OLD vs NEW cycle.
2. `python test_emotion.py` / `verify_pipeline.py` on both backends → confirm
   valence/arousal reads the same (IoU-0.83 overlap predicts yes).

**Verdict: KEEP the code, default to InsightFace for the GPU showcase.** The
unified backend is a real CPU win and a clean simplification, but its benefit is
hardware-dependent — so we ship the proven path and adopt MediaPipe only once
measured on the target GPU.

---

## Change 2 — Temporal stability for the brow gate (hysteresis)

**What:** the grammatical-vs-affective brow decision moved into a pure, testable
unit (`processes/brow_gate.py`, `BrowTemporalGate`). With `TEMPORAL_GATE=True` it
requires a decision change to persist `TEMPORAL_GATE_STABLE_FRAMES` (3) frames
before taking effect, suppressing single-frame flicker. **Default is False** —
exact prior single-frame behaviour, so the live demo is unchanged.

**Honest scope:** this is a *stability* improvement (steadier output), NOT a fix
for the one recall miss (surprise-toned Y/N). That miss needs clause-boundary
timing from the gloss stream, which a face-only feature cannot supply without
risking the 0% false-question rate — so it was deliberately not attempted here.

**No-degradation gate** (`python processes/test_brow_gate.py`):
| check | result | verdict |
|---|---|---|
| identity (temporal=False vs prior gate) | 0 mismatches over input grid | ✅ behaviour-preserving refactor |
| separation accuracy (`evaluate_nmm.py`) | 0.0% FQ / 83.3% recall / 100% affect | ✅ unchanged |
| flicker on noisy statement | flips **14 → 0** | ✅ steadier |
| false-question frames (temporal on) | 0 | ✅ never invents a question |
| sustained Y/N still registers | yes (18/20 frames, ~2-frame latency) | ✅ no recall loss |

**Verdict: KEEP, default OFF for the showcase** (zero behaviour change). Enable
`TEMPORAL_GATE=True` for a steadier live output — it's proven safe, costs only
~3 frames of latency (well under the 6-frame downstream sustain). Hardware-
independent: no GPU caveat.
