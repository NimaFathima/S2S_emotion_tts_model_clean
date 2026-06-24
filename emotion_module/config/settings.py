# config/settings.py

# ── JJ's original constants ───────────────────────────────────────────────
TARGET_FPS            = 30.0
CLAHE_CLIP_LIMIT      = 2.0
CLAHE_TILE_GRID_SIZE  = (8, 8)
# Shared-memory layout (see main.py / shared_memory.py):
#   [0]=valence [1]=arousal [2]=detection_confidence [3]=tracking_lost
#   [4]=is_yn_question [5]=is_wh_question [6]=is_negation
BUFFER_ELEMENT_COUNT  = 7

# ── Face detection ────────────────────────────────────────────────────────
DETECTION_CONFIDENCE_THRESHOLD = 0.65
FACE_CROP_SIZE                 = 224
PADDING_RATIO                  = 0.20

# Face-detection backend (single source of truth for the whole pipeline):
#   "insightface"— RetinaFace/SCRFD via InsightFace. **Default / showcase-safe.**
#                  On a GPU machine this runs on CUDA and is fast; it is the
#                  configuration the project has been running, so it is the
#                  proven-on-hardware choice for a live demo.
#   "mediapipe"  — face box from the MediaPipe FaceLandmarker mesh. Unifies the
#                  pipeline on ONE face system (the SAME model the NMM stage
#                  uses), no extra download. Verified FASTER on CPU (-32% cycle).
#                  CAVEAT: MediaPipe's Python Tasks API has no GPU delegate on
#                  Windows, so on a GPU laptop InsightFace (CUDA) may be faster
#                  than MediaPipe (CPU). Benchmark on the target machine before
#                  switching:  python benchmark_pipeline.py
# Crop geometry (padding, CLAHE, resize) is identical for both — only the
# bounding-box source differs — so switching never changes the emotion crop math.
FACE_BACKEND = "insightface"

# ── Coasting / tracking loss ──────────────────────────────────────────────
COASTING_DECAY_FRAMES      = 15
TRACKING_LOST_HOLD_FRAMES  = 30

# ── EMA smoothing (Process 2 reference) ──────────────────────────────────
EMA_ALPHA_NORMAL   = 0.30
EMA_ALPHA_GRAMMAR  = 0.08
EMA_WARMUP_FRAMES  = 15

# ── Neutral dead band ─────────────────────────────────────────────────────
NEUTRAL_THRESHOLD_VALENCE = 0.20
NEUTRAL_THRESHOLD_AROUSAL = 0.20

# ── NMM geometry thresholds ───────────────────────────────────────────────
# BROW_RAISE_THRESHOLD: increase if getting false Y/N positives at rest
# BROW_FURROW_THRESHOLD: make more negative if WH not triggering
BROW_RAISE_THRESHOLD   =  0.08
BROW_FURROW_THRESHOLD  = -0.05
HEAD_YAW_THRESHOLD     =  8.0
NMM_DAMPEN_ALPHA       =  0.75
NMM_CALIBRATION_FRAMES =  30     # frames to collect for per-session baseline calibration

# ── Grammatical-vs-affective brow gate (confound-specific) ───────────────────
# A grammatical brow movement and an emotional one can land in the same brow
# position, so we disambiguate by the OTHER muscles each emotion recruits:
#   - A Y/N RAISE is confounded with SURPRISE/FEAR, which additionally widen
#     the eyes and drop the jaw (eyeWide, jawOpen). A grammatical raise does
#     not. (A smile does not either — so a smiling Y/N stays detectable.)
#   - A WH FURROW is confounded with ANGER, which additionally sneers the nose,
#     presses the lips and squints the eyes (noseSneer, mouthPress, eyeSquint).
# We read MediaPipe blendshapes; if the confound markers for that direction are
# below this threshold, the brow movement is grammatical, otherwise affective
# (kept as emotion, not mapped to a question).
#   AFFECT_CONFOUND_THRESHOLD: raise if emotion leaks through as a question;
#                              lower if real grammatical questions are missed.
AFFECT_CONFOUND_THRESHOLD = 0.30

# Temporal stability for the brow gate (hysteresis). When True, a change in the
# grammatical/affective decision must persist TEMPORAL_GATE_STABLE_FRAMES frames
# before taking effect — suppresses single-frame flicker for steadier output.
# Default False = exact single-frame behaviour (zero change to the live demo).
# Provably cannot turn a stable statement into a question; adds ~stable_frames
# of latency on real events (well under the downstream sustain window).
TEMPORAL_GATE              = False
TEMPORAL_GATE_STABLE_FRAMES = 3

# ── NMM sustain gating ───────────────────────────────────────────────────
# SUSTAIN_FRAMES_REQUIRED: frames an NMM flag must be held before it is
# considered 'real' (prevents single-frame spikes). At 30 FPS:
#   6  frames =  ~200ms  (responsive — good for demo testing)
#  10  frames =  ~333ms  (original — more noise-resistant)
SUSTAIN_FRAMES_REQUIRED  = 6

# ── Chatterbox TTS ──────────────────────────────────────────────────────────
CHATTERBOX_DEVICE             = "cuda"    # change to "cpu" if VRAM < 4GB
CHATTERBOX_SAMPLE_RATE        = 24000     # Chatterbox output sample rate (Hz)

# Emotion parameter ranges for Chatterbox
EXAGGERATION_DEFAULT          = 0.50     # neutral voice
EXAGGERATION_MIN              = 0.20     # floor — never fully flat
EXAGGERATION_MAX              = 1.10     # ceiling — prevent over-dramatisation
CFG_WEIGHT_DEFAULT            = 0.50
CFG_WEIGHT_MIN                = 0.15
CFG_WEIGHT_MAX                = 0.85

