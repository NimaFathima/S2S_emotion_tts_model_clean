"""
main.py — Signet Aid Emotion Module Entry Point
Three-process architecture:
  Process 1: Vision Producer  — face detection + emotion (V/A) + NMM grammar/affect
  Process 2: Audio Consumer   — text + emotion fusion → Chatterbox TTS prosody
  Process 3: Audio Player     — sounddevice streaming playback
"""
import os, ctypes, time, datetime, logging, multiprocessing
from multiprocessing import Process, Lock, Array, Queue, Value
import numpy as np

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(processName)s %(name)s: %(message)s"


def _setup_logging():
    """
    Configure logging to BOTH the console and a per-run file under emotion_module/logs/.

    Runs at import time, so every process (parent + the three spawned children)
    configures itself. The parent generates one timestamped filename and exports
    it via the SIGNET_LOG_FILE env var; spawned children inherit that variable and
    append to the SAME file, so a run produces a single combined log.

    Override the location by setting SIGNET_LOG_FILE before launching.
    """
    created = False
    log_file = os.environ.get("SIGNET_LOG_FILE")
    if not log_file:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"signet_aid_{ts}.log")
        os.environ["SIGNET_LOG_FILE"] = log_file   # shared with spawned children
        created = True

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Idempotent: don't add our handlers twice if import re-runs in this process.
    if not any(getattr(h, "_signet", False) for h in root.handlers):
        for handler in (logging.StreamHandler(),
                        logging.FileHandler(log_file, mode="a", encoding="utf-8")):
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            handler._signet = True
            root.addHandler(handler)

    if created:
        root.info(f"Logging to {log_file}")
    return log_file


LOG_FILE = _setup_logging()

# Suppress MediaPipe's internal telemetry logger.
# MediaPipe periodically tries to send usage stats to Google ("clearcut")
# and logs "Failed to send to clearcut: Status_ConnectFailed" when offline.
# This is completely harmless and has no impact on inference.
for _noisy_logger in ("clearcut", "absl", "mediapipe"):
    logging.getLogger(_noisy_logger).setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS 1 — Vision Producer
# ─────────────────────────────────────────────────────────────────────────────
def _draw_overlay(frame, det, coast, eff_v, eff_a, nmm_ctx, is_overexposed):
    """
    Draw live annotation overlay onto the display frame.
    Shows bounding box, VA bars, NMM flags, and status indicators.
    """
    import cv2
    h, w = frame.shape[:2]

    # ── Bounding box ─────────────────────────────────────────────────────
    if det.bbox is not None:
        x1, y1, x2, y2 = det.bbox
        # Green if confident, red/orange if coasting or low confidence
        if det.confidence >= 0.65:
            box_color = (0, 220, 100)    # green
        else:
            box_color = (0, 100, 255)    # orange
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        # Confidence label
        cv2.putText(frame, f"conf: {det.confidence:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # ── Status panel background (top-left) ───────────────────────────────
    panel_h, panel_w = 160, 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y_text = 30
    line_h = 22

    # ── Valence bar ──────────────────────────────────────────────────────
    v_label = f"Valence: {eff_v:+.3f}"
    v_color = (100, 220, 100) if eff_v >= 0 else (100, 100, 255)
    cv2.putText(frame, v_label, (16, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, v_color, 1)
    y_text += line_h
    # Draw bar
    bar_x, bar_w, bar_h = 16, 250, 12
    bar_center = bar_x + bar_w // 2
    cv2.rectangle(frame, (bar_x, y_text), (bar_x + bar_w, y_text + bar_h), (60, 60, 60), -1)
    bar_len = int((eff_v) * (bar_w // 2))
    if bar_len > 0:
        cv2.rectangle(frame, (bar_center, y_text), (bar_center + bar_len, y_text + bar_h), v_color, -1)
    elif bar_len < 0:
        cv2.rectangle(frame, (bar_center + bar_len, y_text), (bar_center, y_text + bar_h), v_color, -1)
    cv2.line(frame, (bar_center, y_text), (bar_center, y_text + bar_h), (200, 200, 200), 1)
    y_text += line_h

    # ── Arousal bar ──────────────────────────────────────────────────────
    a_label = f"Arousal: {eff_a:+.3f}"
    a_color = (100, 200, 255) if eff_a >= 0 else (200, 150, 80)
    cv2.putText(frame, a_label, (16, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, a_color, 1)
    y_text += line_h
    cv2.rectangle(frame, (bar_x, y_text), (bar_x + bar_w, y_text + bar_h), (60, 60, 60), -1)
    bar_len = int((eff_a) * (bar_w // 2))
    if bar_len > 0:
        cv2.rectangle(frame, (bar_center, y_text), (bar_center + bar_len, y_text + bar_h), a_color, -1)
    elif bar_len < 0:
        cv2.rectangle(frame, (bar_center + bar_len, y_text), (bar_center, y_text + bar_h), a_color, -1)
    cv2.line(frame, (bar_center, y_text), (bar_center, y_text + bar_h), (200, 200, 200), 1)
    y_text += line_h + 4

    # ── NMM flags ────────────────────────────────────────────────────────
    nmm_parts = []
    if nmm_ctx.is_yn_question:
        nmm_parts.append("Y/N?")
    if nmm_ctx.is_wh_question:
        nmm_parts.append("WH?")
    if nmm_ctx.is_negation:
        nmm_parts.append("NEG")
    if getattr(nmm_ctx, "brow_affective", False):
        # Brow moved but as part of a whole-face emotion → kept as affect.
        nmm_parts.append("AFFECT")
    nmm_str = "NMM: " + (" | ".join(nmm_parts) if nmm_parts else "—")
    if nmm_ctx.any_active:
        nmm_color = (0, 255, 255)            # cyan — grammatical marker active
    elif getattr(nmm_ctx, "brow_affective", False):
        nmm_color = (180, 120, 255)          # violet — brow read as emotion
    else:
        nmm_color = (150, 150, 150)
    cv2.putText(frame, nmm_str, (16, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, nmm_color, 1)
    y_text += line_h

    # ── Tracking status ──────────────────────────────────────────────────
    if coast.tracking_lost:
        cv2.putText(frame, "TRACKING LOST", (16, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif det.bbox is None:
        cv2.putText(frame, "No face detected", (16, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)

    # ── Overexposure warning (top-right) ─────────────────────────────────
    if is_overexposed:
        warn_text = "! OVEREXPOSED"
        tw = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.putText(frame, warn_text, (w - tw - 16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── Quit hint (bottom) ───────────────────────────────────────────────
    cv2.putText(frame, "Press 'q' to quit", (w - 160, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


def vision_producer(shared_array, lock, text_queue, run_flag, headless=False):
    """
    Runs JJ's governor + preprocessor + Nima's detection/inference pipeline.
    Writes VA scores atomically to shared memory.
    Displays a live preview window with annotations (unless headless).

    Interleaved Inference Scheduling:
      On DirectML hardware, RetinaFace alone takes ~130ms — far beyond 33ms.
      To prevent frame stacking, heavy inference is interleaved in round-robin
      slots so that AT MOST one expensive model runs per frame:
        Slot 0: RetinaFace  (~130ms) — face detection
        Slot 1: HSEmotion   (~35ms)  — emotion inference (uses cached crop)
        Slot 2: NMM         (~35ms)  — grammar detection
      Cached outputs are reused on non-active slots.
    """
    import cv2
    import time
    from src.governor        import FrameGovernor
    from src.preprocessor    import apply_clahe_channels
    from src.shared_memory   import AtomicSharedMemory

    # Import Nima's components — read existing file signatures before calling
    from processes.face_detector      import create_face_detector
    from processes.coasting_matrix    import CoastingMatrix
    from processes.emotion_inference  import HSEmotionInference, EmotionResult
    from processes.nmm_classifier     import NMMClassifier, NMMContext

    log = logging.getLogger("VisionProducer")

    # Initialise all components once at process startup
    governor  = FrameGovernor()
    writer    = AtomicSharedMemory(shared_array, lock)
    detector  = create_face_detector()   # backend selected by FACE_BACKEND
    coasting  = CoastingMatrix()
    emotion   = HSEmotionInference()
    nmm       = NMMClassifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam — exiting vision producer.")
        return

    def frame_stream():
        while run_flag.value:
            ret, frame = cap.read()
            if ret:
                yield frame

    log.info("Vision producer started.")
    governed = governor.regulate(frame_stream())

    WINDOW_NAME = "Signet Aid — Emotion Monitor"
    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)

    # ── Persistent cached state for interleaved inference scheduling ────────
    # On this hardware, RetinaFace on DirectML alone takes ~130ms — far beyond
    # the 33ms budget. Running ALL models every frame gives ~150ms/frame (~7 FPS).
    #
    # Solution: Interleave heavy inference in round-robin slots so that AT MOST
    # one expensive model runs per frame:
    #   Slot 0 (frame % 3 == 0): RetinaFace  (~130ms) — face detection
    #   Slot 1 (frame % 3 == 1): HSEmotion   (~35ms)  — emotion inference
    #   Slot 2 (frame % 3 == 2): NMM         (~35ms)  — grammar detection
    #
    # Fast frames (slot 1/2) complete in ~35ms, bringing the average down.
    # Each model still runs at ~(FPS/3) ≈ 5–10 Hz, adequate for all signals.
    frame_count = 0
    from processes.face_detector import DetectionResult
    last_det = DetectionResult(bbox=None, confidence=0.0, face_crop=None)
    last_emo = EmotionResult(valence=0.0, arousal=0.0)
    last_nmm_ctx = NMMContext()

    # Issue 4: Time-based overexposure warning throttle (≤1 per second)
    last_overexposure_warn_ts = 0.0
    # Per-frame inference error throttle (≤1 log per second) so a recurring
    # bad-frame error does not flood the log.
    last_infer_err_ts = 0.0

    for raw_frame in governed:
        frame_count += 1
        slot = frame_count % 3   # round-robin inference slot

        # ── Step 1: Two-pass CLAHE (JJ) — runs every frame (cheap: ~5ms) ──
        result = apply_clahe_channels(raw_frame)
        # apply_clahe_channels returns a TUPLE: (frame, is_overexposed)
        clahe_frame, is_overexposed = result
        if clahe_frame is None:
            continue

        # Issue 4: Throttle overexposure warnings to at most once per second.
        # Previously fired every frame (~30/sec), drowning diagnostic output.
        if is_overexposed:
            now = time.monotonic()
            if now - last_overexposure_warn_ts > 1.0:
                log.warning("Overexposure — signer should move away from bright light.")
                last_overexposure_warn_ts = now

        # ── Step 2: Interleaved inference (one heavy model per frame) ─────
        #
        # Slot 0: RetinaFace face detection (~130ms on DirectML)
        #   Updates last_det with fresh bounding box, confidence, face crop.
        #
        # Slot 1: HSEmotion emotion inference (~35ms)
        #   Uses the cached face crop from last_det. Only runs if a valid
        #   crop exists. Updates last_emo with fresh V/A scores.
        #
        # Slot 2: NMM/MediaPipe grammar detection (~35ms)
        #   Runs on the full CLAHE frame (MediaPipe has its own face mesh).
        #   Updates last_nmm_ctx with fresh grammar flags.
        #
        # Non-active slots reuse their cached values, so downstream code
        # (coasting matrix, shared memory, overlay) always has valid data.

        # Steps 2-5 run model inference, which can occasionally throw on a
        # single malformed frame / transient backend error. Guard the whole
        # block so ONE bad frame is logged and skipped instead of killing the
        # vision producer (which would shut down the entire system).
        try:
            if slot == 0:
                # RetinaFace — the heaviest inference on DirectML
                last_det = detector.detect(clahe_frame)
            elif slot == 1:
                # HSEmotion — uses cached face crop from last RetinaFace run
                if last_det.face_crop is not None:
                    last_emo = emotion.infer(last_det.face_crop)
            else:
                # NMM/MediaPipe — runs on full frame, independent of detection
                last_nmm_ctx = nmm.classify(clahe_frame)

            # ── Step 3: Coasting matrix — runs every frame (cheap: pure math) ──
            # Uses cached detection confidence and emotion values.
            if last_det.face_crop is not None:
                coast = coasting.update(last_det.confidence, last_emo.valence, last_emo.arousal)
            else:
                coast = coasting.update(0.0, 0.0, 0.0)

            # ── Step 4: NMM dampening — runs every frame (cheap: multiply) ─────
            eff_v, eff_a = nmm.apply_dampening(coast.valence, coast.arousal, last_nmm_ctx)

            # ── Step 5: Atomic write to shared memory (JJ) ────────────────────
            writer.write_metrics(
                valence=eff_v,
                arousal=eff_a,
                detection_confidence=last_det.confidence,
                tracking_lost=coast.tracking_lost,
                is_yn_question=last_nmm_ctx.is_yn_question,
                is_wh_question=last_nmm_ctx.is_wh_question,
                is_negation=last_nmm_ctx.is_negation,
                force=coast.tracking_lost
            )
        except Exception as e:
            now = time.monotonic()
            if now - last_infer_err_ts > 1.0:
                log.warning(f"Frame {frame_count} inference error (skipping frame): {e}")
                last_infer_err_ts = now
            continue

        # ── Step 6: Live preview display ──────────────────────────────────
        if not headless:
            display_frame = _draw_overlay(
                clahe_frame.copy(), last_det, coast, eff_v, eff_a, last_nmm_ctx, is_overexposed
            )
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log.info("'q' pressed — requesting shutdown.")
                run_flag.value = False
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    # Issue 5: Explicitly release InsightFace before process exit to prevent
    # the model being re-loaded during Python's object finalization.
    detector.close()
    nmm.close()
    log.info("Vision producer shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS 2 & 3 — Imports for Audio Consumer & Audio Player
# ─────────────────────────────────────────────────────────────────────────────
from processes.audio_consumer import audio_consumer
from processes.audio_player import audio_player


# ─────────────────────────────────────────────────────────────────────────────
# DEMO TEXT FEEDER (dummy text for TTS testing)
# ─────────────────────────────────────────────────────────────────────────────
def dummy_text_feeder(text_queue, shared_array, lock, run_flag, interval=10.0,
                      emotion_matched=True):
    """
    Pushes sample sentences into text_queue at regular intervals.

    Two modes:
      emotion_matched=True  (default): selects the sentence whose NMM flags and
        VA quadrant BEST MATCH the live emotion. Looks congruent in a demo, but
        is CIRCULAR — it picks text to fit the expression, so the fusion can
        never be seen to be wrong. Use only for a "happy path" showcase.

      emotion_matched=False (--demo-fixed): plays a FIXED rotating script of
        varied sentence types regardless of the live expression. This honestly
        exercises the text→sentence-type→prosody path and lets you SEE the
        system handle (or mishandle) each case. Prefer this for evaluation.
    """
    import random

    # Fixed, expression-independent script covering each sentence type so the
    # text/fusion path is exercised honestly (used when emotion_matched=False).
    FIXED_SCRIPT = [
        "You go to the store",            # Y/N only if brows raised (face decides)
        "What happened today",            # WH — text-driven question
        "I do not agree with that",       # negation — text-driven
        "The bus is arriving now",        # plain statement
        "Where did you put the keys",     # WH — text-driven question
        "I am so happy to see you",       # emotional statement (no question)
        "Are you feeling okay",           # Y/N (ends ambiguous; face/brow decides)
    ]

    # Sentence pool: (text, yn_question, wh_question, negation, valence_sign, arousal_sign)
    # valence_sign: +1 = positive, -1 = negative, 0 = neutral
    # arousal_sign: +1 = high energy, -1 = low energy, 0 = neutral
    SENTENCE_POOL = [
        # ── Y/N Questions (raised eyebrows) ─────────────────────────────
        ("Are you feeling okay today?",            True,  False, False, +1,  0),
        ("Did you enjoy the show last night?",     True,  False, False, +1,  0),
        ("Is everything alright with you?",        True,  False, False,  0,  0),
        ("Can you believe that just happened?",    True,  False, False, -1, +1),
        ("Are you coming to the meeting later?",   True,  False, False,  0,  0),

        # ── WH Questions (furrowed brows) ────────────────────────────────
        ("What happened to you today?",            False, True,  False, -1,  0),
        ("Where did you leave the keys?",          False, True,  False,  0,  0),
        ("Who told you about the news?",           False, True,  False,  0, +1),
        ("Why are you feeling so down?",           False, True,  False, -1,  0),
        ("How did you manage to do that?",         False, True,  False, +1, +1),

        # ── Negation (head shake) ────────────────────────────────────────
        ("I do not agree with that decision.",     False, False, True,  -1,  0),
        ("That is not what I expected at all.",    False, False, True,  -1, +1),
        ("No, I never said that.",                 False, False, True,  -1, +1),
        ("I cannot believe that happened.",        False, False, True,  -1, +1),

        # ── Happy / Positive high arousal ────────────────────────────────
        ("I am so excited about this opportunity!", False, False, False, +1, +1),
        ("This is wonderful news, I am thrilled.",  False, False, False, +1, +1),
        ("I feel really great about our progress.", False, False, False, +1, +1),

        # ── Calm / Positive low arousal ──────────────────────────────────
        ("Everything feels calm and peaceful today.", False, False, False, +1, -1),
        ("I am content and at ease right now.",       False, False, False, +1, -1),
        ("Life is going smoothly, I am grateful.",    False, False, False, +1, -1),

        # ── Sad / Negative low arousal ───────────────────────────────────
        ("That makes me very sad to hear.",        False, False, False, -1, -1),
        ("I feel really low and tired today.",     False, False, False, -1, -1),
        ("It has been a very difficult time.",     False, False, False, -1, -1),

        # ── Angry / Negative high arousal ───────────────────────────────
        ("I am furious about what just happened.", False, False, False, -1, +1),
        ("This situation is completely unacceptable.", False, False, False, -1, +1),
        ("I cannot stand this any longer.",        False, False, False, -1, +1),

        # ── Neutral fallback ─────────────────────────────────────────────
        ("Hello, it is good to see you.",          False, False, False,  0,  0),
        ("Let me tell you something important.",   False, False, False,  0,  0),
    ]

    log = logging.getLogger("DemoFeeder")
    _mode = "emotion-matched" if emotion_matched else "fixed-script"
    log.info(f"Demo text feeder started ({_mode} mode). Pushing text every {interval}s.")

    # Wait for TTS model to load before sending first text
    time.sleep(15.0)

    def _read_live_state():
        """Read the current emotion state from the vision producer."""
        with lock:
            v    = shared_array[0]
            a    = shared_array[1]
            conf = shared_array[2]
            lost = shared_array[3]
            yn   = shared_array[4] == 1.0
            wh   = shared_array[5] == 1.0
            neg  = shared_array[6] == 1.0
        return v, a, conf, lost, yn, wh, neg

    def _pick_sentence(v, a, yn, wh, neg):
        """
        Score each sentence by how well it matches the live NMM flags and
        VA quadrant. Return the best-matching sentence text.
        NMM flags are the primary signal; VA quadrant is secondary.
        """
        v_sign = +1 if v > 0.15 else (-1 if v < -0.15 else 0)
        a_sign = +1 if a > 0.15 else (-1 if a < -0.15 else 0)

        best_score   = -1
        best_matches = []

        for (text, s_yn, s_wh, s_neg, s_vsign, s_asign) in SENTENCE_POOL:
            score = 0

            # NMM flags are worth 3 pts each (highest priority)
            if s_yn  == yn:  score += 3
            if s_wh  == wh:  score += 3
            if s_neg == neg: score += 3

            # VA quadrant secondary signal (1 pt each)
            if s_vsign == v_sign: score += 1
            if s_asign == a_sign: score += 1

            if score > best_score:
                best_score   = score
                best_matches = [text]
            elif score == best_score:
                best_matches.append(text)

        chosen = random.choice(best_matches)
        log.info(
            f"[Sentence match] score={best_score} | yn={yn} wh={wh} neg={neg} "
            f"V={v:+.2f}({'+' if v_sign==1 else '-' if v_sign==-1 else '0'}) "
            f"A={a:+.2f}({'+' if a_sign==1 else '-' if a_sign==-1 else '0'}) "
            f"→ '{chosen}'"
        )
        return chosen

    script_idx = 0
    while run_flag.value:

        # Step 1: Read live emotion state from the vision producer
        v, a, conf, lost, yn, wh, neg = _read_live_state()

        # Step 2: Pick the sentence — matched to expression, or fixed script.
        if emotion_matched:
            sentence = _pick_sentence(v, a, yn, wh, neg)
        else:
            sentence = FIXED_SCRIPT[script_idx % len(FIXED_SCRIPT)]
            script_idx += 1

        # Step 3: Let the vision producer continue writing; do NOT overwrite
        #         shared memory with hardcoded values — use the real camera data.
        log.info(
            f"Demo sentence selected ({_mode}): '{sentence}' "
            f"| live V={v:+.2f} A={a:+.2f} yn={yn} wh={wh} neg={neg}"
        )

        # Step 4: Push the matched sentence to the TTS queue
        text_queue.put(sentence)

        # Step 5: Sleep until the next interval (in small increments so we
        #         can exit cleanly on shutdown)
        for _ in range(int(interval * 10)):
            if not run_flag.value:
                break
            time.sleep(0.1)

    log.info("Demo text feeder shut down.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Signet Aid Emotion Module")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--demo", action="store_true",
                        help="Enable demo mode: push dummy text to TTS every 10 seconds")
    parser.add_argument("--demo-fixed", action="store_true",
                        help="With --demo, play a fixed expression-INDEPENDENT script "
                             "(honest test of the text/fusion path) instead of picking "
                             "text to match the detected emotion.")
    args = parser.parse_args()

    # Required on Windows and macOS when using CUDA in subprocesses
    multiprocessing.set_start_method('spawn', force=True)

    from config.settings import BUFFER_ELEMENT_COUNT

    # Shared memory initialisation
    # [0]=valence  [1]=arousal  [2]=detection_confidence  [3]=tracking_lost
    # [4]=is_yn_question  [5]=is_wh_question  [6]=is_negation
    # tracking_lost starts as 1.0 (True) — no face confirmed yet at startup
    # detection_confidence starts as 0.0 — avoids false positive on first frame
    _shm_init = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert len(_shm_init) == BUFFER_ELEMENT_COUNT, (
        f"shared memory init has {len(_shm_init)} slots but "
        f"BUFFER_ELEMENT_COUNT={BUFFER_ELEMENT_COUNT} — keep them in sync."
    )
    shared_array = Array(ctypes.c_double, _shm_init)
    lock         = Lock()
    text_queue   = Queue()    # translation module pushes sentence strings here
    audio_queue  = Queue()    # audio_consumer pushes audio chunks here
    run_flag     = Value(ctypes.c_bool, True)

    p1 = Process(target=vision_producer,
                 args=(shared_array, lock, text_queue, run_flag, args.headless),
                 name="VisionProducer")
    p2 = Process(target=audio_consumer,
                 args=(shared_array, lock, text_queue, audio_queue, run_flag),
                 name="AudioConsumer")
    p3 = Process(target=audio_player,
                 args=(audio_queue, run_flag),
                 name="AudioPlayer")

    # Optional demo text feeder (Process 4)
    p4 = None
    if args.demo:
        p4 = Process(target=dummy_text_feeder,
                     args=(text_queue, shared_array, lock, run_flag, 10.0,
                           not args.demo_fixed),
                     name="DemoFeeder")

    try:
        p1.start()
        p2.start()
        p3.start()
        if p4:
            p4.start()
            logging.info("All four Signet Aid processes started (demo mode). Press Ctrl+C to stop.")
        else:
            logging.info("All three Signet Aid processes started. Press Ctrl+C to stop.")

        while True:
            time.sleep(1.0)
            with lock:
                v    = shared_array[0]
                a    = shared_array[1]
                conf = shared_array[2]
                lost = shared_array[3]
                yn   = shared_array[4]
                wh   = shared_array[5]
                neg  = shared_array[6]
            logging.info(
                f"[STATUS] V={v:+.3f}  A={a:+.3f}  "
                f"conf={conf:.2f}  tracking_lost={bool(lost == 1.0)}  "
                f"yn={bool(yn == 1.0)}  wh={bool(wh == 1.0)}  neg={bool(neg == 1.0)}"
            )
            if not p1.is_alive():
                logging.error("Vision producer died — initiating shutdown.")
                break
            if not p2.is_alive():
                logging.error("Audio consumer died — initiating shutdown.")
                break
            if not p3.is_alive():
                logging.error("Audio player died — initiating shutdown.")
                break

    except KeyboardInterrupt:
        logging.info("Ctrl+C received — shutting down.")
    finally:
        run_flag.value = False
        processes = [p1, p2, p3]
        if p4:
            processes.append(p4)
        for p in processes:
            if p.is_alive():
                p.join(timeout=3.0)
                if p.is_alive():
                    p.terminate()
                    logging.warning(f"{p.name} did not stop cleanly — force terminated.")
        logging.info("Signet Aid emotion module shut down.")
