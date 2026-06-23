"""
audio_consumer.py
Signet Aid — Process 2: Audio Consumer Orchestrator

Full implementation pipeline per sentence:
  1. Read VA + NMM flags from shared memory (30 FPS sampling rate)
  2. Apply 15-frame EMA smoothing to VA scores
  3. Check neutral dead band — skip voice modification if near (0,0)
  4. Map smoothed VA + NMM flags to Chatterbox parameters
  5. Wait for text token from text_queue (translation module output)
  6. Modify text based on NMM flags (punctuation)
  7. Generate speech via Chatterbox Streaming
  8. Push audio chunks to audio_queue for Process 3
"""

import re
import time
import logging
import numpy as np
from src.shared_memory import AtomicSharedMemory

# ── Linguistic sentence-type detection (ASL → English text) ──────────────────
# The translation module emits full English text. For ASL, the text reliably
# carries WH-questions (a WH-word is present) and lexical negation, but NOT
# yes/no questions — "YOU GO STORE" + brow-raise translates to the same words
# as the statement. So:
#   - WH / negation  → decided by the TEXT (high precision).
#   - Y/N            → decided by the (concordance-gated) brow flag, and only
#                      when the text itself gives no sentence-type evidence.
_WH_RE  = re.compile(r"\b(what|where|who|whom|whose|why|how|when|which)\b", re.IGNORECASE)
_NEG_RE = re.compile(
    r"\b(not|no|never|none|nothing|cannot|can't|won't|don't|doesn't|didn't|"
    r"isn't|aren't|wasn't|weren't|haven't|hasn't)\b",
    re.IGNORECASE,
)
# Auxiliary/modal-initial wording is a strong Y/N-question cue *when* the
# translation emits fluent English ("Are you coming"). If it emits ASL gloss
# order instead, this simply won't match and the brow flag decides Y/N.
_YN_AUX_RE = re.compile(
    r"^(are|is|am|was|were|do|does|did|can|could|will|would|shall|should|"
    r"have|has|had|may|might|must)\b",
    re.IGNORECASE,
)


def sentence_type_from_text(text: str) -> str:
    """
    Classify sentence type from the translated English text alone.
    Returns one of: "wh", "neg", "yn", "unknown".
    "unknown" means the text gives no evidence — only then does the brow vote.

    Order matters: negation is checked before the auxiliary-initial Y/N cue so
    that "Do not touch" resolves to negation, not a yes/no question.
    """
    t = text.strip().lower()
    if _WH_RE.search(t):
        return "wh"
    if _NEG_RE.search(t):
        return "neg"
    if t.endswith("?") or _YN_AUX_RE.match(t):
        return "yn"          # explicit '?' or fronted auxiliary → yes/no question
    return "unknown"


def resolve_sentence_type(text: str, yn_face: bool) -> str:
    """
    Fuse the linguistic channel (text) with the face channel (concordance-gated
    Y/N brow flag). Text wins for WH / negation / explicit '?'. The brow flag
    only decides Y/N when the text is silent — which is exactly the ASL case
    the words cannot disambiguate.

    Returns one of: "wh", "neg", "yn", "statement".
    """
    t = sentence_type_from_text(text)
    if t != "unknown":
        return t
    return "yn" if yn_face else "statement"

class EMAFilter:
    """
    Exponential Moving Average over 15-frame window.
    Smooths VA scores to prevent voice flickering from micro-expressions.

    EMA formula: EMA_t = alpha * current + (1 - alpha) * EMA_(t-1)

    Alpha values (CRITICAL — do NOT use 0.0 or 0.75, these are mathematically wrong):
        Normal signing:     alpha = 0.30  (30% weight on new data — responsive)
        Grammar flag active: alpha = 0.08  (8% weight — heavily dampened)
        Low confidence:      alpha = 0.05  (5% weight — near-frozen)
        tracking_lost:       alpha = 0.00  (frozen on last valid baseline)

    Warm-up: suppress output for first WARMUP_FRAMES frames.
    """

    def __init__(self, warmup_frames: int = 15):
        self.ema_v:     float = 0.0
        self.ema_a:     float = 0.0
        self.frame_count: int = 0
        self.warmup_frames    = warmup_frames

    @property
    def warmed_up(self) -> bool:
        return self.frame_count >= self.warmup_frames

    def update(self,
               v: float,
               a: float,
               conf: float,
               tracking_lost: bool,
               is_grammar_active: bool) -> tuple:
        """
        Update EMA with one frame's VA reading.
        Returns (ema_v, ema_a, warmed_up).
        """
        if tracking_lost:
            alpha = 0.00    # frozen — coasting matrix already handles decay
        elif conf < 0.75:
            alpha = 0.05    # low confidence frame
        elif is_grammar_active:
            alpha = 0.08    # grammar NMM active — dampen toward baseline
        else:
            alpha = 0.30    # normal signing

        self.ema_v = alpha * v + (1.0 - alpha) * self.ema_v
        self.ema_a = alpha * a + (1.0 - alpha) * self.ema_a
        self.frame_count += 1
        return self.ema_v, self.ema_a, self.warmed_up


class SustainChecker:
    """
    Ensures punctuation-driven voice changes only fire when a signal
    has been held continuously for SUSTAIN_FRAMES_REQUIRED frames.
    Prevents single-frame spikes from triggering voice changes.
    """

    def __init__(self, required_frames: int = 10):
        self.required  = required_frames
        self._counters = {"yn": 0, "wh": 0, "neg": 0}

    def update(self, yn: bool, wh: bool, neg: bool) -> tuple:
        """Returns (yn_sustained, wh_sustained, neg_sustained)."""
        for key, active in [("yn", yn), ("wh", wh), ("neg", neg)]:
            if active:
                self._counters[key] = min(
                    self._counters[key] + 1,
                    self.required + 1
                )
            else:
                self._counters[key] = 0

        return (
            self._counters["yn"]  >= self.required,
            self._counters["wh"]  >= self.required,
            self._counters["neg"] >= self.required,
        )


class VAToChatterbox:
    """
    Maps smoothed VA scores + sustained NMM flags to Chatterbox parameters.

    Chatterbox parameters:
        exaggeration: float
            0.1  = extremely flat, near-monotone
            0.5  = natural default voice
            0.8  = noticeably expressive
            1.0  = highly dramatic
            1.2+ = over-the-top (use sparingly)
            Drives from: arousal magnitude (how intensely the signer feels)

        cfg_weight: float
            0.1-0.3 = more variation, less controlled (high emotion states)
            0.4-0.6 = balanced (default)
            0.7-0.9 = stable, controlled (calm/neutral states)
            Drives from: emotion quadrant (valence + arousal combination)

    Dead band: if |V| < 0.20 and |A| < 0.20  return defaults (0.50, 0.50)
    """

    # Neutral dead band thresholds
    NEUTRAL_V = 0.20
    NEUTRAL_A = 0.20

    # Chatterbox parameter ranges
    EXAGGERATION_MIN = 0.20
    EXAGGERATION_MAX = 1.10
    CFG_MIN          = 0.15
    CFG_MAX          = 0.85

    def compute(self,
                ema_v: float,
                ema_a: float,
                yn_sustained: bool,
                neg_sustained: bool) -> tuple:
        """
        Returns (exaggeration, cfg_weight).
        """

        # Dead band — no emotion modification
        if abs(ema_v) < self.NEUTRAL_V and abs(ema_a) < self.NEUTRAL_A:
            return 0.50, 0.50

        # exaggeration from arousal magnitude
        arousal_mag   = abs(ema_a)
        exaggeration  = self._linear_map(
            arousal_mag, 0.0, 1.0,
            self.EXAGGERATION_MIN, self.EXAGGERATION_MAX
        )

        # cfg_weight from VA quadrant
        if ema_v < -0.30 and ema_a > 0.30:
            # Anger / Fear quadrant — high energy, negative
            cfg_weight = 0.20
        elif ema_v < -0.30 and ema_a <= 0.0:
            # Sadness / Resignation — low energy, negative
            cfg_weight = 0.65
            exaggeration = min(exaggeration, 0.55)   # cap for sad voice
        elif ema_v >= 0.30 and ema_a > 0.30:
            # Joy / Excitement — high energy, positive
            cfg_weight = 0.30
        elif ema_v >= 0.30 and ema_a <= 0.0:
            # Calm / Content — low energy, positive
            cfg_weight = 0.70
            exaggeration = min(exaggeration, 0.50)
        else:
            cfg_weight = 0.50

        # NMM overrides
        if neg_sustained:
            # Negation — dampen expression, flatten voice
            exaggeration *= 0.40
            cfg_weight    = max(cfg_weight, 0.60)

        # Clamp to valid ranges
        exaggeration = max(self.EXAGGERATION_MIN,
                           min(self.EXAGGERATION_MAX, exaggeration))
        cfg_weight   = max(self.CFG_MIN,
                           min(self.CFG_MAX, cfg_weight))

        return round(exaggeration, 3), round(cfg_weight, 3)

    @staticmethod
    def _linear_map(x, in_min, in_max, out_min, out_max) -> float:
        return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)

    @staticmethod
    def modify_text(text: str, resolved_type: str) -> str:
        """
        Apply sentence-type-driven punctuation.

        Punctuation is driven by the RESOLVED sentence type (text + concordance-
        gated brow), never by a raw brow flag. This prevents an emotional brow
        raise (surprise/fear) on a statement from being turned into a question.

        Both WH and Y/N questions get "?" so Chatterbox's text encoder applies
        question intonation.
        """
        text = text.strip()
        if resolved_type in ("yn", "wh") and not text.endswith("?"):
            # Chatterbox's text encoder reads "?" as a rising-intonation cue
            text = text + "?"
        return text


def audio_consumer(shared_array, lock, text_queue, audio_queue, run_flag):
    """
    Process 2: Audio Consumer Orchestrator
    Reads VA + NMM from shared memory at 30 FPS sampling rate.
    When a text token arrives from the translation module, synthesises
    speech with Chatterbox Streaming using current emotional parameters.
    Pushes audio chunks to audio_queue for Process 3.
    """
    from config.settings   import (EMA_WARMUP_FRAMES, SUSTAIN_FRAMES_REQUIRED,
                                   DETECTION_CONFIDENCE_THRESHOLD)

    log = logging.getLogger("AudioConsumer")

    # Load Chatterbox
    # Load on GPU if available, fall back to CPU
    # Chatterbox Turbo (350M) + HSEmotion (EfficientNet-B0) share GPU safely
    # If VRAM < 4GB: change device to "cpu"
    tts = None
    try:
        import torch
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading Chatterbox TTS on {device} ...")
        tts = ChatterboxTTS.from_pretrained(device=device)
        log.info("Chatterbox TTS loaded successfully.")
    except ImportError:
        log.warning("chatterbox-tts not installed — running in stub mode (no speech synthesis). "
                    "Install with: pip install chatterbox-tts")
    except Exception as e:
        log.warning(f"Failed to load Chatterbox: {e} — running in stub mode.")

    # Initialise helpers
    reader   = AtomicSharedMemory(shared_array, lock)
    ema      = EMAFilter(warmup_frames=EMA_WARMUP_FRAMES)
    sustain  = SustainChecker(required_frames=SUSTAIN_FRAMES_REQUIRED)
    mapper   = VAToChatterbox()

    # Sampling interval — read shared memory at ~30 FPS
    SAMPLE_INTERVAL = 1.0 / 30.0

    log.info("Audio consumer started.")

    while run_flag.value:

        # Step 1: Sample shared memory 
        v, a, conf, lost, yn, wh, neg = reader.read_metrics()
        is_grammar = yn or wh or neg

        # Step 2: EMA smoothing 
        ema_v, ema_a, warmed = ema.update(v, a, conf, lost, is_grammar)

        # Step 3: Sustain check 
        yn_sus, wh_sus, neg_sus = sustain.update(yn, wh, neg)

        # Step 4: Check for incoming text token (non-blocking) 
        if text_queue.empty():
            time.sleep(SAMPLE_INTERVAL)
            continue

        text = text_queue.get_nowait()
        if not isinstance(text, str) or not text.strip():
            continue

        log.info(f"Text received: '{text}'")

        # Step 5: Skip synthesis if not warmed up or tracking lost 
        if not warmed:
            log.info("EMA not warmed up yet — using neutral voice (defaults).")
            exaggeration = 0.50
            cfg_weight   = 0.50
        elif lost:
            log.info("Tracking lost — using neutral voice (defaults).")
            exaggeration = 0.50
            cfg_weight   = 0.50
        else:
            # Step 6: Map VA to Chatterbox params 
            exaggeration, cfg_weight = mapper.compute(
                ema_v, ema_a, yn_sus, neg_sus
            )

        # Step 7: Resolve sentence type (text owns WH/neg; gated brow owns Y/N)
        #         and apply punctuation from the resolved type.
        resolved_type = resolve_sentence_type(text, yn_sus)
        modified_text = VAToChatterbox.modify_text(text, resolved_type)

        log.info(
            f"Synthesising  exag={exaggeration:.3f} cfg={cfg_weight:.3f} "
            f"| V={ema_v:+.2f} A={ema_a:+.2f} "
            f"| type={resolved_type} (yn_face={yn_sus} wh={wh_sus} neg={neg_sus}) "
            f"| '{modified_text}'"
        )

        # Step 8: Chatterbox synthesis 
        if tts is None:
            log.debug(f"Stub mode — skipping synthesis for: '{modified_text}'")
            continue

        try:
            # Use streaming generation for real-time output
            # Each chunk is a numpy float32 array at 24000 Hz sample rate
            # Push chunks to audio_queue as they are generated
            # Process 3 begins playback immediately on first chunk arrival
            if hasattr(tts, 'generate_stream'):
                chunk_idx = 0
                synth_start = time.monotonic()
                for chunk in tts.generate_stream(
                    text=modified_text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                ):
                    # Issue 3: Check shutdown flag between TTS chunks.
                    # Without this, the generator blocks until the full sentence
                    # is synthesised, exceeding main.py's 3-second join timeout
                    # and triggering force termination.
                    if not run_flag.value:
                        log.info("Shutdown requested mid-synthesis — aborting generation.")
                        break

                    if chunk is not None and len(chunk) > 0:
                        chunk_idx += 1
                        chunk_dur = len(chunk) / 24000.0
                        elapsed = time.monotonic() - synth_start
                        # Streaming diagnostic: confirm chunks arrive mid-synthesis
                        # If all chunks arrive at the same elapsed time, generate_stream
                        # is buffering internally (not truly streaming).
                        log.info(
                            f"  chunk {chunk_idx}: {len(chunk)} samples "
                            f"({chunk_dur:.2f}s audio) at t+{elapsed:.2f}s"
                        )
                        # Normalise to float32 in [-1.0, +1.0] if needed
                        if chunk.dtype != np.float32:
                            chunk = chunk.astype(np.float32)
                        audio_queue.put(chunk)
                total_elapsed = time.monotonic() - synth_start
                log.info(f"  synthesis complete: {chunk_idx} chunks in {total_elapsed:.2f}s")
            else:
                wav = tts.generate(
                    text=modified_text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
                # Issue 3: Check shutdown flag after non-streaming generation
                # completes, before pushing the audio buffer to the queue.
                if not run_flag.value:
                    log.info("Shutdown requested after synthesis — discarding audio.")
                elif wav is not None:
                    if hasattr(wav, 'numpy'):
                        chunk = wav.squeeze().numpy()
                    elif hasattr(wav, 'cpu'):
                        chunk = wav.cpu().squeeze().numpy()
                    else:
                        chunk = np.squeeze(wav)
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    audio_queue.put(chunk)

            # Push a None sentinel to signal end of sentence to audio_player
            # (only if we're still running — no point signalling after shutdown)
            if run_flag.value:
                audio_queue.put(None)

        except Exception as e:
            log.error(f"Chatterbox synthesis error: {e}")
            continue

    log.info("Audio consumer shut down.")
