"""
test_tts.py — Standalone Chatterbox TTS test for Signet Aid
Tests VAToChatterbox parameter mapping and audio generation.
Run from emotion_module/: python processes/test_tts.py
"""
import os
import sys
# Ensure the parent directory is in sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def test_va_mapper():
    from processes.audio_consumer import VAToChatterbox
    mapper = VAToChatterbox()

    test_cases = [
        # (valence, arousal, yn_sus, neg_sus, expected_style_note)
        (0.0,   0.0,   False, False, "NEUTRAL   expect ~0.50, ~0.50"),
        (0.8,   0.7,   False, False, "JOY       high exag, med-low cfg"),
        (-0.7,  0.8,   False, False, "ANGER     high exag, very low cfg"),
        (-0.6, -0.4,   False, False, "SAD       low-med exag, high cfg"),
        (0.0,   0.0,   True,  False, "YN_QUEST  neutral params, text gets ?"),
        (0.5,   0.6,   False, True,  "NEGATION  dampened exag"),
    ]

    print("\nVA to Chatterbox Parameter Mapping Test:")
    print(f"{'Case':<35} {'exag':>6} {'cfg':>6}")
    print("-" * 50)
    for v, a, yn, neg, label in test_cases:
        exag, cfg = mapper.compute(v, a, yn, neg)
        print(f"{label:<35} {exag:>6.3f} {cfg:>6.3f}")


def test_text_modification():
    from processes.audio_consumer import (
        VAToChatterbox, resolve_sentence_type,
    )

    # (text, yn_face_flag, expected_type, expected_text)
    cases = [
        # Y/N: text gives no evidence, gated brow flag decides → "?"
        ("I need help",          True,  "yn",        "I need help?"),
        # WH: text is authoritative; brow irrelevant; now gets "?"
        ("Where is the exit",    False, "wh",        "Where is the exit?"),
        # Negation: lexical, no "?"
        ("I cannot do this.",    False, "neg",       "I cannot do this."),
        # CRITICAL regression: emotional statement (surprise). The concordance
        # gate suppressed the Y/N brow flag (yn_face=False), so the statement
        # is NOT turned into a question.
        ("That just happened",   False, "statement", "That just happened"),
        # WH wins even if a brow flag is somehow set.
        ("Why are you upset",    True,  "wh",        "Why are you upset?"),
    ]

    print("\nText Modification Test:")
    for text, yn_face, expected_type, expected_text in cases:
        rtype  = resolve_sentence_type(text, yn_face)
        result = VAToChatterbox.modify_text(text, rtype)
        ok = (rtype == expected_type) and (result == expected_text)
        status = "[OK]" if ok else "[X]"
        print(f"  {status} '{text}' (yn_face={yn_face}) -> [{rtype}] '{result}'")
        assert rtype == expected_type, f"type: expected '{expected_type}', got '{rtype}'"
        assert result == expected_text, f"text: expected '{expected_text}', got '{result}'"


def test_ema_filter():
    from processes.audio_consumer import EMAFilter

    ema = EMAFilter(warmup_frames=5)

    # Warmup period
    for i in range(5):
        v, a, ready = ema.update(0.8, 0.5, 1.0, False, False)
        print(f"  Warmup frame {i+1}: V={v:.3f} A={a:.3f} ready={ready}")

    # EMA should be tracking the signal now
    v, a, ready = ema.update(0.8, 0.5, 1.0, False, False)
    assert ready, "EMA should be warmed up after 6 frames"
    assert 0.3 < v < 0.9, f"EMA V out of expected range: {v}"

    # Grammar flag should dampen
    for _ in range(11):
        v, a, _ = ema.update(0.0, 0.0, 1.0, False, True)  # grammar active
    assert abs(v) < 0.3, f"Grammar dampening not working: V={v}"
    print(f"  After grammar dampening: V={v:.3f} A={a:.3f}")
    print("[OK] EMA filter OK")


def test_sustain_checker():
    from processes.audio_consumer import SustainChecker

    sc = SustainChecker(required_frames=5)

    # Should not sustain before 5 frames
    for i in range(4):
        yn_s, _, _ = sc.update(True, False, False)
        assert not yn_s, f"Sustained too early at frame {i+1}"

    # Should sustain at exactly 5 frames
    yn_s, _, _ = sc.update(True, False, False)
    assert yn_s, "Should be sustained after 5 frames"

    # Reset on False
    yn_s, _, _ = sc.update(False, False, False)
    assert not yn_s, "Should reset on False"
    print("[OK] Sustain checker OK")


def test_chatterbox_synthesis():
    """
    Actual Chatterbox model load + synthesis test.
    Generates audio for each emotion quadrant and saves to .wav files.
    """
    import torch
    import soundfile as sf
    from chatterbox.tts import ChatterboxTTS
    from processes.audio_consumer import VAToChatterbox

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading Chatterbox on {device} ...")
    tts    = ChatterboxTTS.from_pretrained(device=device)
    mapper = VAToChatterbox()

    test_sentences = [
        (0.0,   0.0,   "This is the default neutral voice."),
        (0.8,   0.7,   "I am so happy to see you today!"),
        (-0.7,  0.8,   "I need you to listen to me right now."),
        (-0.6, -0.4,   "I am feeling very tired and sad."),
        (0.4,  -0.3,   "I feel calm and at peace."),
    ]

    print(f"\n{'Emotion':<12} {'exag':>6} {'cfg':>6}  Text")
    print("-" * 70)

    for v, a, text in test_sentences:
        exag, cfg = mapper.compute(v, a, False, False)
        label     = _va_label(v, a)
        print(f"{label:<12} {exag:>6.3f} {cfg:>6.3f}  {text}")

        wav = tts.generate(text=text, exaggeration=exag, cfg_weight=cfg)

        # Save to wav for listening
        filename = f"test_{label.lower()}.wav"
        if hasattr(wav, 'numpy'):
            wav = wav.numpy()
        elif hasattr(wav, 'cpu'):
            wav = wav.cpu().numpy()
        else:
            wav = np.squeeze(wav)
        sf.write(filename, wav.squeeze(), 24000)
        print(f"           Saved: {filename}")

    print("\n[OK] Chatterbox synthesis test complete. Listen to the .wav files.")


def _va_label(v, a):
    if abs(v) < 0.2 and abs(a) < 0.2: return "Neutral"
    if v >= 0.3 and a >= 0.3:          return "Joy"
    if v < -0.3 and a >= 0.3:          return "Anger"
    if v < -0.3 and a < 0:             return "Sadness"
    if v >= 0.3 and a < 0:             return "Calm"
    return "Mixed"


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Signet Aid — Chatterbox TTS Integration Test")
    print("=" * 60)

    # Run logic tests first (no model required)
    try:
        test_va_mapper()
        test_text_modification()
        test_ema_filter()
        test_sustain_checker()
        print("\nAll logic tests (no model required) passed successfully!")
    except Exception as e:
        print(f"\nLogic tests failed: {e}")
        sys.exit(1)

    # Run model synthesis test if requested
    if "--model" in sys.argv:
        try:
            test_chatterbox_synthesis()
        except Exception as e:
            print(f"\nModel synthesis test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nTo test actual model loading & wav generation, run with: python processes/test_tts.py --model")
