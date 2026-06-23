"""
test_integration.py — Full stack integration test for merged Signet Aid module.
Tests JJ's src/ components + Nima's processes/ components together.
Run from signet_aid/emotion_module/: python test_integration.py
DO NOT confuse with test_nima_integration.py (Nima's standalone webcam test).
"""
import sys, logging, ctypes
import numpy as np
logging.basicConfig(level=logging.INFO)


def test_governor():
    import time
    from src.governor import FrameGovernor
    results = []
    def mock_stream():
        for _ in range(5):
            yield np.zeros((480, 640, 3), dtype=np.uint8)
    gov = FrameGovernor(target_fps=30.0)
    t0 = time.monotonic()
    for _ in gov.regulate(mock_stream()):
        results.append(time.monotonic())
    elapsed = results[-1] - t0
    # 5 frames at 30fps ≈ 0.133s, allow generous tolerance
    assert elapsed >= 0.10, f"Governor too fast: {elapsed:.3f}s"
    print(f"[OK] Governor OK ({elapsed:.3f}s for 5 frames at 30 FPS)")


def test_preprocessor():
    from src.preprocessor import apply_clahe_channels
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = apply_clahe_channels(dummy)
    assert isinstance(result, tuple), "preprocessor must return a tuple"
    frame, overexposed = result
    assert frame is not None, "CLAHE returned None frame"
    assert frame.shape == (480, 640, 3)
    assert isinstance(overexposed, bool)
    # Test overexposure detection with a white frame
    white = np.full((480, 640, 3), 255, dtype=np.uint8)
    _, is_blown = apply_clahe_channels(white)
    assert is_blown, "All-white frame should trigger overexposure flag"
    print("[OK] Preprocessor OK (CLAHE + overexposure detection)")


def test_shared_memory():
    from multiprocessing import Array, Lock
    from src.shared_memory import AtomicSharedMemory
    arr  = Array(ctypes.c_double, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    lock = Lock()
    w    = AtomicSharedMemory(arr, lock)
    ok   = w.write_metrics(0.5, -0.3, 0.9, False, is_yn_question=True, is_wh_question=False, is_negation=True, force=True)
    assert ok, "Write returned False"
    v, a, c, lost, yn, wh, neg = w.read_metrics()
    assert abs(v - 0.5) < 0.001,   f"Valence wrong: {v}"
    assert abs(a - (-0.3)) < 0.001, f"Arousal wrong: {a}"
    assert abs(c - 0.9) < 0.001,   f"Confidence wrong: {c}"
    assert not lost,                "tracking_lost should be False"
    assert yn,                      "is_yn_question should be True"
    assert not wh,                  "is_wh_question should be False"
    assert neg,                     "is_negation should be True"
    # Test startup defaults
    arr2  = Array(ctypes.c_double, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    w2    = AtomicSharedMemory(arr2, lock)
    _, _, c2, lost2, _, _, _ = w2.read_metrics()
    assert c2 == 0.0, "Startup confidence must be 0.0 (not 0.65)"
    assert lost2,     "Startup tracking_lost must be True"
    print("[OK] Shared memory OK (atomic write/read + startup state)")


def test_face_detector():
    from processes.face_detector import FaceDetector
    det   = FaceDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    r     = det.detect(blank)
    assert r is not None
    assert hasattr(r, 'confidence')
    assert hasattr(r, 'face_crop')
    assert 0.0 <= r.confidence <= 1.0
    print(f"[OK] Face detector OK (blank frame conf={r.confidence:.2f})")


def test_coasting():
    from processes.coasting_matrix import CoastingMatrix
    cm = CoastingMatrix()
    r  = cm.update(0.9, 0.6, 0.3)
    assert abs(r.valence - 0.6) < 0.01
    assert not r.tracking_lost
    for _ in range(30):
        r = cm.update(0.0, 0.0, 0.0)
    assert r.tracking_lost,    "Must be tracking_lost after 30 bad frames"
    assert abs(r.valence) < 0.1, f"VA must decay near 0: got {r.valence}"
    print("[OK] Coasting matrix OK (valid frames + decay + tracking_lost)")


def test_emotion_inference():
    from processes.emotion_inference import HSEmotionInference
    eng  = HSEmotionInference()
    crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    r    = eng.infer(crop)
    assert hasattr(r, 'valence') and hasattr(r, 'arousal')
    assert -1.0 <= r.valence <= 1.0, f"Valence out of range: {r.valence}"
    assert -1.0 <= r.arousal <= 1.0, f"Arousal out of range: {r.arousal}"
    print(f"[OK] HSEmotion OK (V={r.valence:+.3f} A={r.arousal:+.3f})")


def test_nmm():
    from processes.nmm_classifier import NMMClassifier, NMMContext
    clf   = NMMClassifier()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx   = clf.classify(blank)
    assert not ctx.any_active, "Blank frame must produce no NMM flags"
    v, a  = clf.apply_dampening(0.8, 0.6,
                NMMContext(is_yn_question=True, any_active=True))
    assert abs(v - 0.8 * 0.25) < 0.01, f"Dampening wrong: {v}"
    clf.close()
    print("[OK] NMM classifier OK (blank frame + dampening)")


def test_full_pipeline_one_frame():
    """Simulate one complete frame through the entire merged stack."""
    import ctypes
    from multiprocessing import Array, Lock
    from src.preprocessor   import apply_clahe_channels
    from src.shared_memory  import AtomicSharedMemory
    from processes.face_detector     import FaceDetector
    from processes.coasting_matrix   import CoastingMatrix
    from processes.emotion_inference import HSEmotionInference
    from processes.nmm_classifier    import NMMClassifier

    arr     = Array(ctypes.c_double, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    lock    = Lock()
    writer  = AtomicSharedMemory(arr, lock)
    det     = FaceDetector()
    coast   = CoastingMatrix()
    emo     = HSEmotionInference()
    nmm     = NMMClassifier()

    raw   = np.full((480, 640, 3), 100, dtype=np.uint8)
    frame, _ = apply_clahe_channels(raw)
    d     = det.detect(frame)
    if d.face_crop is not None:
        e = emo.infer(d.face_crop)
        c = coast.update(d.confidence, e.valence, e.arousal)
    else:
        c = coast.update(0.0, 0.0, 0.0)
    ctx  = nmm.classify(frame)
    ev, ea = nmm.apply_dampening(c.valence, c.arousal, ctx)
    ok   = writer.write_metrics(
        valence=ev,
        arousal=ea,
        detection_confidence=d.confidence,
        tracking_lost=c.tracking_lost,
        is_yn_question=ctx.is_yn_question,
        is_wh_question=ctx.is_wh_question,
        is_negation=ctx.is_negation,
        force=True
    )
    v, a, conf, lost, yn, wh, neg = writer.read_metrics()
    nmm.close()

    assert ok,                "Shared memory write failed"
    assert -1.0 <= v <= 1.0, f"Final valence out of range: {v}"
    assert -1.0 <= a <= 1.0, f"Final arousal out of range: {a}"
    print(f"[OK] Full pipeline frame OK (V={v:+.3f} A={a:+.3f} conf={conf:.2f} lost={lost})")


if __name__ == "__main__":
    tests = [
        test_governor,
        test_preprocessor,
        test_shared_memory,
        test_face_detector,
        test_coasting,
        test_emotion_inference,
        test_nmm,
        test_full_pipeline_one_frame,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__} FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("All components merged correctly. Run main.py to start.")
    else:
        print("Fix failing tests before running main.py.")
