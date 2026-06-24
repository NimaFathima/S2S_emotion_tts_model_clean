"""
benchmark_pipeline.py — Signet Aid latency baseline / regression guard

Times each heavy vision stage independently on synthetic frames so we can prove
that an architecture change improves (or at least does not degrade) performance.
Run it BEFORE and AFTER every change and compare.

    python benchmark_pipeline.py                # all stages, 100 iters
    python benchmark_pipeline.py --iters 200    # more iterations
    python benchmark_pipeline.py --json out.json

Each stage is guarded: if a model can't load (e.g. offline first run), that stage
is reported as SKIPPED and the others still run.

Architecture note:
  - CURRENT pipeline runs 3 heavy stages round-robin: face-detect (InsightFace),
    landmarks+blendshapes (MediaPipe), emotion (HSEmotion).
  - UNIFIED pipeline (planned) folds detection into the MediaPipe pass, removing
    the InsightFace stage. The 'detect' number below is exactly what that change
    would save — compare 'current_total' vs 'unified_total'.
"""

import argparse
import json
import time
import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)


def _bench_frame(h=480, w=640):
    """
    Benchmark frame. Uses a REAL face so MediaPipe does its full detection/mesh
    work every iteration (it short-circuits on no-face frames, which would make
    its timing look artificially fast). Falls back to a synthetic blob offline.
    """
    import os, cv2, urllib.request
    path = "_bench_face.jpg"
    try:
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                path)
        img = cv2.imread(path)
        if img is not None:
            return cv2.resize(img, (w, h))
    except Exception:
        pass
    print("  (no real face available — using synthetic frame; MediaPipe timing "
          "may be optimistic)")
    img = (np.random.rand(h, w, 3) * 40 + 100).astype(np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (90, 120), 0, 0, 360, (200, 180, 160), -1)
    return img


def _synth_crop(size=224):
    return (np.random.rand(size, size, 3) * 40 + 110).astype(np.uint8)


def _time_stage(name, fn, iters, warmup=5):
    """Run fn() warmup+iters times, return mean ms / FPS, or None on failure."""
    try:
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters
        ms = dt * 1000.0
        print(f"  {name:<22} {ms:8.2f} ms   ({1.0/dt:6.1f} stage-FPS)")
        return ms
    except Exception as e:
        print(f"  {name:<22} SKIPPED ({type(e).__name__}: {e})")
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json", metavar="FILE", help="write results as JSON")
    args = ap.parse_args(argv)

    frame = _bench_frame()
    crop = _synth_crop()
    results = {"iters": args.iters}

    print("=" * 58)
    print(f"Signet Aid — latency benchmark ({args.iters} iters/stage)")
    print("=" * 58)

    # Stage 1a: face detection (InsightFace) — the OLD backend
    detect_ms = None
    try:
        from processes.face_detector import FaceDetector
        det = FaceDetector()
        detect_ms = _time_stage("detect (InsightFace)", lambda: det.detect(frame), args.iters)
        det.close()
    except Exception as e:
        print(f"  detect (InsightFace)   SKIPPED ({type(e).__name__}: {e})")

    # Stage 1b: face detection (MediaPipe) — the NEW unified backend
    detect_mp_ms = None
    try:
        from processes.face_detector import MediaPipeFaceDetector
        detmp = MediaPipeFaceDetector()
        detect_mp_ms = _time_stage("detect (MediaPipe)", lambda: detmp.detect(frame), args.iters)
        detmp.close()
    except Exception as e:
        print(f"  detect (MediaPipe)     SKIPPED ({type(e).__name__}: {e})")

    # Stage 2: landmarks + blendshapes (MediaPipe) — kept, becomes the detector too
    nmm_ms = None
    try:
        from processes.nmm_classifier import NMMClassifier
        nmm = NMMClassifier()
        nmm_ms = _time_stage("landmarks (MediaPipe)", lambda: nmm.classify(frame), args.iters)
        nmm.close()
    except Exception as e:
        print(f"  landmarks (MediaPipe)  SKIPPED ({type(e).__name__}: {e})")

    # Stage 3: emotion (HSEmotion)
    emo_ms = None
    try:
        from processes.emotion_inference import HSEmotionInference
        emo = HSEmotionInference()
        emo_ms = _time_stage("emotion (HSEmotion)", lambda: emo.infer(crop), args.iters)
    except Exception as e:
        print(f"  emotion (HSEmotion)    SKIPPED ({type(e).__name__}: {e})")

    results.update(detect_ms=detect_ms, detect_mp_ms=detect_mp_ms,
                   nmm_ms=nmm_ms, emotion_ms=emo_ms)

    # Architecture comparison (only when all needed stages measured)
    print("-" * 58)
    if None not in (detect_ms, detect_mp_ms, nmm_ms, emo_ms):
        current = detect_ms + nmm_ms + emo_ms        # InsightFace detect
        unified = detect_mp_ms + nmm_ms + emo_ms     # MediaPipe detect
        saved = detect_ms - detect_mp_ms
        results.update(current_total_ms=current, unified_total_ms=unified, saved_ms=saved)
        print(f"  OLD  (InsightFace detect + landmarks + emotion): {current:7.2f} ms/cycle")
        print(f"  NEW  (MediaPipe   detect + landmarks + emotion): {unified:7.2f} ms/cycle")
        print(f"  => unifying on MediaPipe saves {saved:6.2f} ms "
              f"({saved/current*100:4.1f}% of the cycle)")
    else:
        print("  (architecture comparison needs all stages — some skipped)")
    print("=" * 58)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
