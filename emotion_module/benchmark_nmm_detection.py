"""
benchmark_nmm_detection.py
Compare full-frame vs. crop-based MediaPipe FaceLandmarker for NMM classification.

Usage:
    cd emotion_module
    python benchmark_nmm_detection.py

This script compares two approaches without modifying production code:
  1. CURRENT: Run MediaPipe on the full CLAHE frame (includes internal face detection)
  2. PROPOSED: Crop to RetinaFace bbox, run MediaPipe on the smaller crop

Metrics measured per image:
  - Wall-clock time for NMM classification
  - NMM flag differences (Y/N question, WH question, negation)
  - Face detection success rate

Conclusion will be printed at the end.
"""

import time
import cv2
import numpy as np
from pathlib import Path

# Ensure imports work when run from emotion_module/ directory
from processes.face_detector import FaceDetector
from processes.nmm_classifier import NMMClassifier
import urllib.request
import os


TEST_IMAGES = {
    "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "obama.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
    "biden.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
}

# How many iterations per image for stable timing
WARMUP_ITERS = 5
BENCH_ITERS = 20


def download_image(url: str, filename: str):
    if os.path.exists(filename):
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(filename, "wb") as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        print(f"  [WARN] Could not download {filename}: {e}")
        return False


def benchmark_full_frame(clahe_frame: np.ndarray, nmm: NMMClassifier):
    """Run NMM classification on the full frame (current approach)."""
    t0 = time.perf_counter()
    ctx = nmm.classify(clahe_frame)
    elapsed = time.perf_counter() - t0
    return ctx, elapsed


def benchmark_crop(clahe_frame: np.ndarray, bbox, nmm: NMMClassifier):
    """
    Run NMM classification on a tight crop around the detected face.
    Simulates what would happen if we reused the RetinaFace crop.
    """
    x1, y1, x2, y2 = bbox
    h, w = clahe_frame.shape[:2]

    # Add 20% padding (same as FaceDetector) so MediaPipe has margin
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * 0.20)
    pad_y = int(bh * 0.20)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)

    crop = clahe_frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None, 0.0

    t0 = time.perf_counter()
    ctx = nmm.classify(crop)
    elapsed = time.perf_counter() - t0
    return ctx, elapsed


def fmt_ctx(ctx):
    flags = []
    if ctx.is_yn_question:
        flags.append("Y/N")
    if ctx.is_wh_question:
        flags.append("WH")
    if ctx.is_negation:
        flags.append("NEG")
    return "|".join(flags) if flags else "—"


def main():
    print("=" * 60)
    print("Benchmark: Full-frame vs. Crop-based NMM Detection")
    print("=" * 60)

    # Download test images
    available = []
    for name, url in TEST_IMAGES.items():
        if download_image(url, name):
            available.append(name)

    if not available:
        print("No test images available — aborting benchmark.")
        return

    detector = FaceDetector()
    nmm = NMMClassifier()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    results = []

    for img_name in available:
        img = cv2.imread(img_name)
        if img is None:
            continue

        # Apply same CLAHE preprocessing as production pipeline
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        clahe_frame = cv2.merge([l_eq, a, b])
        clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_LAB2BGR)

        # Run RetinaFace once to get the bbox
        det = detector.detect(clahe_frame)
        if det.bbox is None:
            print(f"\n{img_name}: No face detected by RetinaFace — skipping.")
            continue

        print(f"\n--- {img_name} (bbox={det.bbox}, conf={det.confidence:.2f}) ---")

        # Warmup
        for _ in range(WARMUP_ITERS):
            benchmark_full_frame(clahe_frame, nmm)
            if det.bbox:
                benchmark_crop(clahe_frame, det.bbox, nmm)

        # Benchmark full-frame
        full_times = []
        full_ctx = None
        for _ in range(BENCH_ITERS):
            ctx, t = benchmark_full_frame(clahe_frame, nmm)
            full_times.append(t)
            full_ctx = ctx

        full_ms = sum(full_times) / len(full_times) * 1000.0

        # Benchmark crop
        crop_times = []
        crop_ctx = None
        for _ in range(BENCH_ITERS):
            ctx, t = benchmark_crop(clahe_frame, det.bbox, nmm)
            crop_times.append(t)
            crop_ctx = ctx

        crop_ms = sum(crop_times) / len(crop_times) * 1000.0

        print(f"  Full-frame: {full_ms:.2f} ms/frame  flags={fmt_ctx(full_ctx)}")
        print(f"  Crop:       {crop_ms:.2f} ms/frame  flags={fmt_ctx(crop_ctx)}")

        speedup = full_ms / crop_ms if crop_ms > 0 else float("inf")
        print(f"  Speedup:    {speedup:.2f}x")

        # Check for flag mismatches
        mismatch = (
            full_ctx.is_yn_question != crop_ctx.is_yn_question or
            full_ctx.is_wh_question != crop_ctx.is_wh_question or
            full_ctx.is_negation != crop_ctx.is_negation
        )
        if mismatch:
            print(f"  [ALERT] NMM flags differ between full-frame and crop!")

        results.append({
            "image": img_name,
            "full_ms": full_ms,
            "crop_ms": crop_ms,
            "speedup": speedup,
            "mismatch": mismatch,
            "full_ctx": full_ctx,
            "crop_ctx": crop_ctx,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not results:
        print("No successful benchmarks.")
        nmm.close()
        return

    avg_full = sum(r["full_ms"] for r in results) / len(results)
    avg_crop = sum(r["crop_ms"] for r in results) / len(results)
    avg_speedup = avg_full / avg_crop if avg_crop > 0 else 0.0
    mismatches = sum(1 for r in results if r["mismatch"])

    print(f"Average full-frame: {avg_full:.2f} ms")
    print(f"Average crop:       {avg_crop:.2f} ms")
    print(f"Average speedup:    {avg_speedup:.2f}x")
    print(f"Flag mismatches:    {mismatches}/{len(results)}")

    if mismatches > 0:
        print("\n⚠️  Crop-based detection produced DIFFERENT NMM flags on some images.")
        print("   This suggests reduced accuracy when MediaPipe sees less context.")
        print("   Recommendation: KEEP full-frame approach for NMM.")
    else:
        if avg_speedup > 1.3:
            print("\n✅ Crop is significantly faster with matching accuracy.")
            print("   Recommendation: Consider switching to crop-based NMM.")
        else:
            print("\nℹ️  Minor speedup with matching accuracy.")
            print("   Recommendation: KEEP full-frame (not worth the complexity).")

    nmm.close()


if __name__ == "__main__":
    main()
