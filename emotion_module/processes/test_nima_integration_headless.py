import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Integration test for all four of Nima's components in headless mode.
Saves screenshots to a directory for review.
Usage: python -m processes.test_nima_integration_headless
"""
import cv2, numpy as np

def test_headless():
    from processes.face_detector    import FaceDetector
    from processes.coasting_matrix  import CoastingMatrix
    from processes.emotion_inference import HSEmotionInference
    from processes.nmm_classifier   import NMMClassifier, NMMContext

    print("Loading all components...")
    detector  = FaceDetector()
    coasting  = CoastingMatrix()
    emotion   = HSEmotionInference()
    nmm       = NMMClassifier()
    print("All components loaded successfully.\n")

    # Try webcam, if not available use blank frames
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    if not use_webcam:
        print("No webcam found — running with blank frames.")
    else:
        print("Webcam opened.")

    # Create a directory to save screenshots
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)

    frame_count = 0
    max_frames = 30  # Capture 30 frames

    while frame_count < max_frames:
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam. Exiting.")
                break
        else:
            # Generate a blank frame (gray)
            frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        det = detector.detect(frame)
        if det.face_crop is not None:
            emo   = emotion.infer(det.face_crop)
            coast = coasting.update(det.confidence, emo.valence, emo.arousal)
        else:
            coast = coasting.update(0.0, 0.0, 0.0)

        nmm_ctx = nmm.classify(frame)
        eff_v, eff_a = nmm.apply_dampening(coast.valence, coast.arousal, nmm_ctx)

        # Draw the label on the frame
        label = (f"V:{eff_v:+.2f} A:{eff_a:+.2f} | "
                 f"conf:{det.confidence:.2f} lost:{coast.tracking_lost} | "
                 f"yn:{nmm_ctx.is_yn_question} wh:{nmm_ctx.is_wh_question} neg:{nmm_ctx.is_negation}")
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Print the label to console for verification
        print(f"Frame {frame_count:02d}: {label}")

        # Save the frame
        filename = os.path.join(screenshot_dir, f"frame_{frame_count:02d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        frame_count += 1

        # In headless mode, we don't need to wait for a key press, but we can break early if needed.
        # For simplicity, we just run for max_frames.

    if use_webcam:
        cap.release()
    # No windows to destroy since we didn't create any
    nmm.close()
    print(f"Done. {frame_count} screenshots saved to {screenshot_dir}")

if __name__ == "__main__":
    test_headless()
