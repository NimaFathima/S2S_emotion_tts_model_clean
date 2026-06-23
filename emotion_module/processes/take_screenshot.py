import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Take a screenshot of the integration test output.
Usage: python -m processes.take_screenshot
"""
import cv2, numpy as np

def take_screenshot():
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

    # Try webcam
    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()
    if not use_webcam:
        print("No webcam found — using blank frame.")
    else:
        print("Webcam opened.")

    # Capture a single frame
    if use_webcam:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Using blank frame.")
            frame = np.full((480, 640, 3), 100, dtype=np.uint8)
            use_webcam = False
    else:
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    # Process the frame
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

    # Save the screenshot
    screenshot_path = "screenshot.png"
    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved to {screenshot_path}")

    # Cleanup
    if use_webcam:
        cap.release()
    nmm.close()

if __name__ == "__main__":
    take_screenshot()