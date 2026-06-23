import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Integration test for all four of Nima's components.
Run this before handing off to JJ for final assembly.
Usage: python -m processes.test_nima_integration
"""
import cv2, numpy as np

def test_with_webcam_frame():
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

    # Try live webcam — exit with 'q'
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found — running with blank frames instead.")
        for i in range(10):
            frame = np.full((480, 640, 3), 100, dtype=np.uint8)
            det = detector.detect(frame)
            coast = coasting.update(det.confidence, 0.0, 0.0)
            print(f"Frame {i}: conf={det.confidence:.2f} V={coast.valence:.3f} lost={coast.tracking_lost}")
        return

    print("Webcam opened. Press 'q' to quit.\n")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Apply CLAHE to frame as expected by components
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        clahe_frame = cv2.merge([l_eq, a, b])
        clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_LAB2BGR)

        det = detector.detect(clahe_frame)
        if det.face_crop is not None:
            emo   = emotion.infer(det.face_crop)
            coast = coasting.update(det.confidence, emo.valence, emo.arousal)
        else:
            coast = coasting.update(0.0, 0.0, 0.0)

        nmm_ctx = nmm.classify(clahe_frame)
        eff_v, eff_a = nmm.apply_dampening(coast.valence, coast.arousal, nmm_ctx)

        # Display
        label = (f"V:{eff_v:+.2f} A:{eff_a:+.2f} | "
                 f"conf:{det.confidence:.2f} lost:{coast.tracking_lost} | "
                 f"yn:{nmm_ctx.is_yn_question} wh:{nmm_ctx.is_wh_question} neg:{nmm_ctx.is_negation}")
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Signet Aid — Nima Integration Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    nmm.close()


if __name__ == "__main__":
    test_with_webcam_frame()