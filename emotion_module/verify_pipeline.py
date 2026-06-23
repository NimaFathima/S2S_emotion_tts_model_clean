import cv2
import numpy as np
import urllib.request
import os

from processes.face_detector import FaceDetector
from processes.emotion_inference import HSEmotionInference
from processes.nmm_classifier import NMMClassifier


def download_image(url, filename):
    if not os.path.exists(filename):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Failed to download {filename}: {e}")


def main():
    # Download standard test faces
    images = {
        "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "obama.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
        "biden.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg"
    }

    for name, url in images.items():
        download_image(url, name)

    print("Loading components for verification...")
    detector = FaceDetector()
    emotion = HSEmotionInference()
    nmm = NMMClassifier()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print("--------------------------------------------------")

    for name in images.keys():
        if not os.path.exists(name):
            continue

        img = cv2.imread(name)
        if img is None:
            continue

        # Apply CLAHE as in integration test
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        clahe_frame = cv2.merge([l_eq, a, b])
        clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_LAB2BGR)

        # 1. Detect Face
        det = detector.detect(clahe_frame)
        if det.face_crop is not None:
            # 2. Emotion Inference
            emo = emotion.infer(det.face_crop)
            # 3. NMM Classification
            nmm_ctx = nmm.classify(clahe_frame)

            print(f"Image: {name}")
            print(f"  -> Face Found! (Confidence: {det.confidence:.2f})")
            print(f"  -> Valence: {emo.valence:+.3f}")
            print(f"  -> Arousal: {emo.arousal:+.3f}")
            print(f"  -> NMM Flags: Y/N={nmm_ctx.is_yn_question}, WH={nmm_ctx.is_wh_question}, Neg={nmm_ctx.is_negation}")
        else:
            print(f"Image: {name} -> NO FACE DETECTED")
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()
