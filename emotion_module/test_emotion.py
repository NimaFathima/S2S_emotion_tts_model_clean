import cv2
import numpy as np
import urllib.request
from processes.emotion_inference import HSEmotionInference
import os

urls = {
    "happy.jpg": "https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/examples/images/happy.jpg",
    "angry.jpg": "https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/examples/images/angry.jpg"
}

def main():
    for name, url in urls.items():
        if not os.path.exists(name):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(name, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                pass

    engine = HSEmotionInference()

    for name in ["happy.jpg", "angry.jpg"]:
        img = cv2.imread(name)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            res = engine.infer(img)
            print(f"{name:15s} -> V:{res.valence:+.3f}, A:{res.arousal:+.3f}")

if __name__ == "__main__":
    main()
