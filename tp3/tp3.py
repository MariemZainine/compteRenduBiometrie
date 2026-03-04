import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class FaceVerificationSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.reference_features = None

    def _load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image : {path}")
        return img

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        face = cv2.resize(gray[y:y+h, x:x+w], (128, 128))
        return face, (x, y, w, h)

    def extract_lbp_features(self, face_image):
        lbp = np.zeros_like(face_image)
        for i in range(1, 127):
            for j in range(1, 127):
                c = face_image[i, j]
                neigh = [
                    face_image[i-1, j-1], face_image[i-1, j],
                    face_image[i-1, j+1], face_image[i, j+1],
                    face_image[i+1, j+1], face_image[i+1, j],
                    face_image[i+1, j-1], face_image[i, j-1]
                ]
                bs = ''.join('1' if n >= c else '0' for n in neigh)
                lbp[i, j] = int(bs, 2)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
        hist = hist.astype(float); hist /= (hist.sum() + 1e-6)
        return hist

    def setup_reference(self, image_path):
        image = self._load_image(image_path)
        face, _ = self.detect_face(image)
        if face is None:
            raise ValueError("Aucun visage détecté dans l'image de référence")
        self.reference_features = self.extract_lbp_features(face)

    def verify_face(self, image_path, threshold=0.75):
        image = self._load_image(image_path)
        face, coords = self.detect_face(image)
        if face is None:
            raise ValueError("Aucun visage détecté dans l'image test")
        test_features = self.extract_lbp_features(face)
        dist = euclidean(self.reference_features, test_features)
        sim = 1 - dist
        decision = "Match" if sim >= threshold else "No Match"
        return sim, decision, image, coords

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, "faces")          # sous‑répertoire
    ref_img  = sys.argv[1] if len(sys.argv) > 1 else \
               os.path.join(images_dir, "f1.png")
    test_img = sys.argv[2] if len(sys.argv) > 2 else \
               os.path.join(images_dir, "f4.png")

    system = FaceVerificationSystem()
    system.setup_reference(ref_img)
    similarity, decision, image, coords = system.verify_face(test_img)
    print("Similarité :", round(similarity, 4))
    print("Decision   :", decision)