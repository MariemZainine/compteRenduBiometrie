import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet


class FaceRecognitionDL:

    def __init__(self):

        self.detector = MTCNN()
        self.embedder = FaceNet()

        self.database = []
        self.labels = []

    def detect_face(self, image):

        results = self.detector.detect_faces(image)

        if len(results) == 0:
            face = cv2.resize(image, (160,160))
            return face

        x, y, w, h = results[0]['box']

        x = max(0, x)
        y = max(0, y)

        face = image[y:y+h, x:x+w]

        face = cv2.resize(face, (160,160))

        return face


    def extract_embedding(self, face):

        face = face.astype("float32")

        face = np.expand_dims(face, axis=0)

        embedding = self.embedder.embeddings(face)

        return embedding[0]

  
    def build_database(self, dataset_path):

        print("Construction de la base...")

        for person in os.listdir(dataset_path):

            person_path = os.path.join(dataset_path, person)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path)[:5]:

                img_path = os.path.join(person_path, img_name)

                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue

             
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                face = self.detect_face(image)

                if face is None:
                    continue

                embedding = self.extract_embedding(face)

                self.database.append(embedding)
                self.labels.append(person)

        print("Base construite :", len(self.database), "embeddings")


    def cosine_similarity(self, emb1, emb2):

        dot = np.dot(emb1, emb2)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return dot / (norm1 * norm2)

    def euclidean_distance(self, emb1, emb2):

        return np.linalg.norm(emb1 - emb2)

    def recognize(self, image_path, threshold=0.8):

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return None, None, "Image introuvable"

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        face = self.detect_face(image)

        emb = self.extract_embedding(face)

        best_distance = float("inf")
        best_label = None

        for db_emb, label in zip(self.database, self.labels):

            dist = self.euclidean_distance(emb, db_emb)

            if dist < best_distance:
                best_distance = dist
                best_label = label

        decision = "Match" if best_distance <= threshold else "No Match"

        return best_label, best_distance, decision

def main():

    dataset = "dataset"
    test_image = "test.jpg"
    model = FaceRecognitionDL()
    model.build_database(dataset)
    print("\nReconnaissance en cours...\n")
    label, distance, decision = model.recognize(test_image)
    print("Résultat :")
    print("Identité :", label)
    print("Distance :", distance)
    print("Décision :", decision)

if __name__ == "__main__":
    main()