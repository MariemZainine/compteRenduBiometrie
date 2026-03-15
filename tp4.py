import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class FaceRecognitionPCA:

    def __init__(self, n_components=30):
        """
        Initialise :
        - détecteur Viola-Jones
        - nombre de composantes principales
        - variables internes
        """

        self.n_components = n_components

        # détecteur Viola-Jones
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.mean = None
        self.eigenvectors = None
        self.projections = None
        self.labels = None

    # --------------------------------------------------

    def detect_face(self, image):

        """
        Détection et extraction du visage
        Retour : visage gris 100x100
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) == 0:
            return None

        # choisir le plus grand visage
        face = max(faces, key=lambda x: x[2] * x[3])

        x, y, w, h = face

        face_img = gray[y:y + h, x:x + w]

        face_img = cv2.resize(face_img, (100, 100))

        return face_img

    # --------------------------------------------------

    def load_dataset(self, dataset_path):

        """
        Charge les images et crée la matrice X
        """

        X = []
        y = []

        for person in os.listdir(dataset_path):

            person_path = os.path.join(dataset_path, person)

            if not os.path.isdir(person_path):
                continue

            for image_name in os.listdir(person_path):

                img_path = os.path.join(person_path, image_name)

                img = cv2.imread(img_path)

                face = self.detect_face(img)

                if face is None:
                    continue

                vector = face.flatten()

                X.append(vector)
                y.append(person)

        X = np.array(X)
        y = np.array(y)

        return X, y

    # --------------------------------------------------

    def compute_pca(self, X):

        """
        Calcul PCA
        """

        # moyenne
        self.mean = np.mean(X, axis=0)

        # centrage
        X_centered = X - self.mean

        # covariance
        cov = np.cov(X_centered, rowvar=False)

        # valeurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # tri décroissant
        idx = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[:, idx]

        # sélection des k composantes
        self.eigenvectors = eigenvectors[:, :self.n_components]

        return X_centered

    # --------------------------------------------------

    def project(self, face_vector):

        """
        Projection dans l'espace PCA
        """

        centered = face_vector - self.mean

        projection = np.dot(centered, self.eigenvectors)

        return projection

    # --------------------------------------------------

    def train(self, X, y):

        """
        Entraînement du modèle
        """

        self.labels = y

        X_centered = self.compute_pca(X)

        self.projections = []

        for face in X:

            proj = self.project(face)

            self.projections.append(proj)

        self.projections = np.array(self.projections)

    # --------------------------------------------------

    def recognize(self, image_path, threshold=4000):

        img = cv2.imread(image_path)

        face = self.detect_face(img)

        if face is None:
            return None, None, "No Face Detected"

        vector = face.flatten()

        proj_test = self.project(vector)

        distances = []

        for proj in self.projections:

            d = np.linalg.norm(proj_test - proj)

            distances.append(d)

        distances = np.array(distances)

        min_index = np.argmin(distances)

        min_distance = distances[min_index]

        identity = self.labels[min_index]

        decision = "Match" if min_distance < threshold else "No Match"

        return identity, min_distance, decision

    def show_eigenfaces(self):

        fig, axes = plt.subplots(2,5, figsize=(10,4))

        for i, ax in enumerate(axes.flat):

            eigenface = self.eigenvectors[:, i].reshape(100,100)

            ax.imshow(eigenface, cmap='gray')
            ax.set_title(f"Eigenface {i+1}")
            ax.axis("off")

        plt.show()

if __name__ == "__main__":

    dataset = "TP4/dataset"
    test_image = "TP4/test.jpg"

    for k in [10, 20, 50]:

        print("\n==============================")
        print("Test avec k =", k)
        print("==============================")

        model = FaceRecognitionPCA(n_components=k)

        print("Chargement du dataset...")
        X, y = model.load_dataset(dataset)

        print("Training PCA...")
        model.train(X, y)

        print("Reconnaissance...")

        identity, distance, decision = model.recognize(test_image, threshold=4000)

        print("Distance minimale :", distance)
        print("Identité prédite :", identity)
        print("Décision :", decision)

        # affichage image
        img = cv2.imread(test_image)

        face = model.detect_face(img)

        if face is not None:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = model.face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(img,
                    f"k={k} | {decision} | Dist={distance:.2f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Result", img)

        # afficher les eigenfaces
        model.show_eigenfaces()

        cv2.waitKey(0)

    cv2.destroyAllWindows()