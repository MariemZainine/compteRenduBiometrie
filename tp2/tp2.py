import os 
import matplotlib.pyplot as plt
import numpy as np 
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
#pretraitement

def preprocess(image_path: str) -> np.ndarray:

    img = Image.open(image_path).convert("L")
    img = img.resize((300, 300))
    img = ImageOps.equalize(img)
    img = img.point(lambda x: 255 if x > 128 else 0)
    img = img.filter(ImageFilter.FIND_EDGES)

    return np.array(img)
#calcul de similarité
def compute_ssim(image1_path: str, image2_path: str) -> float:
    
    image1 = preprocess(image1_path)
    image2 = preprocess(image2_path)
    similarity = compare_ssim(image1, image2, data_range=255)

    return similarity
#decision automatique selon le seuil defini
def decision(similarity: float, threshold: float = 0.75) -> str:
    if similarity >= threshold:
        return "acceptee"
    else:
        return "rejetee"

if __name__ == "__main__":
# resultat Score SSIM : 0.2163
    img1 = "empreinte1.png"
    img2 = "empreinte2.png"

#Score SSIM : 1.000

    """Même doigt 100% la similarité
    img1 = "empreinte1.png"
       img2 = "empreinte1.png"
    """
    score = compute_ssim(img1, img2)
    result = decision(score)

    print("Score SSIM :", round(score, 4))
    print("Décision :", result)