import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# 📁 Gestion des chemins (IMPORTANT)
# ==============================
base_path = os.path.dirname(__file__)

gray_path = os.path.join(base_path, "lina.jpg")
rgb_path = os.path.join(base_path, "rgb.jpg")

# ==============================
# 🔑 Génération des paires
# ==============================
def generate_pairs(shape, N, key=42):
    np.random.seed(key)
    h, w = shape
    pairs = []

    for _ in range(N):
        x1, y1 = np.random.randint(0, h), np.random.randint(0, w)
        x2, y2 = np.random.randint(0, h), np.random.randint(0, w)
        pairs.append(((x1, y1), (x2, y2)))

    return pairs

# ==============================
# 🧩 Insertion Patchwork
# ==============================
def patchwork_embed(img, pairs, delta=5):
    watermarked = img.copy().astype(np.int32)

    for (x1,y1),(x2,y2) in pairs:
        watermarked[x1,y1] += delta
        watermarked[x2,y2] -= delta

    return np.clip(watermarked, 0, 255).astype(np.uint8)

# ==============================
# 🔍 Détection
# ==============================
def patchwork_detect(img, pairs):
    diff = []

    for (x1,y1),(x2,y2) in pairs:
        diff.append(int(img[x1,y1]) - int(img[x2,y2]))

    return np.mean(diff)

# ==============================
# 🔴 Attaques
# ==============================
def add_noise(img):
    noise = np.random.normal(0, 5, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def compress_jpeg(img, filename):
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def apply_blur(img):
    return cv2.GaussianBlur(img, (5,5), 0)

# ==============================
# 🟣 PARTIE 1 — GRAYSCALE
# ==============================
print("===== PARTIE 1 : GRAYSCALE =====")

img_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    print("❌ Image lina.jpg non trouvée")
    print("Chemin utilisé :", gray_path)
    exit()

pairs = generate_pairs(img_gray.shape, 5000)

watermarked_gray = patchwork_embed(img_gray, pairs)

score_gray = patchwork_detect(watermarked_gray, pairs)

print("Score :", score_gray)
print("Décision :", "Tatouage détecté" if score_gray > 1 else "Pas de tatouage")

# ==============================
# 🔴 Attaques
# ==============================
noisy = add_noise(watermarked_gray)

compressed_path = os.path.join(base_path, "compressed.jpg")
compressed = compress_jpeg(watermarked_gray, compressed_path)

blurred = apply_blur(watermarked_gray)

print("\n--- Robustesse ---")
print("Original :", patchwork_detect(watermarked_gray, pairs))
print("Bruit :", patchwork_detect(noisy, pairs))
print("JPEG :", patchwork_detect(compressed, pairs))
print("Flou :", patchwork_detect(blurred, pairs))

# ==============================
# 🟠 PARTIE 2 — RGB
# ==============================
print("\n===== PARTIE 2 : RGB =====")

img_color = cv2.imread(rgb_path)

if img_color is None:
    print("❌ Image rgb.jpg non trouvée")
    print("Chemin utilisé :", rgb_path)
    exit()

B, G, R = cv2.split(img_color)

pairs_rgb = generate_pairs(R.shape, 5000)

R_w = patchwork_embed(R, pairs_rgb)

watermarked_rgb = cv2.merge([B, G, R_w])

score_rgb = patchwork_detect(R_w, pairs_rgb)

print("Score RGB :", score_rgb)
print("Décision RGB :", "Tatouage détecté" if score_rgb > 1 else "Pas de tatouage")

# ==============================
# 📊 AFFICHAGE
# ==============================
plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.title("Original Grayscale")
plt.imshow(img_gray, cmap='gray')

plt.subplot(2,2,2)
plt.title("Tatouée Grayscale")
plt.imshow(watermarked_gray, cmap='gray')

plt.subplot(2,2,3)
plt.title("Original RGB")
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

plt.subplot(2,2,4)
plt.title("Tatouée RGB (canal R)")
plt.imshow(cv2.cvtColor(watermarked_rgb, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()