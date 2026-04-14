from PIL import Image
import numpy as np
import random


# =========================================================
# Conversion Texte <-> Binaire
# =========================================================

def text_to_bin(text):
    return ''.join(format(ord(c), '08b') for c in text)


def bin_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)


# =========================================================
# PARTIE 1 : LSB Niveau de Gris
# =========================================================

def embed_lsb_gray(image_path, message, output_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    binary_msg = text_to_bin(message)
    flat = img_array.flatten()

    if len(binary_msg) > len(flat):
        raise ValueError("Message trop grand pour l'image.")

    for i, bit in enumerate(binary_msg):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    watermarked = flat.reshape(img_array.shape)
    Image.fromarray(watermarked.astype(np.uint8)).save(output_path)


def extract_lsb_gray(image_path, msg_len):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).flatten()

    total_bits = msg_len * 8
    bits = ''.join(str(img_array[i] & 1) for i in range(total_bits))

    return bin_to_text(bits)


# =========================================================
# PARTIE 2 : LSB RGB
# =========================================================

def embed_lsb_rgb(image_path, message, output_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    binary_msg = text_to_bin(message)
    flat = img_array.flatten()

    if len(binary_msg) > len(flat):
        raise ValueError("Message trop grand pour l'image RGB.")

    for i, bit in enumerate(binary_msg):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    watermarked = flat.reshape(img_array.shape)
    Image.fromarray(watermarked.astype(np.uint8)).save(output_path)


def extract_lsb_rgb(image_path, msg_len):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).flatten()

    total_bits = msg_len * 8
    bits = ''.join(str(img_array[i] & 1) for i in range(total_bits))

    return bin_to_text(bits)


# =========================================================
# PARTIE 3 : LSB avec Clé Secrète
# =========================================================

def embed_lsb_key(image_path, message, output_path, key):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    binary_msg = text_to_bin(message)
    flat = img_array.flatten()

    if len(binary_msg) > len(flat):
        raise ValueError("Message trop grand pour l'image.")

    random.seed(key)
    positions = random.sample(range(len(flat)), len(binary_msg))

    for pos, bit in zip(positions, binary_msg):
        flat[pos] = (flat[pos] & 0xFE) | int(bit)

    watermarked = flat.reshape(img_array.shape)
    Image.fromarray(watermarked.astype(np.uint8)).save(output_path)


def extract_lsb_key(image_path, msg_len, key):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).flatten()

    total_bits = msg_len * 8

    random.seed(key)
    positions = random.sample(range(len(img_array)), total_bits)

    bits = ''.join(str(img_array[pos] & 1) for pos in positions)

    return bin_to_text(bits)


# =========================================================
# Programme Principal
# =========================================================

def main():
    message = "bonjour"

    print("=== PARTIE 1 : LSB Grayscale ===")
    embed_lsb_gray("greythird.jpg", message, "gray_output.png")
    print(extract_lsb_gray("gray_output.png", len(message)))

    print("\n=== PARTIE 2 : LSB RGB ===")
    embed_lsb_rgb("img_rgb.jpg", message, "rgb_output.png")
    print(extract_lsb_rgb("rgb_output.png", len(message)))

    print("\n=== PARTIE 3 : LSB avec Clé ===")
    embed_lsb_key("greythird.jpg", message, "key_output.png", key=42)
    print(extract_lsb_key("key_output.png", len(message), key=42))

if __name__ == "__main__":
    main()