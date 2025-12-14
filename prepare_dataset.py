import os
import shutil
import random

RAW_PATH = "dataset/raw"
TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

# Aspek Ratio
train_ratio = 0.8

# Path Penyimpanan
for path in [TRAIN_PATH, TEST_PATH]:
    for sub in ["real", "fake"]:
        os.makedirs(os.path.join(path, sub), exist_ok=True)

# Ambil Daftar Gambar Asli dan Palsu
real_images = [f for f in os.listdir(os.path.join(RAW_PATH, "Au")) if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff'))]
fake_images = [f for f in os.listdir(os.path.join(RAW_PATH, "Tp")) if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff'))]

print(f"📊 Found {len(real_images)} real images, {len(fake_images)} fake images")

# Acak urutan file
random.shuffle(real_images)
random.shuffle(fake_images)

def split_and_copy(images, label):
    split_idx = int(len(images) * train_ratio)
    train_files = images[:split_idx]
    test_files = images[split_idx:]

    for img in train_files:
        src = os.path.join(RAW_PATH, "Au" if label == "real" else "Tp", img)
        dst = os.path.join(TRAIN_PATH, label, img)
        if os.path.isfile(src):
            shutil.copy(src, dst)

    for img in test_files:
        src = os.path.join(RAW_PATH, "Au" if label == "real" else "Tp", img)
        dst = os.path.join(TEST_PATH, label, img)
        if os.path.isfile(src):
            shutil.copy(src, dst)

split_and_copy(real_images, "real")
split_and_copy(fake_images, "fake")

# 🆕 TAMBAH DATA BALANCE CHECK
train_real = len(os.listdir(os.path.join(TRAIN_PATH, "real")))
train_fake = len(os.listdir(os.path.join(TRAIN_PATH, "fake")))
test_real = len(os.listdir(os.path.join(TEST_PATH, "real")))
test_fake = len(os.listdir(os.path.join(TEST_PATH, "fake")))

print(f"\n✅ Dataset berhasil dipisahkan!")
print(f"📁 Train: {train_real} real, {train_fake} fake")
print(f"📁 Test: {test_real} real, {test_fake} fake")
print(f"📊 Balance ratio: {train_real/(train_real+train_fake):.2f} real, {train_fake/(train_real+train_fake):.2f} fake")