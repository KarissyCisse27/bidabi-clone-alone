import os
import shutil
import random

RAW_DIR = "data/raw/images"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20
SEED        = 42

random.seed(SEED)

for category in os.listdir(RAW_DIR):
    cat_path = os.path.join(RAW_DIR, category)
    if not os.path.isdir(cat_path):
        continue

    images = [f for f in os.listdir(cat_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        TRAIN_DIR: images[:n_train],
        VAL_DIR:   images[n_train:n_train + n_val],
        TEST_DIR:  images[n_train + n_val:],
    }

    for split_dir, split_images in splits.items():
        dest = os.path.join(split_dir, category)
        os.makedirs(dest, exist_ok=True)
        for img in split_images:
            shutil.copy(
                os.path.join(cat_path, img),
                os.path.join(dest, img)
            )

    print(f"{category}: Train={len(splits[TRAIN_DIR])}, "
          f"Val={len(splits[VAL_DIR])}, "
          f"Test={len(splits[TEST_DIR])}")

print("✅ Split terminé !")