#!/usr/bin/env python3
"""
One-shot script: extract PolypDataset.zip and organize into the directory
structure expected by ProtoSAM's get_polyp_dataset().

Run from ProtoSAM-main/:
    python organize_polyp_data.py
"""
import os
import shutil
import random
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(BASE, "data", "PolypDataset")
ZIP_PATH = "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/PolypDataset.zip"

TRAIN_IMG_DIR = os.path.join(DST, "TrainDataset", "images")
TRAIN_MSK_DIR = os.path.join(DST, "TrainDataset", "masks")
TEST_IMG_DIR  = os.path.join(DST, "TestDataset", "test", "images")
TEST_MSK_DIR  = os.path.join(DST, "TestDataset", "test", "masks")

for d in [TRAIN_IMG_DIR, TRAIN_MSK_DIR, TEST_IMG_DIR, TEST_MSK_DIR]:
    os.makedirs(d, exist_ok=True)

print("=== Extracting from zip ===")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    # --- Training data (mixed Kvasir + ClinicDB, numeric filenames) ---
    print("Extracting NewTRimage (train images)...")
    for name in zf.namelist():
        if name.startswith("NewTRimage/") and name.endswith(".png"):
            zf.extract(name, os.path.join(DST, "_tmp"))
    print("Extracting NewTRmask (train masks)...")
    for name in zf.namelist():
        if name.startswith("NewTRmask/") and name.endswith(".png"):
            zf.extract(name, os.path.join(DST, "_tmp"))

    # --- Test data (organized by dataset) ---
    test_ds = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB"]
    for ds in test_ds:
        prefix = f"TestDataset/TestDataset/{ds}/"
        print(f"Extracting {ds} test set...")
        for name in zf.namelist():
            if name.startswith(prefix + "images/") and name.endswith(".png"):
                zf.extract(name, os.path.join(DST, "_tmp"))
            if name.startswith(prefix + "masks/") and name.endswith(".png"):
                zf.extract(name, os.path.join(DST, "_tmp"))

tmp = os.path.join(DST, "_tmp")

# --- Move training data (flat, into TrainDataset) ---
print("\n=== Organizing training data ===")
src_train_img = os.path.join(tmp, "NewTRimage")
src_train_msk = os.path.join(tmp, "NewTRmask")
if os.path.isdir(src_train_img):
    for f in os.listdir(src_train_img):
        shutil.move(os.path.join(src_train_img, f), os.path.join(TRAIN_IMG_DIR, f))
if os.path.isdir(src_train_msk):
    for f in os.listdir(src_train_msk):
        shutil.move(os.path.join(src_train_msk, f), os.path.join(TRAIN_MSK_DIR, f))

n_train_imgs = len(os.listdir(TRAIN_IMG_DIR)) if os.path.isdir(TRAIN_IMG_DIR) else 0
n_train_msks = len(os.listdir(TRAIN_MSK_DIR)) if os.path.isdir(TRAIN_MSK_DIR) else 0
print(f"  Train images: {n_train_imgs}, masks: {n_train_msks}")

# --- Move test data (keep subdir structure for dataset identification) ---
print("\n=== Organizing test data ===")
src_test = os.path.join(tmp, "TestDataset", "TestDataset")
test_ds = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB"]
for ds in test_ds:
    src_img_ds = os.path.join(src_test, ds, "images")
    src_msk_ds = os.path.join(src_test, ds, "masks")
    dst_img_ds = os.path.join(TEST_IMG_DIR, ds)
    dst_msk_ds = os.path.join(TEST_MSK_DIR, ds)
    os.makedirs(dst_img_ds, exist_ok=True)
    os.makedirs(dst_msk_ds, exist_ok=True)

    n_img = n_msk = 0
    if os.path.isdir(src_img_ds):
        for f in os.listdir(src_img_ds):
            shutil.move(os.path.join(src_img_ds, f), os.path.join(dst_img_ds, f))
            n_img += 1
    if os.path.isdir(src_msk_ds):
        for f in os.listdir(src_msk_ds):
            shutil.move(os.path.join(src_msk_ds, f), os.path.join(dst_msk_ds, f))
            n_msk += 1
    print(f"  {ds}: {n_img} images, {n_msk} masks")

# --- Also merge user's existing raw data into test set ---
print("\n=== Merging existing raw test data ===")
RAW_ROOTS = {
    "CVC-ColonDB":      "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/colondb/raw",
    "ETIS-LaribPolypDB": "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/etis/raw",
}
for ds, raw_root in RAW_ROOTS.items():
    for sub in ["images", "masks"]:
        src = os.path.join(raw_root, sub)
        if ds == "CVC-ColonDB":
            dst = os.path.join(TEST_IMG_DIR if sub == "images" else TEST_MSK_DIR, ds)
        else:
            dst = os.path.join(TEST_IMG_DIR if sub == "images" else TEST_MSK_DIR, ds)
        os.makedirs(dst, exist_ok=True)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.endswith(".png"):
                    src_path = os.path.join(src, f)
                    dst_path = os.path.join(dst, f)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
    n_img = len(os.listdir(os.path.join(TEST_IMG_DIR, ds))) if os.path.isdir(os.path.join(TEST_IMG_DIR, ds)) else 0
    n_msk = len(os.listdir(os.path.join(TEST_MSK_DIR, ds))) if os.path.isdir(os.path.join(TEST_MSK_DIR, ds)) else 0
    print(f"  {ds}: {n_img} images, {n_msk} masks")

# --- Cleanup ---
shutil.rmtree(tmp, ignore_errors=True)

# --- Final report ---
print("\n=== Final dataset summary ===")
print(f"TrainDataset: {n_train_imgs} images, {n_train_msks} masks (support pool)")
for ds in test_ds:
    d_img = os.path.join(TEST_IMG_DIR, ds)
    d_msk = os.path.join(TEST_MSK_DIR, ds)
    ni = len(os.listdir(d_img)) if os.path.isdir(d_img) else 0
    nm = len(os.listdir(d_msk)) if os.path.isdir(d_msk) else 0
    print(f"  {ds}: {ni} images, {nm} masks")

print("\nDone. Ready to run:  ./run_protosam.sh polyp")
