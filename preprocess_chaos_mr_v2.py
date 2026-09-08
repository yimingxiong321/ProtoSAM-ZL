#!/usr/bin/env python3
"""
Preprocess CHAOS MR T2SPIR data for ProtoSAM evaluation.
Structure: T2SPIR/DICOM_anon/*.dcm + T2SPIR/Ground/*.png

Usage:  python preprocess_chaos_mr_v2.py
Run from ProtoSAM-main/ directory.
"""
import os, sys, glob, json
import numpy as np
import SimpleITK as sitk
from PIL import Image

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "CHAOST2")
CHAOS_RAW = "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/CHAOS/raw"
TRAIN_MR = os.path.join(CHAOS_RAW, "Train_Sets", "MR")
TEST_MR  = os.path.join(CHAOS_RAW, "Test_Sets", "MR")
OUT_NORM    = os.path.join(DATA_DIR, "chaos_MR_T2_normalized")
OUT_NORM672 = os.path.join(DATA_DIR, "chaos_MR_T2_normalized_672")

# CHAOS label value -> class index mapping
LABEL_MAP = {63: 1, 126: 2, 189: 3, 252: 4}  # liver, R-kidney, L-kidney, spleen
LABEL_NAMES = ["LIVER", "RK", "LK", "SPLEEN"]

def find_patients(mr_root, is_train):
    patients = []
    for pid in sorted([d for d in os.listdir(mr_root) if os.path.isdir(os.path.join(mr_root, d))], key=int):
        t2 = os.path.join(mr_root, pid, "T2SPIR")
        dcm = os.path.join(t2, "DICOM_anon")
        gnd = os.path.join(t2, "Ground")
        if os.path.isdir(dcm):
            patients.append((pid, dcm, gnd if os.path.isdir(gnd) else None))
    return patients

def copy_spacing(src, dst):
    for attr in ["SetSpacing", "SetOrigin", "SetDirection"]:
        getattr(dst, attr)(getattr(src, f"Get{attr[3:]}")())
    return dst

def read_dicom_series(dcm_dir):
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(dcm_dir)
    if not names:
        names = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    reader.SetFileNames(names)
    return reader.Execute()

def read_png_labels(gnd_dir):
    files = sorted(glob.glob(os.path.join(gnd_dir, "*.png")))
    if not files:
        return None
    slices = []
    for f in files:
        arr = np.array(Image.open(f))
        mapped = np.zeros_like(arr, dtype=np.uint8)
        for raw_val, cls_idx in LABEL_MAP.items():
            mapped[arr == raw_val] = cls_idx
        slices.append(mapped)
    return np.stack(slices, axis=0)

def resample_volume(img_obj, new_spacing, is_label=False):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resample.SetOutputDirection(img_obj.GetDirection())
    resample.SetOutputOrigin(img_obj.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    old_spacing = np.array(img_obj.GetSpacing())
    old_size = np.array(img_obj.GetSize())
    new_size = (old_size * old_spacing / np.array(new_spacing)).astype(int)
    resample.SetSize([int(s) for s in new_size])
    return resample.Execute(img_obj)

def normalize_mri(img_obj, clip_pct=1):
    arr = sitk.GetArrayFromImage(img_obj).astype(np.float64)
    lo, hi = np.percentile(arr, clip_pct), np.percentile(arr, 100 - clip_pct)
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    return copy_spacing(img_obj, out)

def generate_classmap(out_dir):
    segs = sorted(glob.glob(os.path.join(out_dir, "label_*.nii.gz")),
                  key=lambda x: int(x.split("_")[-1].split(".")[0]))
    cmap = {lb: {str(p): [] for p in range(len(segs))} for lb in LABEL_NAMES}
    for pid, seg in enumerate(segs):
        vol = sitk.GetArrayFromImage(sitk.ReadImage(seg))
        for slc in range(vol.shape[0]):
            for ci, lb in enumerate(LABEL_NAMES):
                if ci in vol[slc] and (vol[slc] == ci).sum() >= 1:
                    cmap[lb][str(pid)].append(slc)
        print(f"  pid {pid}: {vol.shape[0]} slices")
    with open(os.path.join(out_dir, "classmap_1.json"), "w") as f:
        json.dump(cmap, f)
    print(f"  Saved classmap_1.json")

def main():
    os.makedirs(OUT_NORM, exist_ok=True)
    os.makedirs(OUT_NORM672, exist_ok=True)

    print("=== Step 1: Finding T2SPIR patients ===")
    train = find_patients(TRAIN_MR, True)
    test  = find_patients(TEST_MR, False)
    all_p = train + test
    print(f"  Train: {len(train)}, Test: {len(test)}, Total: {len(all_p)}")
    if not all_p:
        print("ERROR: No T2SPIR data found!")
        sys.exit(1)

    TARGET_SP = (1.25, 1.25, 7.7)

    print(f"\n=== Step 2: DICOM -> NIfTI, normalize, resample -> {OUT_NORM} ===")
    for idx, (pid, dcm_dir, gnd_dir) in enumerate(all_p):
        print(f"  [{idx}] patient {pid}")
        img = normalize_mri(read_dicom_series(dcm_dir), clip_pct=1)
        if gnd_dir:
            lbl_vol = read_png_labels(gnd_dir)
            if lbl_vol is not None:
                lbl = sitk.GetImageFromArray(lbl_vol.astype(np.int16))
                lbl = copy_spacing(img, lbl)
            else:
                lbl = sitk.GetImageFromArray(np.zeros(sitk.GetArrayFromImage(img).shape, dtype=np.int16))
                lbl = copy_spacing(img, lbl)
        else:
            lbl = sitk.GetImageFromArray(np.zeros(sitk.GetArrayFromImage(img).shape, dtype=np.int16))
            lbl = copy_spacing(img, lbl)

        img_r = resample_volume(img, TARGET_SP, is_label=False)
        lbl_r = resample_volume(lbl, TARGET_SP, is_label=True)
        sitk.WriteImage(img_r, os.path.join(OUT_NORM, f"image_{idx}.nii.gz"), True)
        sitk.WriteImage(lbl_r, os.path.join(OUT_NORM, f"label_{idx}.nii.gz"), True)

    print(f"\n=== Step 3: 672x672 versions -> {OUT_NORM672} ===")
    BD, RES = 32, 672
    for idx in range(len(all_p)):
        ip = os.path.join(OUT_NORM, f"image_{idx}.nii.gz")
        lp = os.path.join(OUT_NORM, f"label_{idx}.nii.gz")
        if not os.path.exists(ip):
            continue
        img = sitk.ReadImage(ip)
        lbl = sitk.ReadImage(lp)

        arr_i = sitk.GetArrayFromImage(img)
        arr_l = sitk.GetArrayFromImage(lbl)
        H = arr_i.shape[-1]
        spa_fac = (H - 2 * BD) / RES

        arr_i = arr_i[:, BD:-BD, BD:-BD]
        arr_l = arr_l[:, BD:-BD, BD:-BD]
        ci = sitk.GetImageFromArray(arr_i); ci = copy_spacing(img, ci)
        cl = sitk.GetImageFromArray(arr_l.astype(np.int16)); cl = copy_spacing(lbl, cl)

        old_sp = img.GetSpacing()
        new_sp = (old_sp[0] * spa_fac, old_sp[1] * spa_fac, old_sp[2])

        ci_r = resample_volume(ci, new_sp, is_label=False)
        cl_r = resample_volume(cl, new_sp, is_label=True)
        sitk.WriteImage(ci_r, os.path.join(OUT_NORM672, f"image_{idx}.nii.gz"), True)
        sitk.WriteImage(cl_r, os.path.join(OUT_NORM672, f"label_{idx}.nii.gz"), True)
    print(f"  Done: {len(all_p)} volumes")

    print(f"\n=== Step 4: Classmap ===")
    for d in [OUT_NORM, OUT_NORM672]:
        generate_classmap(d)

    print(f"\n=== Done! ===\n  {OUT_NORM}\n  {OUT_NORM672}")
    print("Run:  ./run_protosam.sh mri 0   (kidneys)")
    print("      ./run_protosam.sh mri 1   (liver+spleen)")

if __name__ == "__main__":
    main()
