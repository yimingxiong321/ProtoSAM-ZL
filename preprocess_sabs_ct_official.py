#!/usr/bin/env python3
"""
Preprocess BTCV/SABS CT data following the OFFICIAL ProtoSAM pipeline.
Reproduces data_processing.ipynb CELL 7 (intensity clip), CELL 10 (resample),
and CELL 12 (classmap generation) exactly.

The BTCV dataset (MICCAI 2015 Multi-Atlas Abdomen Labeling) IS the SABS
dataset used by ProtoSAM/ALPNet.  Both use the same 13-organ label scheme:
  0=BG, 1=Spleen, 2=RK, 3=LK, 4=Gallbladder, 5=Esophagus, 6=Liver,
  7=Stomach, 8=Aorta, 9=IVC, 10=PS_Vein, 11=Pancreas, 12=AG_R, 13=AG_L

Pipeline (3 steps, matching the notebook):
  Step 1 (CELL 7): Read raw NIfTI -> clip [-125, 275] HU -> min-max [0,255]
                   -> save to tmp_normalized/image_N.nii.gz + label_N.nii.gz
  Step 2 (CELL 10): Read tmp_normalized -> crop border (BD_BIAS=32)
                   -> resample to 256 and 672 -> save to sabs_CT_normalized/
                   and sabs_CT_normalized_672/
  Step 3 (CELL 12): Generate classmap_1.json for both resolutions

Patient ordering: sorted alphabetically -> reindexed 0..29
  img0001->0, img0002->1, ..., img0010->9, img0021->10, ..., img0040->29
  _SEP = [0, 6, 12, 18, 24, 30]  (5 folds, 6 patients each)

Usage (run from ProtoSAM-main/ on the server):
  python preprocess_sabs_ct_official.py
"""
import os
import sys
import glob
import json
import copy
import numpy as np
import SimpleITK as sitk

# ============================================================
# Paths
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
# Adjust if running from ProtoSAM-main/ directly
if os.path.isdir("./data"):
    BASE = os.getcwd()

DATA_DIR = os.path.join(BASE, "data", "SABS")
TMP_DIR = os.path.join(DATA_DIR, "tmp_normalized")
OUT_256 = os.path.join(DATA_DIR, "sabs_CT_normalized")
OUT_672 = os.path.join(DATA_DIR, "sabs_CT_normalized_672")

# Raw BTCV data (30 training cases with labels)
RAW_DIR = "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/BTCV/raw"

# SABS label names (must match dataset_utils.py REAL_LABEL_NAME)
LABEL_NAMES = [
    "BGD", "SPLEEN", "KID_R", "KID_l", "GALLBLADDER", "ESOPHAGUS",
    "LIVER", "STOMACH", "AORTA", "IVC", "PS_VEIN", "PANCREAS",
    "AG_R", "AG_L"
]

# CT intensity windowing (HU)
LIR = -125
HIR = 275

# Border crop bias (pixels)
BD_BIAS = 32

# ============================================================
# Helper functions (copied from data_processing.ipynb CELL 2/3)
# ============================================================

def copy_spacing_ori(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst


def resample_by_res(mov_img_obj, new_spacing, interpolator=sitk.sitkLinear, logging=True):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_img_obj.GetDirection())
    resample.SetOutputOrigin(mov_img_obj.GetOrigin())
    mov_spacing = mov_img_obj.GetSpacing()
    resample.SetOutputSpacing(new_spacing)
    RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
    new_size = np.array(mov_img_obj.GetSize()) * RES_COE
    resample.SetSize([int(sz + 1) for sz in new_size])
    if logging:
        print("    Spacing: {} -> {}".format(mov_spacing, new_spacing))
        print("    Size {} -> {}".format(mov_img_obj.GetSize(), new_size))
    return resample.Execute(mov_img_obj)


def resample_lb_by_res(mov_lb_obj, new_spacing, interpolator=sitk.sitkLinear,
                       ref_img=None, logging=True):
    """Per-label binary resampling: each label value resampled separately."""
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)
    lbvs = np.unique(src_mat)
    if logging:
        print("    Label values: {}".format(lbvs))
    for idx, lbv in enumerate(lbvs):
        _src_curr_mat = np.float32(src_mat == lbv)
        _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
        _src_curr_obj.CopyInformation(mov_lb_obj)
        _tar_curr_obj = resample_by_res(_src_curr_obj, new_spacing, interpolator, logging=False)
        _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv
        if idx == 0:
            out_vol = _tar_curr_mat
        else:
            out_vol[_tar_curr_mat == lbv] = lbv
    out_obj = sitk.GetImageFromArray(out_vol)
    out_obj.SetSpacing(_tar_curr_obj.GetSpacing())
    if ref_img is not None:
        out_obj.CopyInformation(ref_img)
    return out_obj


def resample_imgs(imgs, segs, pids, scan_dir, bd_bias, spa_fac, required_res=512):
    """Crop border and resample to required_res (CELL 10 logic)."""
    _spa_fac = spa_fac
    for img_fid, seg_fid, pid in zip(imgs, segs, pids):
        print("  Processing pid {} (res={})".format(pid, required_res))
        img_obj = sitk.ReadImage(img_fid)
        seg_obj = sitk.ReadImage(seg_fid)

        # --- Image ---
        array = sitk.GetArrayFromImage(img_obj)
        H = W = array.shape[-1]
        if spa_fac is None:
            _spa_fac = (H - 2 * bd_bias) / required_res

        # Crop border
        array = array[:, bd_bias:-bd_bias, bd_bias:-bd_bias]
        cropped_img_o = sitk.GetImageFromArray(array)
        cropped_img_o = copy_spacing_ori(img_obj, cropped_img_o)

        # Resample
        img_spa_ori = img_obj.GetSpacing()
        res_img_o = resample_by_res(
            cropped_img_o,
            [img_spa_ori[0] * _spa_fac, img_spa_ori[1] * _spa_fac, img_spa_ori[-1]],
            logging=True
        )

        # --- Label ---
        lb_arr = sitk.GetArrayFromImage(seg_obj)
        lb_arr = lb_arr[:, bd_bias:-bd_bias, bd_bias:-bd_bias]
        cropped_lb_o = sitk.GetImageFromArray(lb_arr)
        cropped_lb_o = copy_spacing_ori(seg_obj, cropped_lb_o)

        lb_spa_ori = seg_obj.GetSpacing()
        res_lb_o = resample_lb_by_res(
            cropped_lb_o,
            [lb_spa_ori[0] * _spa_fac, lb_spa_ori[1] * _spa_fac, lb_spa_ori[-1]],
            interpolator=sitk.sitkLinear, ref_img=res_img_o, logging=True
        )

        out_img_fid = os.path.join(scan_dir, "image_{}.nii.gz".format(pid))
        out_lb_fid = os.path.join(scan_dir, "label_{}.nii.gz".format(pid))
        sitk.WriteImage(res_img_o, out_img_fid, True)
        sitk.WriteImage(res_lb_o, out_lb_fid, True)
        print("    saved {} shape {}".format(out_img_fid, res_img_o.GetSize()))


def generate_classmap(scan_dir):
    """Generate classmap_1.json (CELL 12 logic)."""
    print("  Generating classmap for {} ...".format(scan_dir))
    segs = sorted(
        glob.glob(os.path.join(scan_dir, "label_*.nii.gz")),
        key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
    )
    MIN_TP = 1
    classmap = {}
    for lb in LABEL_NAMES:
        classmap[lb] = {}
        for i in range(len(segs)):
            classmap[lb][str(i)] = []

    for pid, seg in enumerate(segs):
        lb_vol = sitk.GetArrayFromImage(sitk.ReadImage(seg))
        n_slice = lb_vol.shape[0]
        for slc in range(n_slice):
            for cls in range(len(LABEL_NAMES)):
                if cls in lb_vol[slc]:
                    if np.sum(lb_vol[slc] == cls) >= MIN_TP:
                        classmap[LABEL_NAMES[cls]][str(pid)].append(slc)
        print("    pid {} done ({} slices)".format(pid, n_slice))

    fid = os.path.join(scan_dir, "classmap_1.json")
    with open(fid, "w") as f:
        json.dump(classmap, f)
    print("    saved {}".format(fid))


# ============================================================
# Main pipeline
# ============================================================

def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUT_256, exist_ok=True)
    os.makedirs(OUT_672, exist_ok=True)

    # Find raw image and label files
    raw_imgs = sorted(glob.glob(os.path.join(RAW_DIR, "img*.nii.gz")))
    raw_segs = sorted(glob.glob(os.path.join(RAW_DIR, "label*.nii.gz")))

    # Filter to training set only (img0001-img0010, img0021-img0040)
    # Testing set (img0061-img0080) has no labels
    raw_imgs = [f for f in raw_imgs if int(os.path.basename(f).split("img")[-1].split(".")[0]) <= 40]
    raw_segs = [f for f in raw_segs if int(os.path.basename(f).split("label")[-1].split(".")[0]) <= 40]

    print("=== Found {} training images, {} labels ===".format(len(raw_imgs), len(raw_segs)))
    assert len(raw_imgs) == 30, "Expected 30 training cases, got {}".format(len(raw_imgs))
    assert len(raw_segs) == 30, "Expected 30 labels, got {}".format(len(raw_segs))

    for img, seg in zip(raw_imgs, raw_segs):
        print("  {}  {}".format(os.path.basename(img), os.path.basename(seg)))

    # ============================================================
    # Step 1: Intensity normalization (CELL 7)
    # Clip to [-125, 275] HU, then min-max normalize to [0, 255]
    # ============================================================
    print("\n=== Step 1: Intensity normalization (clip [-125,275] -> [0,255]) ===")
    reindex = 0
    for img_fid, seg_fid in zip(raw_imgs, raw_segs):
        print("  [{}] {} -> image_{}".format(reindex, os.path.basename(img_fid), reindex))
        img_obj = sitk.ReadImage(img_fid)
        seg_obj = sitk.ReadImage(seg_fid)

        array = sitk.GetArrayFromImage(img_obj)
        print("    shape: {}, label shape: {}".format(
            array.shape, sitk.GetArrayFromImage(seg_obj).shape))

        # HU windowing
        array[array > HIR] = HIR
        array[array < LIR] = LIR

        # Min-max normalize to [0, 255]
        array = (array - array.min()) / (array.max() - array.min()) * 255.0

        wined_img = sitk.GetImageFromArray(array)
        wined_img = copy_spacing_ori(img_obj, wined_img)

        out_img_fid = os.path.join(TMP_DIR, "image_{}.nii.gz".format(reindex))
        out_lb_fid = os.path.join(TMP_DIR, "label_{}.nii.gz".format(reindex))
        sitk.WriteImage(wined_img, out_img_fid, True)
        sitk.WriteImage(seg_obj, out_lb_fid, True)  # label saved as-is
        print("    saved {}".format(out_img_fid))
        reindex += 1

    # ============================================================
    # Step 2: Border crop + resample (CELL 10)
    # Crop BD_BIAS=32 pixels from each border, resample to 256 and 672
    # ============================================================
    print("\n=== Step 2: Border crop (BD_BIAS={}) + resample ===".format(BD_BIAS))

    tmp_imgs = sorted(
        glob.glob(os.path.join(TMP_DIR, "image_*.nii.gz")),
        key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
    )
    tmp_segs = sorted(
        glob.glob(os.path.join(TMP_DIR, "label_*.nii.gz")),
        key=lambda x: int(x.split("_")[-1].split(".nii.gz")[0])
    )
    pids = [str(i) for i in range(len(tmp_imgs))]

    for res, out_dir in [(256, OUT_256), (672, OUT_672)]:
        print("\n  --- Resampling to {}x{} ---".format(res, res))
        resample_imgs(tmp_imgs, tmp_segs, pids, out_dir,
                      bd_bias=BD_BIAS, spa_fac=None, required_res=res)

    # ============================================================
    # Step 3: Generate classmap_1.json (CELL 12)
    # ============================================================
    print("\n=== Step 3: Generate classmap_1.json ===")
    generate_classmap(OUT_256)
    generate_classmap(OUT_672)

    # ============================================================
    # Done
    # ============================================================
    print("\n=== Done! ===")
    print("  256: {}".format(OUT_256))
    print("  672: {}".format(OUT_672))
    print("  tmp: {} (can be removed after verification)".format(TMP_DIR))
    print("\nRun CT evaluation:")
    print("  sed -i 's/ORGAN=\".*\"/ORGAN=\"rk\"/' run_protosam.sh && ./run_protosam.sh ct")
    print("  sed -i 's/ORGAN=\".*\"/ORGAN=\"lk\"/' run_protosam.sh && ./run_protosam.sh ct")
    print("  sed -i 's/ORGAN=\".*\"/ORGAN=\"liver\"/' run_protosam.sh && ./run_protosam.sh ct")
    print("  sed -i 's/ORGAN=\".*\"/ORGAN=\"spleen\"/' run_protosam.sh && ./run_protosam.sh ct")


if __name__ == "__main__":
    main()
