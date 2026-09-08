#!/usr/bin/env python3
"""
Preprocess CHAOS MR T2SPIR data following the OFFICIAL ProtoSAM pipeline.
Reproduces data_processing.ipynb CELL 10-12 exactly, but reads DICOM directly
since dcm_img_to_nii.sh is not in the repo.

Key differences from the user's previous preprocess_chaos_mr_v2.py:
  1. Center-crop to 256x256 (the user's script skipped this entirely)
  2. Only cut top 0.5% of histogram (not 1%-99% both ends)
  3. Per-label binary resampling for labels (not direct nearest-neighbor)
  4. Generate 672 from the 256 version (not from the full uncropped volume)
  5. classmap uses enumerate start=1 to avoid off-by-one

Usage:
  python preprocess_chaos_mr_official.py
Run from ProtoSAM-main/ directory on the server.
"""
import os, sys, glob, json, copy
import numpy as np
import SimpleITK as sitk
from PIL import Image

# ---- Paths ----
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "CHAOST2")
CHAOS_RAW = "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/CHAOS/raw"
TRAIN_MR = os.path.join(CHAOS_RAW, "Train_Sets", "MR")
OUT_NORM = os.path.join(DATA_DIR, "chaos_MR_T2_normalized")
OUT_NORM672 = os.path.join(DATA_DIR, "chaos_MR_T2_normalized_672")

LABEL_MAP = {63: 1, 126: 2, 189: 3, 252: 4}
LABEL_NAMES = ["BG", "LIVER", "RK", "LK", "SPLEEN"]

# ============================================================
# Helper functions (from data_processing.ipynb CELL 2)
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
        print("  Spacing: {} -> {}".format(mov_spacing, new_spacing))
        print("  Size {} -> {}".format(mov_img_obj.GetSize(), new_size))
    return resample.Execute(mov_img_obj)

def resample_lb_by_res(mov_lb_obj, new_spacing, interpolator=sitk.sitkLinear,
                       ref_img=None, logging=True):
    """Per-label binary resampling: each label resampled separately."""
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)
    lbvs = np.unique(src_mat)
    if logging:
        print("  Label values: {}".format(lbvs))
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

def image_crop(ori_vol, crop_size, referece_ctr_idx, padval=0.0, only_2d=True):
    """Crop a 3D matrix to crop_size centered at referece_ctr_idx (2D only)."""
    _expand_cropsize = [x + 1 for x in crop_size]
    if only_2d:
        _expand_cropsize.append(ori_vol.shape[-1])
    image_patch = np.ones(tuple(_expand_cropsize)) * padval
    half_size = tuple([int(x * 1.0 / 2) for x in _expand_cropsize])
    _bias_start = [0, 0, 0]
    _bias_end = [0, 0, 0]
    for dim, hsize in enumerate(half_size):
        if dim == 2 and only_2d:
            break
        _bias_start[dim] = np.min([hsize, referece_ctr_idx[dim]])
        _bias_end[dim] = np.min([hsize, ori_vol.shape[dim] - referece_ctr_idx[dim]])
    if only_2d:
        image_patch[half_size[0] - _bias_start[0]:half_size[0] + _bias_end[0],
                    half_size[1] - _bias_start[1]:half_size[1] + _bias_end[1],
                    ...] = \
            ori_vol[referece_ctr_idx[0] - _bias_start[0]:referece_ctr_idx[0] + _bias_end[0],
                    referece_ctr_idx[1] - _bias_start[1]:referece_ctr_idx[1] + _bias_end[1],
                    ...]
        image_patch = image_patch[0:crop_size[0], 0:crop_size[1], :]
    return image_patch

def resample_imgs(imgs, segs, pids, scan_dir, BD_BIAS, SPA_FAC, required_res=512):
    """Generate 672 version from 256 version (CELL 11 logic)."""
    spa_fac = SPA_FAC
    for img_fid, seg_fid, pid in zip(imgs, segs, pids):
        img_obj = sitk.ReadImage(img_fid)
        seg_obj = sitk.ReadImage(seg_fid)
        array = sitk.GetArrayFromImage(img_obj)
        H = W = array.shape[-1]
        if SPA_FAC is None:
            spa_fac = (H - 2 * BD_BIAS) / required_res
        array = array[:, BD_BIAS:-BD_BIAS, BD_BIAS:-BD_BIAS]
        cropped_img_o = sitk.GetImageFromArray(array)
        cropped_img_o = copy_spacing_ori(img_obj, cropped_img_o)
        img_spa_ori = img_obj.GetSpacing()
        res_img_o = resample_by_res(cropped_img_o,
                                     [img_spa_ori[0] * spa_fac, img_spa_ori[1] * spa_fac, img_spa_ori[-1]],
                                     logging=True)
        lb_arr = sitk.GetArrayFromImage(seg_obj)
        lb_arr = lb_arr[:, BD_BIAS:-BD_BIAS, BD_BIAS:-BD_BIAS]
        cropped_lb_o = sitk.GetImageFromArray(lb_arr)
        cropped_lb_o = copy_spacing_ori(seg_obj, cropped_lb_o)
        lb_spa_ori = seg_obj.GetSpacing()
        res_lb_o = resample_lb_by_res(cropped_lb_o,
                                       [lb_spa_ori[0] * spa_fac, lb_spa_ori[1] * spa_fac, lb_spa_ori[-1]],
                                       interpolator=sitk.sitkLinear, ref_img=res_img_o, logging=True)
        out_img_fid = os.path.join(scan_dir, 'image_{}.nii.gz'.format(pid))
        out_lb_fid = os.path.join(scan_dir, 'label_{}.nii.gz'.format(pid))
        sitk.WriteImage(res_img_o, out_img_fid, True)
        sitk.WriteImage(res_lb_o, out_lb_fid, True)
        print("  {} saved, shape: {}".format(out_img_fid, res_img_o.GetSize()))

# ============================================================
# DICOM reading helpers
# ============================================================

def find_patients(mr_root):
    patients = []
    for pid in sorted([d for d in os.listdir(mr_root)
                       if os.path.isdir(os.path.join(mr_root, d))], key=int):
        t2 = os.path.join(mr_root, pid, "T2SPIR")
        dcm = os.path.join(t2, "DICOM_anon")
        gnd = os.path.join(t2, "Ground")
        if os.path.isdir(dcm):
            patients.append((pid, dcm, gnd if os.path.isdir(gnd) else None))
    return patients

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

# ============================================================
# Main pipeline
# ============================================================

def main():
    os.makedirs(OUT_NORM, exist_ok=True)
    os.makedirs(OUT_NORM672, exist_ok=True)

    print("=== Step 0: Finding T2SPIR patients ===")
    train = find_patients(TRAIN_MR)
    print("  Train patients: {}".format(len(train)))
    if len(train) == 0:
        print("ERROR: No T2SPIR training data found at {}".format(TRAIN_MR))
        sys.exit(1)

    HIST_CUT_TOP = 0.5
    NEW_SPA = [1.25, 1.25, 7.70]

    print("\n=== Step 1: DICOM -> clip -> resample -> CENTER CROP 256 ===")
    for idx, (pid, dcm_dir, gnd_dir) in enumerate(train):
        print("  [{}] patient {}".format(idx, pid))
        img_obj = read_dicom_series(dcm_dir)

        # Image: cut top 0.5% only (NOT 1%-99%)
        array = sitk.GetArrayFromImage(img_obj)
        hir = float(np.percentile(array, 100.0 - HIST_CUT_TOP))
        array[array > hir] = hir
        his_img_o = sitk.GetImageFromArray(array)
        his_img_o = copy_spacing_ori(img_obj, his_img_o)

        # Resample to (1.25, 1.25, 7.70)
        res_img_o = resample_by_res(his_img_o, NEW_SPA, logging=True)

        # Label
        if gnd_dir:
            lbl_vol = read_png_labels(gnd_dir)
            lbl = sitk.GetImageFromArray(lbl_vol.astype(np.int16))
            lbl = copy_spacing_ori(img_obj, lbl)
        else:
            print("  WARNING: no Ground for patient {}, using zeros".format(pid))
            lbl = sitk.GetImageFromArray(
                np.zeros(sitk.GetArrayFromImage(img_obj).shape, dtype=np.int16))
            lbl = copy_spacing_ori(img_obj, lbl)

        # Per-label binary resampling
        res_lb_o = resample_lb_by_res(lbl, NEW_SPA, interpolator=sitk.sitkLinear,
                                      ref_img=None, logging=True)

        # CENTER CROP to 256x256 (the step that was missing!)
        res_img_a = sitk.GetArrayFromImage(res_img_o)
        crop_img_a = image_crop(res_img_a.transpose(1, 2, 0), [256, 256],
                                referece_ctr_idx=[res_img_a.shape[1] // 2,
                                                  res_img_a.shape[2] // 2],
                                padval=res_img_a.min(), only_2d=True).transpose(2, 0, 1)
        out_img_obj = copy_spacing_ori(res_img_o, sitk.GetImageFromArray(crop_img_a))

        res_lb_a = sitk.GetArrayFromImage(res_lb_o)
        crop_lb_a = image_crop(res_lb_a.transpose(1, 2, 0), [256, 256],
                               referece_ctr_idx=[res_lb_a.shape[1] // 2,
                                                 res_lb_a.shape[2] // 2],
                               padval=0, only_2d=True).transpose(2, 0, 1)
        out_lb_obj = copy_spacing_ori(res_img_o, sitk.GetImageFromArray(crop_lb_a))

        out_img_fid = os.path.join(OUT_NORM, 'image_{}.nii.gz'.format(idx))
        out_lb_fid = os.path.join(OUT_NORM, 'label_{}.nii.gz'.format(idx))
        sitk.WriteImage(out_img_obj, out_img_fid, True)
        sitk.WriteImage(out_lb_obj, out_lb_fid, True)
        print("  saved {} shape {}".format(out_img_fid, out_img_obj.GetSize()))

    print("\n=== Step 2: Generate 672 from 256 (BD_BIAS=1) ===")
    imgs = sorted(glob.glob(OUT_NORM + "/image_*.nii.gz"),
                  key=lambda x: int(x.split("_")[-1].split(".")[0]))
    segs = sorted(glob.glob(OUT_NORM + "/label_*.nii.gz"),
                  key=lambda x: int(x.split("_")[-1].split(".")[0]))
    pids = [str(i) for i in range(len(imgs))]
    resample_imgs(imgs, segs, pids, OUT_NORM672, BD_BIAS=1, SPA_FAC=None, required_res=672)

    print("\n=== Step 3: Generate classmap ===")
    for d in [OUT_NORM, OUT_NORM672]:
        print("  Processing {} ...".format(d))
        segs = sorted(glob.glob(os.path.join(d, "label_*.nii.gz")),
                      key=lambda x: int(x.split("_")[-1].split(".")[0]))
        classmap = {}
        MIN_TP = 1
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
        fid = os.path.join(d, 'classmap_1.json')
        with open(fid, 'w') as f:
            json.dump(classmap, f)
        print("    saved {}".format(fid))

    print("\n=== Done! ===")
    print("  256: {}".format(OUT_NORM))
    print("  672: {}".format(OUT_NORM672))
    print("\nRun:  ./run_protosam.sh mri 0   (kidneys)")
    print("      ./run_protosam.sh mri 1   (liver+spleen)")

if __name__ == "__main__":
    main()
