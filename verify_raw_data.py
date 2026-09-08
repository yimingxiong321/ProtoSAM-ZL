#!/usr/bin/env python3
"""Verify raw polyp dataset counts & image-mask filename correspondence."""
import os
import sys

RAW_ROOTS = {
    "Kvasir":           "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/Kvasir-SEG/raw",
    "CVC-ClinicDB":     "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/CVC-ClinicDB/raw",
    "CVC-ColonDB":      "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/colondb/raw",
    "ETIS-LaribPolypDB": "/share/home/huafuchen01/zl/R2Seg-main/benchmark/data/datasets/etis/raw",
}

for ds_name, root in RAW_ROOTS.items():
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")

    if not os.path.isdir(img_dir):
        print(f"[{ds_name}] MISSING images dir: {img_dir}")
        continue
    if not os.path.isdir(mask_dir):
        print(f"[{ds_name}] MISSING masks dir:  {mask_dir}")
        continue

    imgs = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    img_stems = {os.path.splitext(f)[0] for f in imgs}
    mask_stems = {os.path.splitext(f)[0] for f in masks}

    only_imgs = img_stems - mask_stems
    only_masks = mask_stems - img_stems
    matched = img_stems & mask_stems

    img_exts = set(os.path.splitext(f)[1] for f in imgs)
    mask_exts = set(os.path.splitext(f)[1] for f in masks)

    print(f"[{ds_name}]")
    print(f"  images dir:  {len(imgs)} files  (exts: {img_exts})")
    print(f"  masks dir:   {len(masks)} files  (exts: {mask_exts})")
    print(f"  matched:      {len(matched)}")
    if only_imgs:
        show = sorted(only_imgs)[:5]
        tag = " ..." if len(only_imgs) > 5 else ""
        print(f"  orphan images (no mask): {show}{tag}")
    if only_masks:
        show = sorted(only_masks)[:5]
        tag = " ..." if len(only_masks) > 5 else ""
        print(f"  orphan masks (no image): {show}{tag}")
    print()
