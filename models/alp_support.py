"""Functional helpers for QSPA support pools on 3D ALP datasets (CHAOS/SABS)."""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch


def resolve_alp_support_scan_ids(config, manual_dataset) -> List:
    """Resolve configured support scan indices to dataset scan IDs."""
    scan_idx = config.get("support_idx", [-1])
    if not isinstance(scan_idx, (list, tuple)):
        scan_idx = [scan_idx]
    return [manual_dataset.pid_curr_load[ii] for ii in scan_idx]


def build_alp_support_pool(
    manual_dataset,
    curr_class: int,
    support_scan_ids: Sequence,
) -> List[Dict]:
    """Collect organ-positive slices from the official support scan(s)."""
    label_name = manual_dataset.label_name[curr_class]
    pool = []
    for scan_id in support_scan_ids:
        z_ids = manual_dataset.tp1_cls_map[label_name][scan_id]
        for z_id in z_ids:
            pool.append({
                "glb_idx": int(manual_dataset.scan_z_idx[scan_id][z_id]),
                "scan_id": scan_id,
                "z_id": int(z_id),
            })
    if not pool:
        raise ValueError(
            "ALP QSPA support pool is empty for class "
            f"{curr_class} on scans {list(support_scan_ids)}"
        )
    return pool


def load_alp_support_slice(
    manual_dataset,
    pool_entry: Dict,
    curr_class: int,
    class_idx: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Load one support slice and binary fg mask for ALPNet/QSPA."""
    curr_dict = manual_dataset.actual_dataset[pool_entry["glb_idx"]]
    img = np.float32(curr_dict["img"])
    lb = np.float32(curr_dict["lb"]).squeeze(-1)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    lb = torch.from_numpy(lb)
    if manual_dataset.tile_z_dim:
        img = img.repeat([manual_dataset.tile_z_dim, 1, 1])
    masks = manual_dataset.getMaskMedImg(lb, curr_class, list(class_idx))
    case = f"{pool_entry['scan_id']}:{pool_entry['z_id']}"
    return img.unsqueeze(0), masks["fg_mask"].unsqueeze(0), case


def make_alp_support_loader(
    manual_dataset,
    pool: Sequence[Dict],
    curr_class: int,
    class_idx: Sequence[int],
) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, str]]:
    """Return index-based loader aligned with precomputed support features."""

    def _load(pool_index: int):
        return load_alp_support_slice(
            manual_dataset, pool[pool_index], curr_class, class_idx)

    return _load
