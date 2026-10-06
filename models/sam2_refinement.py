"""SAM2 image refinement for ProtoSAM (functional API, SAM1/SAM3 paths unchanged).

Loads SAM 2 from an external repo (default: IBISAgent-main/sam2) and runs
box + point refinement like SAM1 ``predict_w_points_bbox``.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

_DEFAULT_SAM2_REPO = "/share/home/huafuchen01/huangwei/YuHuang/IBISAgent-main/sam2"
_DEFAULT_CHECKPOINT = (
    "/share/home/huafuchen01/huangwei/XiongYiming/memory-sam/checkpoints/sam2.1_hiera_large.pt"
)

# Hydra config_name relative to the sam2 package (see sam2/build_sam.py HF map).
_CHECKPOINT_TO_CONFIG = {
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_large.pt": "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
}


def default_sam2_repo() -> str:
    return os.environ.get("SAM2_REPO", _DEFAULT_SAM2_REPO)


def default_sam2_checkpoint() -> str:
    return os.environ.get("SAM2_CHECKPOINT", _DEFAULT_CHECKPOINT)


def is_sam2_refinement_ver(protosam_sam_ver: str) -> bool:
    return (protosam_sam_ver or "").lower() == "sam2"


def is_sam2_checkpoint(path: str) -> bool:
    if not path:
        return False
    base = os.path.basename(path).lower()
    if base in _CHECKPOINT_TO_CONFIG:
        return True
    return "sam2" in base and base.endswith(".pt")


def resolve_sam2_config(checkpoint_path: str, config_override: Optional[str] = None) -> str:
    if config_override:
        return config_override
    base = os.path.basename(checkpoint_path)
    if base not in _CHECKPOINT_TO_CONFIG:
        raise ValueError(
            f"No default SAM2 config for checkpoint {base!r}; "
            f"set sam2_config= or SAM2_CONFIG env. Known: {list(_CHECKPOINT_TO_CONFIG)}"
        )
    return _CHECKPOINT_TO_CONFIG[base]


def _ensure_sam2_importable(repo_root: Optional[str] = None) -> str:
    root = os.path.abspath(repo_root or default_sam2_repo())
    if not os.path.isdir(root):
        raise FileNotFoundError(f"SAM2 repo not found: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


@dataclass
class Sam2RefinementRuntime:
    predictor: object
    device: torch.device
    checkpoint_path: str
    config_name: str


def load_sam2_image_predictor(
    checkpoint_path: Optional[str] = None,
    *,
    config_name: Optional[str] = None,
    device: Optional[str] = None,
    sam2_repo: Optional[str] = None,
) -> Sam2RefinementRuntime:
    """Build ``SAM2ImagePredictor`` from local checkpoint + yaml config."""
    ckpt = checkpoint_path or default_sam2_checkpoint()
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")

    _ensure_sam2_importable(sam2_repo)
    import sam2  # noqa: F401 — initializes Hydra config module
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg = resolve_sam2_config(ckpt, config_name or os.environ.get("SAM2_CONFIG"))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Loading SAM2 config=%s ckpt=%s device=%s", cfg, ckpt, dev)
    sam2_model = build_sam2(config_file=cfg, ckpt_path=ckpt, device=str(dev), mode="eval")
    predictor = SAM2ImagePredictor(sam2_model)
    return Sam2RefinementRuntime(
        predictor=predictor,
        device=dev,
        checkpoint_path=ckpt,
        config_name=cfg,
    )


def predict_masks_points_bbox(
    runtime: Sam2RefinementRuntime,
    qry_img_uint8: np.ndarray,
    sam_input_points: Sequence,
    bboxes: Sequence,
    sam_neg_input_points: Sequence,
    *,
    use_neg_points: bool = False,
    use_cca: bool = False,
    ablation_fixes: Optional[Sequence[str]] = None,
    return_logits: bool = False,
) -> Tuple[List[np.ndarray], List[float]]:
    """Mirror ProtoSAM ``predict_w_points_bbox`` for SAM2ImagePredictor."""
    ablation_fixes = list(ablation_fixes or [])
    predictor = runtime.predictor
    predictor.set_image(qry_img_uint8)

    masks: List[np.ndarray] = []
    scores: List[float] = []

    for point, bbox_xyxy, neg_point in zip(sam_input_points, bboxes, sam_neg_input_points):
        points = point
        point_labels = np.array([1] * len(point), dtype=np.int32) if point is not None else None
        if use_neg_points and neg_point is not None:
            neg_points = [npoint for npoint in neg_point if None not in npoint]
            if neg_points:
                points = np.vstack([point, *neg_points])
                point_labels = np.array(
                    [1] * len(point) + [0] * len(neg_points), dtype=np.int32
                )

        multimask = not (use_cca and "multimask_score" not in ablation_fixes)
        mask, score, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=bbox_xyxy if bbox_xyxy is not None else None,
            mask_input=None,
            multimask_output=multimask,
            return_logits=return_logits,
        )
        best_idx = np.argmax(score) if "multimask_score" in ablation_fixes else 0
        masks.append(mask[best_idx])
        scores.append(float(score[best_idx]))

    return masks, scores
