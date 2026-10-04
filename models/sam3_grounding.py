"""SAM3 open-vocabulary grounding for ProtoSAM (MedSAM3 / XiongYiming eval path).

Functional API — does not mimic SamPredictor. SAM1 refinement stays in ProtoSAM unchanged.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.ops import nms

_SAM3_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "sam3_pkg"))
if _SAM3_PKG not in sys.path:
    sys.path.insert(0, _SAM3_PKG)

_DEFAULT_CHECKPOINT = (
    "/share/home/huafuchen01/huangwei/WangRuiFeng/MedicalSAM3/checkpoint/sam3.pt"
)
_BPE_CANDIDATES = (
    os.path.join(_SAM3_PKG, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    "/share/home/huafuchen01/huangwei/XiongYiming/MedSAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
)


def is_sam3_checkpoint(path: str) -> bool:
    if not path:
        return False
    base = os.path.basename(path).lower()
    return base in ("sam3.pt", "medsam3.pt") or "sam3" in base


def _resolve_bpe_path() -> str:
    for p in _BPE_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"SAM3 BPE vocab not found; tried {_BPE_CANDIDATES}")


def query_chw_tensor_to_pil(chw: torch.Tensor) -> PILImage.Image:
    """Query tensor (3,H,W) at original resolution → RGB PIL for SAM3 grounding."""
    x = chw.detach().float().cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.min() < -0.01 or x.max() > 1.01:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(arr, mode="RGB")


def xyxy_prompt_space_to_original(
    box_xyxy: Union[np.ndarray, Sequence[float]],
    prompt_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> List[int]:
    """Map bbox from coarse/SAM prompt grid (e.g. 1024) to original image pixels (e.g. 672)."""
    h_p, w_p = int(prompt_hw[0]), int(prompt_hw[1])
    h_o, w_o = int(orig_hw[0]), int(orig_hw[1])
    b = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
    sx = w_o / float(w_p)
    sy = h_o / float(h_p)
    out = np.array([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], dtype=np.float64)
    out[0::2] = np.clip(out[0::2], 0, w_o - 1)
    out[1::2] = np.clip(out[1::2], 0, h_o - 1)
    if out[2] <= out[0]:
        out[2] = min(w_o - 1, out[0] + 1)
    if out[3] <= out[1]:
        out[3] = min(h_o - 1, out[1] + 1)
    return [int(round(v)) for v in out]


def points_prompt_space_to_original(
    points_xy: Union[np.ndarray, Sequence[Sequence[float]]],
    prompt_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> np.ndarray:
    """Map (N,2) points from coarse/SAM prompt grid to original image pixels."""
    if points_xy is None:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return pts
    h_p, w_p = int(prompt_hw[0]), int(prompt_hw[1])
    h_o, w_o = int(orig_hw[0]), int(orig_hw[1])
    sx = w_o / float(w_p)
    sy = h_o / float(h_p)
    out = pts.copy()
    out[:, 0] = np.clip(out[:, 0] * sx, 0, w_o - 1)
    out[:, 1] = np.clip(out[:, 1] * sy, 0, h_o - 1)
    return out


def _points_xy_to_input_tensor(
    points_xy_orig: Optional[np.ndarray],
) -> Optional[torch.Tensor]:
    """Build SAM3 FindQuery input_points: (1, N, 3) with x,y in original pixels, label=1."""
    if points_xy_orig is None or len(points_xy_orig) == 0:
        return None
    pts = np.asarray(points_xy_orig, dtype=np.float32).reshape(-1, 2)
    labels = np.ones((pts.shape[0], 1), dtype=np.float32)
    packed = np.concatenate([pts, labels], axis=1)
    return torch.from_numpy(packed).view(1, -1, 3)


@dataclass
class Sam3GroundingRuntime:
    model: torch.nn.Module
    transform: object
    device: torch.device
    resolution: int
    text_prompt: str
    score_threshold: float
    nms_iou: float


def load_sam3_grounding_runtime(
    checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    resolution: int = 1008,
    text_prompt: str = "polyp",
    score_threshold: float = 0.5,
    nms_iou: float = 0.5,
) -> Sam3GroundingRuntime:
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )

    ckpt = checkpoint_path or os.environ.get("SAM3_CHECKPOINT") or _DEFAULT_CHECKPOINT
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {ckpt}")

    dev = torch.device(device)
    model = build_sam3_image_model(
        bpe_path=_resolve_bpe_path(),
        device=dev.type,
        eval_mode=True,
        checkpoint_path=ckpt,
        load_from_HF=False,
        compile=False,
    )
    model.to(dev)
    model.eval()

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=resolution,
                max_size=resolution,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return Sam3GroundingRuntime(
        model=model,
        transform=transform,
        device=dev,
        resolution=resolution,
        text_prompt=text_prompt,
        score_threshold=score_threshold,
        nms_iou=nms_iou,
    )


def _create_datapoint(
    pil_image: PILImage.Image,
    text: str,
    box_xyxy: Optional[List[int]],
    input_points: Optional[torch.Tensor] = None,
):
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )

    w, h = pil_image.size
    sam_image = SAMImage(data=pil_image, objects=[], size=[h, w])
    input_bbox = None
    input_bbox_label = None
    if box_xyxy is not None:
        input_bbox = torch.tensor(box_xyxy, dtype=torch.float32).view(1, 4)
        input_bbox_label = torch.ones(1, dtype=torch.bool)
    query = FindQueryLoaded(
        query_text=text,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        input_bbox=input_bbox,
        input_bbox_label=input_bbox_label,
        input_points=input_points,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=1,
            original_size=[h, w],
            object_id=0,
            frame_index=0,
        ),
    )
    return Datapoint(find_queries=[query], images=[sam_image])


@torch.inference_mode()
def predict_grounding_mask(
    runtime: Sam3GroundingRuntime,
    pil_image: PILImage.Image,
    box_xyxy_orig: Optional[List[int]],
    *,
    text_prompt: Optional[str] = None,
    input_points: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, float]:
    """T+I grounding: text + box (+ optional points) in **original** pixels. Returns (H,W) bool, score."""
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    h, w = pil_image.size[1], pil_image.size[0]
    if box_xyxy_orig is None:
        return np.zeros((h, w), dtype=bool), 0.0

    text = text_prompt if text_prompt is not None else runtime.text_prompt
    if not text or text == "visual":
        query_text = "visual"
    else:
        query_text = text

    datapoint = _create_datapoint(
        pil_image, query_text, list(box_xyxy_orig), input_points=input_points
    )
    datapoint = runtime.transform(datapoint)
    batch = collate_fn_api([datapoint], dict_key="input")["input"]
    batch = copy_data_to_device(batch, runtime.device, non_blocking=True)

    with torch.autocast(
        device_type=runtime.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.device.type == "cuda",
    ):
        outputs = runtime.model(batch)
    last = outputs[-1]
    pred_logits = last["pred_logits"]
    pred_boxes = last["pred_boxes"]
    pred_masks = last.get("pred_masks")

    scores = pred_logits.sigmoid()[0, :, :].max(dim=-1)[0]
    keep = scores > runtime.score_threshold
    if keep.sum().item() == 0:
        return np.zeros((h, w), dtype=bool), 0.0

    boxes_cxcywh = pred_boxes[0, keep]
    kept_scores = scores[keep]
    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    keep_nms = nms(boxes_xyxy, kept_scores, runtime.nms_iou)
    kept_scores = kept_scores[keep_nms]

    if pred_masks is None:
        return np.zeros((h, w), dtype=bool), float(kept_scores.max().item())

    masks_small = pred_masks[0, keep][keep_nms].sigmoid()
    best_idx = int(torch.argmax(kept_scores).item())
    mask = masks_small[best_idx] > 0.5

    mask_up = (
        torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
        > 0.5
    )
    return mask_up, float(kept_scores[best_idx].item())


def predict_grounding_masks_for_boxes(
    runtime: Sam3GroundingRuntime,
    pil_image: PILImage.Image,
    boxes_prompt_space: Sequence[Optional[np.ndarray]],
    prompt_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    *,
    points_prompt_space: Optional[Sequence[Optional[np.ndarray]]] = None,
    return_logits: bool = False,
) -> Tuple[List[np.ndarray], List[float]]:
    masks: List[np.ndarray] = []
    scores: List[float] = []
    for idx, box in enumerate(boxes_prompt_space):
        if box is None:
            continue
        box_o = xyxy_prompt_space_to_original(box, prompt_hw, orig_hw)
        pts_o = None
        if points_prompt_space is not None and idx < len(points_prompt_space):
            raw_pts = points_prompt_space[idx]
            if raw_pts is not None:
                pts_o = points_prompt_space_to_original(raw_pts, prompt_hw, orig_hw)
        point_tensor = _points_xy_to_input_tensor(pts_o)
        mask_bool, score = predict_grounding_mask(
            runtime, pil_image, box_o, input_points=point_tensor
        )
        if return_logits:
            arr = np.where(mask_bool, 1.0, -1.0).astype(np.float32)
        else:
            arr = mask_bool.astype(np.float32)
        masks.append(arr)
        scores.append(score)
    return masks, scores
