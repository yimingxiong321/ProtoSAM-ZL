"""Read-only proposal auditing for the ProtoSAM P1 experiment.

The functions in this module inspect coarse foreground probabilities before
ProtoSAM's single-component CCA.  They never return a mask to the inference
path, so enabling the audit cannot change prompts or the final SAM prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

try:  # ProtoSAM's server environment has OpenCV; tests also support a fallback.
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal environments
    cv2 = None


DEFAULT_PROPOSAL_THRESHOLDS = (0.3, 0.4, 0.5)


def _as_2d_numpy(value, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().to(device="cpu").numpy()
    value = np.asarray(value)
    while value.ndim > 2:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"{name} must resolve to [H,W], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _resize_map(value, shape: Tuple[int, int], *, binary: bool) -> np.ndarray:
    value = _as_2d_numpy(value, "map")
    if value.shape == shape:
        return value
    tensor = torch.as_tensor(value, dtype=torch.float32)[None, None]
    if binary:
        resized = F.interpolate(tensor, size=shape, mode="nearest")
    else:
        resized = F.interpolate(
            tensor, size=shape, mode="bilinear", align_corners=False
        )
    return resized[0, 0].numpy()


def _connected_components(mask: np.ndarray):
    """Return ``(component_id, ys, xs)`` tuples using 8-connectivity."""

    mask = np.asarray(mask, dtype=np.uint8)
    if cv2 is not None:
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        return [
            (component_id, *np.where(labels == component_id))
            for component_id in range(1, count)
        ]

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    component_id = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            component_id += 1
            stack = [(y, x)]
            visited[y, x] = True
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < height and 0 <= nx < width
                            and mask[ny, nx] and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            components.append(
                (component_id, np.asarray(ys, dtype=int), np.asarray(xs, dtype=int))
            )
    return components


def audit_candidate_proposals(
    foreground_probability,
    likelihood_ratio_map,
    gt_mask,
    thresholds: Sequence[float] = DEFAULT_PROPOSAL_THRESHOLDS,
):
    """Return component rows and one complete slice row per threshold.

    ``coverage`` is the fraction of all GT pixels covered by a component;
    ``gt_overlap`` is component/GT IoU; ``purity`` is the fraction of the
    component inside GT.  The likelihood-ratio map is resized only for
    reporting and is not used to generate candidates.
    """

    probability = _as_2d_numpy(foreground_probability, "foreground_probability")
    shape = probability.shape
    likelihood = _resize_map(likelihood_ratio_map, shape, binary=False)
    gt = _resize_map(gt_mask, shape, binary=True) > 0
    gt_area = int(gt.sum())
    image_area = int(probability.size)

    candidate_rows: List[dict] = []
    slice_rows: List[dict] = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("proposal thresholds must be in [0, 1]")
        proposal = (probability >= threshold).astype(np.uint8)
        components = _connected_components(proposal)
        proposal_area = int(proposal.sum())
        proposal_intersection = int(np.logical_and(proposal > 0, gt).sum())
        oracle_area = 0
        oracle_intersection = 0

        order = sorted(components, key=lambda item: len(item[1]), reverse=True)
        for rank, (component_id, ys, xs) in enumerate(order, start=1):
            component = np.zeros(shape, dtype=bool)
            component[ys, xs] = True
            area = int(len(ys))
            intersection = int(np.logical_and(component, gt).sum())
            union = area + gt_area - intersection
            if intersection > 0:
                oracle_area += area
                oracle_intersection += intersection
            component_probability = probability[component]
            component_likelihood = likelihood[component]
            candidate_rows.append({
                "threshold": threshold,
                "component_id": int(component_id),
                "candidate_rank": rank,
                "area_pixels": area,
                "area_ratio": area / float(image_area),
                "bbox_x": int(xs.min()),
                "bbox_y": int(ys.min()),
                "bbox_width": int(xs.max() - xs.min() + 1),
                "bbox_height": int(ys.max() - ys.min() + 1),
                "mean_probability": float(component_probability.mean()),
                "max_probability": float(component_probability.max()),
                "probability_mass": float(component_probability.sum()),
                "mean_likelihood_ratio": float(component_likelihood.mean()),
                "max_likelihood_ratio": float(component_likelihood.max()),
                "gt_present": bool(gt_area > 0),
                "gt_area_pixels": gt_area,
                "intersection_pixels": intersection,
                "coverage": intersection / float(gt_area) if gt_area else 0.0,
                "gt_overlap": intersection / float(union) if union else 0.0,
                "purity": intersection / float(area) if area else 0.0,
                "oracle_keep": bool(intersection > 0),
            })

        slice_rows.append({
            "threshold": threshold,
            "image_area_pixels": image_area,
            "gt_present": bool(gt_area > 0),
            "gt_area_pixels": gt_area,
            "candidate_count": len(order),
            "proposal_area_pixels": proposal_area,
            "proposal_intersection_pixels": proposal_intersection,
            "proposal_recall": (
                proposal_intersection / float(gt_area) if gt_area else 0.0
            ),
            "proposal_hit": bool(gt_area > 0 and proposal_intersection > 0),
            "oracle_area_pixels": oracle_area,
            "oracle_intersection_pixels": oracle_intersection,
        })
    return candidate_rows, slice_rows


def summarize_proposal_slices(rows: Iterable[Mapping]):
    """Aggregate P1 proposal and oracle-filtering diagnostics by threshold."""

    def as_bool(value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no", ""}:
                return False
            raise ValueError(f"invalid boolean value: {value!r}")
        return bool(value)

    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["threshold"]), []).append(row)
    summaries = []
    for threshold in sorted(grouped):
        items = grouped[threshold]
        positive = [row for row in items if as_bool(row["gt_present"])]
        negative = [row for row in items if not as_bool(row["gt_present"])]
        gt_area = sum(int(row["gt_area_pixels"]) for row in positive)
        intersection = sum(
            int(row["proposal_intersection_pixels"]) for row in positive
        )
        oracle_area = sum(int(row["oracle_area_pixels"]) for row in positive)
        oracle_intersection = sum(
            int(row["oracle_intersection_pixels"]) for row in positive
        )
        negative_pixels = sum(int(row["image_area_pixels"]) for row in negative)
        negative_fp = sum(int(row["proposal_area_pixels"]) for row in negative)
        summaries.append({
            "threshold": threshold,
            "slice_count": len(items),
            "positive_slice_count": len(positive),
            "negative_slice_count": len(negative),
            "proposal_recall": intersection / float(gt_area) if gt_area else 0.0,
            "positive_slice_hit_rate": (
                sum(as_bool(row["proposal_hit"]) for row in positive) / float(len(positive))
                if positive else 0.0
            ),
            "mean_candidates_per_slice": (
                sum(int(row["candidate_count"]) for row in items) / float(len(items))
                if items else 0.0
            ),
            "negative_fp_area_ratio": (
                negative_fp / float(negative_pixels) if negative_pixels else 0.0
            ),
            "oracle_precision": (
                oracle_intersection / float(oracle_area) if oracle_area else 0.0
            ),
            "oracle_recall": (
                oracle_intersection / float(gt_area) if gt_area else 0.0
            ),
            "oracle_dice": (
                2.0 * oracle_intersection / float(oracle_area + gt_area)
                if oracle_area + gt_area else 0.0
            ),
        })
    return summaries
