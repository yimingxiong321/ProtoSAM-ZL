"""Training-free mutual-nearest-neighbour prompt selection."""

import numpy as np
import torch
from torch.nn import functional as F


def select_mnn_prompt_points(
    support_features,
    query_features,
    support_mask,
    query_candidate_mask,
    max_points=1,
    hard_threshold=0.95,
    min_feature_distance=2.0,
    eps=1e-6,
):
    """Select reciprocal DINO patch matches inside a coarse query candidate.

    MNN is used only to rank positive SAM prompts. It never hard-masks the
    coarse prediction, so failure to find a reciprocal match safely returns an
    empty result and lets ProtoSAM retain its original confidence point.
    """
    if support_features.ndim != 3 or query_features.ndim != 3:
        raise ValueError("support_features and query_features must be [C,H,W]")
    if max_points < 1:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    device = query_features.device
    dtype = query_features.dtype
    support_mask = torch.as_tensor(support_mask, device=device, dtype=dtype).squeeze()
    query_candidate_mask = torch.as_tensor(
        query_candidate_mask, device=device, dtype=dtype
    ).squeeze()
    if support_mask.ndim != 2 or query_candidate_mask.ndim != 2:
        raise ValueError("support_mask and query_candidate_mask must reduce to [H,W]")

    hs, ws = support_features.shape[-2:]
    hq, wq = query_features.shape[-2:]
    support_occupancy = F.interpolate(
        support_mask[None, None], size=(hs, ws), mode="area"
    )[0, 0]
    support_indices = torch.nonzero(
        support_occupancy > hard_threshold, as_tuple=False
    )
    query_candidates = F.interpolate(
        query_candidate_mask[None, None], size=(hq, wq), mode="nearest"
    )[0, 0] > 0.5
    query_indices = torch.nonzero(query_candidates, as_tuple=False)
    if support_indices.numel() == 0 or query_indices.numel() == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    support_tokens = support_features[:, support_indices[:, 0], support_indices[:, 1]].T
    query_tokens = query_features[:, query_indices[:, 0], query_indices[:, 1]].T
    support_tokens = F.normalize(support_tokens, dim=1, eps=eps)
    query_tokens = F.normalize(query_tokens, dim=1, eps=eps)
    correlation = query_tokens @ support_tokens.T

    query_best_support = correlation.argmax(dim=1)
    support_best_query = correlation.argmax(dim=0)
    query_order = torch.arange(query_tokens.shape[0], device=device)
    reciprocal = support_best_query[query_best_support] == query_order
    reciprocal_rows = torch.nonzero(reciprocal, as_tuple=False).flatten()
    if reciprocal_rows.numel() == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    reciprocal_scores = correlation[
        reciprocal_rows, query_best_support[reciprocal_rows]
    ]
    ranked_rows = reciprocal_rows[torch.argsort(reciprocal_scores, descending=True)]
    selected_rows = []
    for row in ranked_rows.tolist():
        coordinate = query_indices[row].float()
        if all(
            torch.linalg.vector_norm(coordinate - query_indices[old].float()).item()
            >= min_feature_distance
            for old in selected_rows
        ):
            selected_rows.append(row)
        if len(selected_rows) == max_points:
            break

    selected = torch.as_tensor(selected_rows, device=device, dtype=torch.long)
    feature_yx = query_indices[selected].float()
    image_h, image_w = query_candidate_mask.shape
    points_xy = torch.stack(
        [
            (feature_yx[:, 1] + 0.5) * image_w / wq - 0.5,
            (feature_yx[:, 0] + 0.5) * image_h / hq - 0.5,
        ],
        dim=1,
    )
    points_xy[:, 0].clamp_(0, image_w - 1)
    points_xy[:, 1].clamp_(0, image_h - 1)
    selected_scores = correlation[selected, query_best_support[selected]]
    return (
        points_xy.detach().cpu().numpy().astype(np.float32, copy=False),
        selected_scores.detach().cpu().numpy().astype(np.float32, copy=False),
    )
