import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def tensor_image_to_uint8(img: torch.Tensor) -> np.ndarray:
    arr = _to_numpy(img)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _squeeze_2d(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().float()
    while y.ndim > 2:
        y = y[0]
    return y


def _resize_nearest_2d(mask: torch.Tensor, size_hw) -> np.ndarray:
    m = _squeeze_2d(mask)
    up = F.interpolate(m[None, None], size=size_hw, mode="nearest")[0, 0]
    return _to_numpy(up)


def overlay_mask(image_uint8: np.ndarray, mask_2d: np.ndarray, color_bgr, alpha: float = 0.5) -> np.ndarray:
    base_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    mask = mask_2d > 0.5
    color = np.zeros_like(base_bgr)
    color[..., 0] = color_bgr[0]
    color[..., 1] = color_bgr[1]
    color[..., 2] = color_bgr[2]
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0)
    out = base_bgr.copy()
    out[mask] = blended[mask]
    return out


def save_mask_overlay(image: torch.Tensor, mask: torch.Tensor, out_path: str, color_bgr=(0, 0, 255), alpha: float = 0.5) -> None:
    _ensure_dir(os.path.dirname(out_path))
    image_uint8 = tensor_image_to_uint8(image)
    mask_up = _resize_nearest_2d(mask, image_uint8.shape[:2])
    cv2.imwrite(out_path, overlay_mask(image_uint8, mask_up, color_bgr=color_bgr, alpha=alpha))


def pca_to_rgb(fts: torch.Tensor) -> np.ndarray:
    x = fts.detach().float()
    while x.ndim > 3:
        x = x[0]
    c, h, w = x.shape
    flat = x.reshape(c, -1).t()
    flat = flat - flat.mean(dim=0, keepdim=True)
    try:
        _, _, vh = torch.pca_lowrank(flat, q=min(3, c), center=False)
        comp = flat @ vh[:, :3]
    except Exception:
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        comp = flat @ vh[:3].t()
    if comp.shape[1] < 3:
        comp = F.pad(comp, (0, 3 - comp.shape[1]))
    comp = comp.reshape(h, w, 3)
    comp = comp - comp.amin(dim=(0, 1), keepdim=True)
    comp = comp / (comp.amax(dim=(0, 1), keepdim=True) + 1e-8)
    return np.clip(_to_numpy(comp) * 255.0, 0, 255).astype(np.uint8)


def save_feature_pca(fts: torch.Tensor, out_path: str, target_size: Optional[tuple] = None) -> None:
    _ensure_dir(os.path.dirname(out_path))
    rgb = pca_to_rgb(fts)
    if target_size is not None:
        rgb = cv2.resize(rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_assign_overlay(assign_map: torch.Tensor, base_fts: torch.Tensor, out_path: str, target_size: Optional[tuple] = None, alpha: float = 0.45) -> None:
    _ensure_dir(os.path.dirname(out_path))
    base_rgb = pca_to_rgb(base_fts)
    assign = _squeeze_2d(assign_map)
    assign_np = _to_numpy(assign)
    if assign_np.shape != base_rgb.shape[:2]:
        assign_np = cv2.resize(assign_np, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    max_idx = int(np.nanmax(assign_np)) if assign_np.size else 0
    norm = np.zeros_like(assign_np, dtype=np.uint8) if max_idx <= 0 else np.clip(assign_np / max_idx * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    active = assign_np > 0
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0)
    out = base_bgr.copy()
    out[active] = blended[active]
    if target_size is not None:
        out = cv2.resize(out, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, out)


def save_heatmap(score: torch.Tensor, out_path: str, target_size: Optional[tuple] = None, image: Optional[torch.Tensor] = None, alpha: float = 0.45) -> None:
    _ensure_dir(os.path.dirname(out_path))
    s = _squeeze_2d(score)
    s_np = _to_numpy(s)
    mn, mx = float(np.nanmin(s_np)), float(np.nanmax(s_np))
    s_np = (s_np - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(s_np)
    heat = cv2.applyColorMap(np.clip(s_np * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    if target_size is not None:
        heat = cv2.resize(heat, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    if image is not None:
        img = tensor_image_to_uint8(image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img_bgr.shape[:2] != heat.shape[:2]:
            img_bgr = cv2.resize(img_bgr, (heat.shape[1], heat.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat = cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)
    cv2.imwrite(out_path, heat)


def save_points_overlay(image: torch.Tensor, fg_hw: torch.Tensor, center_idx: torch.Tensor, feature_hw, out_path: str, radius: int = 5) -> None:
    _ensure_dir(os.path.dirname(out_path))
    img = tensor_image_to_uint8(image)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fg_hw is not None and center_idx is not None and len(center_idx) > 0:
        coords = fg_hw.detach().float().cpu()
        idx = center_idx.detach().long().cpu()
        h_feat, w_feat = int(feature_hw[0]), int(feature_hw[1])
        h_img, w_img = out.shape[:2]
        for point in coords[idx]:
            y = int(round(float(point[0]) / max(h_feat - 1, 1) * (h_img - 1)))
            x = int(round(float(point[1]) / max(w_feat - 1, 1) * (w_img - 1)))
            cv2.circle(out, (x, y), radius, (0, 255, 0), thickness=-1)
            cv2.circle(out, (x, y), radius + 1, (0, 0, 0), thickness=1)
    cv2.imwrite(out_path, out)


def save_bar(weights: torch.Tensor, out_path: str, width: int = 640, height: int = 360) -> None:
    _ensure_dir(os.path.dirname(out_path))
    vals = weights.detach().float().cpu().numpy().reshape(-1)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    if vals.size == 0:
        cv2.imwrite(out_path, canvas)
        return
    margin = 40
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    bar_w = max(1, usable_w // max(vals.size, 1))
    vmax = max(float(vals.max()), 1e-8)
    for i, v in enumerate(vals):
        x0 = margin + i * bar_w
        x1 = min(margin + (i + 1) * bar_w - 2, width - margin)
        bh = int(float(v) / vmax * usable_h)
        y0 = height - margin - bh
        cv2.rectangle(canvas, (x0, y0), (x1, height - margin), (60, 120, 240), -1)
        cv2.putText(canvas, f"{float(v):.2f}", (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 1)
    cv2.imwrite(out_path, canvas)
