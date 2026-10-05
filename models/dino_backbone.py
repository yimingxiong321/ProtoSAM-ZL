"""Functional DINO backbone helpers (DINOv2 unchanged path + optional DINOv3).

DINOv2 logic is extracted verbatim from the original ``get_features`` block so
existing ``dinov2_*`` configs behave the same. DINOv3 ViT-L/16 is opt-in via
``which_model=dinov3_l16``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.consts import DEFAULT_FEATURE_SIZE

_PROTO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DINOV3_HF_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_DINOV3_LOCAL_DIR = os.path.join(
    _PROTO_ROOT,
    "pretrained_model",
    "dinov3-vitl16-pretrain-lvd1689m",
)
DEFAULT_DINOV3_H16_HF_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DEFAULT_DINOV3_H16_LOCAL_DIR = os.path.join(
    _PROTO_ROOT,
    "pretrained_model",
    "dinov3-vith16plus-pretrain-lvd1689m",
)

DINOV2_MODELS = frozenset({"dinov2_l14", "dinov2_l14_reg", "dinov2_b14"})
DINOV3_MODELS = frozenset({"dinov3_l16", "dinov3_h16"})


@dataclass(frozen=True)
class DinoBackboneMeta:
    patch_size: int
    embed_dim: int
    family: str  # "v2" | "v3"


_BACKBONE_META = {
    "dinov2_l14": DinoBackboneMeta(14, 1024, "v2"),
    "dinov2_l14_reg": DinoBackboneMeta(14, 1024, "v2"),
    "dinov2_b14": DinoBackboneMeta(14, 768, "v2"),
    "dinov3_l16": DinoBackboneMeta(16, 1024, "v3"),
    "dinov3_h16": DinoBackboneMeta(16, 1280, "v3"),
}


def is_dinov2_backbone(which_model: str) -> bool:
    return which_model in DINOV2_MODELS


def is_dinov3_backbone(which_model: str) -> bool:
    return which_model in DINOV3_MODELS


def is_dino_backbone(which_model: str) -> bool:
    return is_dinov2_backbone(which_model) or is_dinov3_backbone(which_model)


def backbone_meta(which_model: str) -> DinoBackboneMeta:
    try:
        return _BACKBONE_META[which_model]
    except KeyError as exc:
        raise KeyError(f"Unknown DINO backbone: {which_model!r}") from exc


def prototype_embed_dim(which_model: str) -> int:
    return backbone_meta(which_model).embed_dim


def feature_hw_for_image(
    image_size: Union[int, Tuple[int, int]],
    which_model: str,
) -> Tuple[int, int]:
    if isinstance(image_size, int):
        h = w = int(image_size)
    else:
        h, w = int(image_size[0]), int(image_size[1])
    side = max(h, w)
    meta = backbone_meta(which_model)
    grid = max(side // meta.patch_size, DEFAULT_FEATURE_SIZE)
    return grid, grid


def load_dinov2_encoder(which_model: str) -> nn.Module:
    """Load DINOv2 via torch.hub (same as legacy ``get_encoder``)."""
    if which_model == "dinov2_l14":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    if which_model == "dinov2_l14_reg":
        try:
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        except RuntimeError:
            return torch.hub.load(
                "facebookresearch/dino", "dinov2_vitl14_reg", force_reload=True
            )
    if which_model == "dinov2_b14":
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    raise NotImplementedError(f"DINOv2 variant {which_model!r} not supported")


class _DinoV3ForwardAdapter(nn.Module):
    """Expose ``forward_features`` with ``x_norm_patchtokens`` like DINOv2 hub."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        cfg = getattr(inner, "config", None)
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 4))

    def forward_features(self, imgs: torch.Tensor) -> dict:
        tokens = _dinov3_patch_tokens(self.inner, imgs, self.num_register_tokens)
        return {"x_norm_patchtokens": tokens}

    def forward(self, imgs: torch.Tensor) -> dict:
        return self.forward_features(imgs)


def _resolve_dinov3_pretrained_path(
    hf_model_id: Optional[str],
    *,
    which_model: str,
) -> Tuple[str, bool]:
    """Return (path_or_repo_id, use_local_files_only)."""
    if which_model == "dinov3_h16":
        default_local = DEFAULT_DINOV3_H16_LOCAL_DIR
        default_hub = DEFAULT_DINOV3_H16_HF_ID
        env_key = "DINOV3_H16_HF_ID"
    else:
        default_local = DEFAULT_DINOV3_LOCAL_DIR
        default_hub = DEFAULT_DINOV3_HF_ID
        env_key = "DINOV3_HF_ID"

    raw = hf_model_id or os.environ.get(env_key) or os.environ.get("DINOV3_HF_ID")
    candidates: list[str] = []
    if raw:
        candidates.append(raw)
        if not os.path.isabs(raw):
            candidates.append(os.path.join(_PROTO_ROOT, raw))
    candidates.append(default_local)

    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
            return os.path.abspath(path), True
    if raw:
        return raw, os.path.isdir(raw)
    return default_hub, False


def load_dinov3_encoder(
    hf_model_id: Optional[str] = None,
    *,
    which_model: str = "dinov3_l16",
    local_files_only: bool = False,
) -> nn.Module:
    """Load DINOv3 ViT from Hugging Face (Transformers >= 4.56)."""
    model_id, local_dir = _resolve_dinov3_pretrained_path(
        hf_model_id, which_model=which_model
    )
    offline = local_files_only or local_dir
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "DINOv3 requires `transformers>=4.56`. Install in your eval env."
        ) from exc

    kwargs = {}
    if offline:
        kwargs["local_files_only"] = True
    inner = AutoModel.from_pretrained(model_id, **kwargs)
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)
    return _DinoV3ForwardAdapter(inner)


def load_dino_encoder(
    which_model: str,
    *,
    dinov3_hf_id: Optional[str] = None,
) -> nn.Module:
    if is_dinov2_backbone(which_model):
        return load_dinov2_encoder(which_model)
    if which_model in DINOV3_MODELS:
        return load_dinov3_encoder(dinov3_hf_id, which_model=which_model)
    raise NotImplementedError(f"DINO encoder {which_model!r} not implemented")


def _resize_divisible(imgs: torch.Tensor, patch_size: int, image_size: int) -> torch.Tensor:
    side = max(int(image_size), patch_size)
    aligned = (side // patch_size) * patch_size
    if imgs.shape[-2:] == (aligned, aligned):
        return imgs
    return F.interpolate(imgs, size=(aligned, aligned), mode="bilinear", align_corners=False)


def forward_dinov2_patch_features(
    encoder: nn.Module,
    imgs_concat: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Original DINOv2 patch grid path (patch size 14)."""
    meta = DinoBackboneMeta(14, 0, "v2")
    imgs_concat = _resize_divisible(imgs_concat, meta.patch_size, image_size)
    dino_fts = encoder.forward_features(imgs_concat)
    img_fts = dino_fts["x_norm_patchtokens"]
    img_fts = img_fts.permute(0, 2, 1)
    c, hw = img_fts.shape[-2:]
    img_fts = img_fts.view(-1, c, int(hw**0.5), int(hw**0.5))
    if hw < DEFAULT_FEATURE_SIZE**2:
        img_fts = F.interpolate(
            img_fts,
            size=(DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE),
            mode="bilinear",
        )
    return img_fts


def _dinov3_imagenet_norm(imgs: torch.Tensor) -> torch.Tensor:
    """Map dataloader tensors to [0,1] ImageNet-normalized inputs for HF DINOv3."""
    x = imgs.float()
    if x.min() < -0.05 or x.max() > 1.05:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def _dinov3_patch_tokens(
    model: nn.Module,
    imgs: torch.Tensor,
    num_register_tokens: int,
) -> torch.Tensor:
    x = _dinov3_imagenet_norm(imgs)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x = x.to(device=device, dtype=dtype)
    out = model(pixel_values=x)
    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state
    elif isinstance(out, dict) and "last_hidden_state" in out:
        hidden = out["last_hidden_state"]
    else:
        raise TypeError("DINOv3 forward did not return last_hidden_state")
    prefix = 1 + int(num_register_tokens)
    return hidden[:, prefix:, :]


def forward_dinov3_patch_features(
    encoder: nn.Module,
    which_model: str,
    imgs_concat: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """DINOv3 ViT patch grid (patch size 16)."""
    meta = backbone_meta(which_model)
    imgs_concat = _resize_divisible(imgs_concat, meta.patch_size, image_size)
    if not hasattr(encoder, "forward_features"):
        raise TypeError("DINOv3 encoder must expose forward_features")
    dino_fts = encoder.forward_features(imgs_concat)
    tokens = dino_fts["x_norm_patchtokens"]
    img_fts = tokens.permute(0, 2, 1)
    c, hw = img_fts.shape[-2:]
    side = int(math.isqrt(hw))
    if side * side != hw:
        raise ValueError(f"DINOv3 token count {hw} is not a square grid")
    img_fts = img_fts.view(-1, c, side, side)
    if hw < DEFAULT_FEATURE_SIZE**2:
        img_fts = F.interpolate(
            img_fts,
            size=(DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE),
            mode="bilinear",
        )
    return img_fts


def forward_dino_patch_features(
    encoder: nn.Module,
    which_model: str,
    imgs_concat: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    if is_dinov2_backbone(which_model):
        return forward_dinov2_patch_features(encoder, imgs_concat, image_size)
    if is_dinov3_backbone(which_model):
        return forward_dinov3_patch_features(encoder, which_model, imgs_concat, image_size)
    raise NotImplementedError(f"Not a DINO backbone: {which_model!r}")
