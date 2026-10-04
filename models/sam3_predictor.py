"""SAM3 backend for ProtoSAM (SamPredictor-compatible subset)."""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch

_SAM3_PKG = os.path.join(os.path.dirname(__file__), "..", "third_party", "sam3_pkg")
_MEDSAM3_ROOT = os.environ.get(
    "MEDICAL_SAM3_ROOT",
    "/share/home/huafuchen01/huangwei/WangRuiFeng",
)
for _p in (_SAM3_PKG, _MEDSAM3_ROOT):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def is_sam3_checkpoint(path: str) -> bool:
    if not path:
        return False
    base = os.path.basename(path).lower()
    return base in ("sam3.pt", "medsam3.pt") or "sam3" in base


class Sam3PredictorCompat:
    """Minimal SamPredictor API: set_image + predict(points/box)."""

    def __init__(self, wrapper, resolution: int, device: torch.device, text_prompt: str = "polyp"):
        self.wrapper = wrapper
        self.resolution = int(resolution)
        self.device = device
        self.text_prompt = text_prompt
        self._orig_hw: Optional[tuple[int, int]] = None
        # Optional ResizeLongestSide (same as SamPredictor) for prompt coords in pre-SAM image space.
        self.transform = None
        self.original_size: Optional[tuple[int, int]] = None

    def set_image(self, image: np.ndarray) -> None:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        self._orig_hw = (int(image.shape[0]), int(image.shape[1]))
        self.original_size = self._orig_hw
        self._image = np.ascontiguousarray(image)

    def _prompt_coords_to_model_space(self, point_coords, box):
        """Map prompts from qry_img space to the tensor passed into SAM3 forward."""
        h, w = self._orig_hw
        if self.transform is not None and self.original_size is not None:
            if point_coords is not None:
                point_coords = self.transform.apply_coords(
                    np.asarray(point_coords, dtype=np.float32), self.original_size
                )
            if box is not None:
                box = self.transform.apply_boxes(
                    np.asarray(box, dtype=np.float32).reshape(1, 4), self.original_size
                )[0]
        sx = self.resolution / float(w)
        sy = self.resolution / float(h)
        if point_coords is not None:
            pts = np.asarray(point_coords, dtype=np.float32).copy()
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            point_coords = pts
        if box is not None:
            b = np.asarray(box, dtype=np.float32).reshape(-1, 4).copy()
            b[:, [0, 2]] *= sx
            b[:, [1, 3]] *= sy
            box = b.reshape(-1)
        return point_coords, box

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        if self._orig_hw is None:
            raise RuntimeError("call set_image before predict")
        if mask_input is not None:
            raise NotImplementedError("SAM3 backend does not support mask_input prompts yet")

        point_coords, box = self._prompt_coords_to_model_space(point_coords, box)

        img = cv2.resize(self._image, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

        boxes_t = None
        if box is not None:
            boxes_t = torch.from_numpy(np.asarray(box, dtype=np.float32).reshape(-1, 4)).to(self.device)
            if boxes_t.dim() == 1:
                boxes_t = boxes_t.unsqueeze(0)

        points_t = None
        labels_t = None
        if point_coords is not None and len(point_coords) > 0:
            points_t = torch.from_numpy(np.asarray(point_coords, dtype=np.float32)).to(self.device)
            if points_t.dim() == 1:
                points_t = points_t.unsqueeze(0)
            points_t = points_t.unsqueeze(0)  # [1, N, 2]
            if point_labels is not None:
                labels_t = torch.as_tensor(point_labels, device=self.device, dtype=torch.long).unsqueeze(0)
            else:
                labels_t = torch.ones((1, points_t.shape[1]), device=self.device, dtype=torch.long)

        with torch.inference_mode():
            out = self.wrapper(
                images=img_t,
                boxes=boxes_t,
                points=points_t,
                point_labels=labels_t,
                text_prompt=[self.text_prompt],
            )

        h, w = self._orig_hw
        if return_logits and "mask_logits" in out:
            mask = out["mask_logits"][0, 0].detach().float().cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            prob = out["masks"][0, 0].detach().float().cpu().numpy()
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
            # ProtoSAM fuses masks with `pred > 0`; soft probabilities must be binarized like SAM1.
            mask = (prob >= 0.5).astype(np.float32)
        score = float(out["scores"][0, 0].detach().cpu())
        if multimask_output:
            return np.stack([mask, mask, mask], axis=0), np.array([score, score * 0.99, score * 0.98]), None
        return np.stack([mask], axis=0), np.array([score]), None


class _SamImageEncoderStub:
    img_size = 1024


class _SamStub:
    image_encoder = _SamImageEncoderStub()

    def requires_grad_(self, _req=True):
        return self

    def to(self, device):
        return self

    def eval(self):
        return self


def build_sam3_predictor_compat(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: str = "fp32",
    text_prompt: str = "polyp",
) -> Sam3PredictorCompat:
    torch.set_grad_enabled(False)
    from MedicalSAM3.sam3_official.build_model import build_official_sam3_image_model
    from MedicalSAM3.sam3_official.tensor_forward import Sam3TensorForwardWrapper

    model = build_official_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
        compile_model=False,
        allow_dummy_fallback=False,
    )
    wrapper = Sam3TensorForwardWrapper(model, device=device, dtype=dtype, use_hooks=False)
    if not wrapper.is_official_sam3 or wrapper.used_dummy_fallback:
        raise RuntimeError("SAM3 official model failed to load (dummy fallback or missing sam3 runtime)")
    resolution = 1008
    try:
        from MedicalSAM3.sam3_official.tensor_forward import _infer_official_resolution

        resolution = int(_infer_official_resolution(model))
    except Exception:
        pass
    dev = torch.device(device)
    return Sam3PredictorCompat(wrapper, resolution=resolution, device=dev, text_prompt=text_prompt)
