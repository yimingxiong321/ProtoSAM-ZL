"""Utilities for turning coarse foreground probabilities into SAM mask prompts."""

import numpy as np

try:
    import cv2
except ImportError:  # Minimal local test environments may omit OpenCV.
    cv2 = None


SOFT_PROMPT_ALPHAS = {
    "soft_prompt_a025": 0.25,
    "soft_prompt_a05": 0.5,
    "soft_prompt_a1": 1.0,
}


def build_soft_prompt_mask_input(fg_prob, mode, size=(256, 256), eps=1e-4):
    """Build the low-resolution logit prompt expected by ``SamPredictor``.

    ``soft_prompt`` retains the legacy mapping for reproducibility.  The alpha
    variants use calibrated log-odds, so alpha has a direct interpretation as
    the strength of the coarse-mask prior.
    """
    # Pipeline modifiers such as ``_mnn`` change point selection, not the mask
    # calibration. Resolve them to the underlying soft-prompt mode so future
    # combinations do not require duplicate alpha-table entries.
    base_mode = mode[:-4] if mode.endswith("_mnn") else mode

    fg_prob = np.asarray(fg_prob, dtype=np.float32)
    if cv2 is not None:
        resized = cv2.resize(fg_prob, size, interpolation=cv2.INTER_LINEAR)
    else:
        import torch
        from torch.nn import functional as F

        resized = F.interpolate(
            torch.from_numpy(fg_prob)[None, None],
            size=(size[1], size[0]),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

    if base_mode == "soft_prompt":
        logits = (resized - 0.5) * 20.0
    elif base_mode in SOFT_PROMPT_ALPHAS:
        probability = np.clip(resized, eps, 1.0 - eps)
        log_odds = np.log(probability) - np.log1p(-probability)
        logits = SOFT_PROMPT_ALPHAS[base_mode] * log_odds
    else:
        supported = ", ".join(["soft_prompt", *SOFT_PROMPT_ALPHAS])
        raise ValueError(f"Unsupported soft-prompt mode {mode!r}; expected one of: {supported}")

    return logits.astype(np.float32, copy=False)[None, ...]
