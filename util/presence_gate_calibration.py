"""Patient-wise calibration utilities for a score-only presence gate."""

import math

import numpy as np


def sensitivity_constrained_threshold(labels, scores, target_sensitivity=0.95):
    """Return the highest threshold with empirical positive sensitivity >= target.

    A query is declared present when ``score >= threshold``.  Only positive
    calibration samples determine the threshold; negative calibration samples
    are used solely to report achieved specificity.  This prevents an
    accidental optimization of the operating point on the held-out test set.
    """
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if not np.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    positive_scores = np.sort(scores[labels])
    if positive_scores.size == 0:
        raise ValueError("calibration data must contain at least one positive sample")

    allowed_misses = int(
        math.floor((1.0 - float(target_sensitivity)) * positive_scores.size + 1e-12)
    )
    allowed_misses = min(allowed_misses, positive_scores.size - 1)
    return float(positive_scores[allowed_misses])


def gate_metrics(labels, predicted_present, predicted_area_ratio=None, hard_negative=None):
    """Compute presence and negative-FP metrics for fixed gate predictions."""
    labels = np.asarray(labels, dtype=np.bool_)
    predicted_present = np.asarray(predicted_present, dtype=np.bool_)
    if labels.shape != predicted_present.shape:
        raise ValueError("labels and predicted_present must have the same shape")
    positives = labels
    negatives = ~labels

    def _mean_or_nan(values):
        return float(np.mean(values)) if len(values) else float("nan")

    result = {
        "n": int(labels.size),
        "n_positive": int(positives.sum()),
        "n_negative": int(negatives.sum()),
        "sensitivity": _mean_or_nan(predicted_present[positives]),
        "specificity": _mean_or_nan(~predicted_present[negatives]),
        "false_negative_count": int((~predicted_present & positives).sum()),
        "rejected_negative_count": int((~predicted_present & negatives).sum()),
    }

    if hard_negative is not None:
        hard_negative = np.asarray(hard_negative, dtype=np.bool_)
        if hard_negative.shape != labels.shape:
            raise ValueError("hard_negative must have the same shape as labels")
        hard_negative = hard_negative & negatives
        result["n_hard_negative"] = int(hard_negative.sum())
        result["hard_negative_specificity"] = _mean_or_nan(
            ~predicted_present[hard_negative]
        )

    if predicted_area_ratio is not None:
        area = np.asarray(predicted_area_ratio, dtype=np.float64)
        if area.shape != labels.shape:
            raise ValueError("predicted_area_ratio must have the same shape as labels")
        if not np.isfinite(area).all():
            raise ValueError("predicted_area_ratio must all be finite")
        before = _mean_or_nan(area[negatives])
        after = _mean_or_nan(area[negatives] * predicted_present[negatives])
        reduction = (
            1.0 - after / before
            if math.isfinite(before) and before > 0.0
            else float("nan")
        )
        result.update({
            "mean_negative_fp_area_before": before,
            "mean_negative_fp_area_after": after,
            "negative_fp_area_reduction": reduction,
        })
    return result


def evaluate_fixed_threshold(
    labels,
    scores,
    threshold,
    predicted_area_ratio=None,
    hard_negative=None,
):
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= float(threshold)
    return gate_metrics(
        labels,
        predictions,
        predicted_area_ratio=predicted_area_ratio,
        hard_negative=hard_negative,
    )
