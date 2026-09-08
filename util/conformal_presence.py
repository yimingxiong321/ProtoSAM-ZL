"""Split-conformal utilities for target-presence evidence scores."""

import math

import numpy as np


def upper_tail_p_values(null_scores, test_scores):
    """Conservative empirical p-values for scores where larger means present.

    The null hypothesis is target absence.  With an independent exchangeable
    negative calibration set, ``p <= alpha`` rejects target absence at level
    ``alpha``.  The plus-one correction prevents zero p-values.
    """
    null_scores = np.asarray(null_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    if null_scores.ndim != 1 or null_scores.size == 0:
        raise ValueError("null_scores must be a non-empty 1-D array")
    if not np.isfinite(null_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("conformal scores must be finite")
    exceedances = (null_scores[:, None] >= test_scores.reshape(1, -1)).sum(axis=0)
    return (1.0 + exceedances) / float(null_scores.size + 1)


def sensitivity_calibrated_alpha(positive_p_values, target_sensitivity):
    """Smallest alpha whose empirical positive sensitivity reaches target."""
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    positive_p_values = np.sort(np.asarray(positive_p_values, dtype=np.float64))
    if positive_p_values.ndim != 1 or positive_p_values.size == 0:
        raise ValueError("positive_p_values must be a non-empty 1-D array")
    required = int(math.ceil(target_sensitivity * positive_p_values.size))
    return float(positive_p_values[required - 1])


def combine_conformal_p_values(first, second, method):
    """Combine two null p-values using dependence-safe elementary rules.

    ``both`` is an intersection rule: both evidence sources must reject the
    absence null. ``either_bonferroni`` allows either source to reject while
    applying the Bonferroni correction.  Both remain valid without assuming
    independent score streams when their inputs are valid p-values.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("p-value arrays must have the same shape")
    if method == "both":
        return np.maximum(first, second)
    if method == "either_bonferroni":
        return np.minimum(1.0, 2.0 * np.minimum(first, second))
    raise ValueError(f"unsupported p-value combination: {method}")
