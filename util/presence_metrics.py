"""Small dependency-free metrics for target-presence score evaluation."""

import math

import numpy as np


def binary_auroc(labels, scores):
    """Rank-based AUROC with average ranks for ties."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[order[end]] == scores[order[start]]:
            end += 1
        # Ranks are one-indexed for the Mann-Whitney identity.
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def binary_average_precision(labels, scores):
    """Average precision (step-wise area under the precision-recall curve)."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, sorted_labels.size + 1)
    return float(precision[sorted_labels == 1].sum() / positives)


def ranking_metrics(rows, score_names, negative_filter=None):
    """Compute AUROC/AUPRC for positives versus selected negative rows."""
    selected = []
    for row in rows:
        if bool(row["gt_present"]):
            selected.append(row)
        elif negative_filter is None or negative_filter(row):
            selected.append(row)
    labels = [int(bool(row["gt_present"])) for row in selected]
    result = {}
    for score_name in score_names:
        scores = [float(row[score_name]) for row in selected]
        result[score_name] = {
            "auroc": binary_auroc(labels, scores),
            "auprc": binary_average_precision(labels, scores),
            "n": len(selected),
            "n_positive": int(sum(labels)),
            "n_negative": int(len(labels) - sum(labels)),
        }
    return result


def mask_presence_metrics(rows):
    positives = [row for row in rows if bool(row["gt_present"])]
    negatives = [row for row in rows if not bool(row["gt_present"])]
    sensitivity = (
        sum(bool(row["pred_present"]) for row in positives) / len(positives)
        if positives else float("nan")
    )
    specificity = (
        sum(not bool(row["pred_present"]) for row in negatives) / len(negatives)
        if negatives else float("nan")
    )
    mean_fp_area_ratio = (
        float(np.mean([float(row["final_predicted_area_ratio"]) for row in negatives]))
        if negatives else float("nan")
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mean_negative_fp_area_ratio": mean_fp_area_ratio,
        "n_positive": len(positives),
        "n_negative": len(negatives),
    }


def is_finite_number(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
