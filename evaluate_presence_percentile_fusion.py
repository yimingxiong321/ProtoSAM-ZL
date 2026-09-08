"""Patient-wise LOVO evaluation of null-percentile presence-score fusion.

The LR and PatchCore scores are independently mapped to empirical percentiles
using only target-absent slices from the calibration patients.  Fusion and the
high-sensitivity threshold are then fitted without using the held-out patient.

This is empirical rank fusion, not by itself a conformal p-value.  A strict
conformal claim requires an additional independent calibration stage for the
final fused score.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from util.presence_gate_calibration import (
    gate_metrics,
    sensitivity_constrained_threshold,
)
from util.presence_metrics import binary_average_precision, binary_auroc


TRUE_VALUES = {"1", "true", "yes", "y", "t"}
DEFAULT_LR_SCORE = "fg_bg_likelihood_ratio"
DEFAULT_PATCHCORE_SCORE = "patchcore_bidirectional_score"
PERCENTILE_NAMES = (
    "lr_percentile",
    "patchcore_percentile",
    "percentile_avg",
    "percentile_lr60",
    "percentile_min",
)
EVALUATION_NAMES = ("lr_raw", "patchcore_raw", *PERCENTILE_NAMES)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def empirical_null_percentile(null_scores, scores):
    """Return the fraction of null scores strictly below each score.

    This implements P(x) = N^-1 sum_i 1[S_null_i < x].  The transform is
    fitted separately inside each LOVO split, so held-out patients never
    contribute to their own reference distribution.
    """
    null_scores = np.asarray(null_scores, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if null_scores.ndim != 1 or null_scores.size == 0:
        raise ValueError("null_scores must be a non-empty 1D array")
    if not np.isfinite(null_scores).all() or not np.isfinite(scores).all():
        raise ValueError("null and query scores must all be finite")
    ordered = np.sort(null_scores)
    return np.searchsorted(ordered, scores, side="left") / ordered.size


def fuse_percentiles(lr_percentile, patchcore_percentile):
    lr = np.asarray(lr_percentile, dtype=np.float64)
    pc = np.asarray(patchcore_percentile, dtype=np.float64)
    if lr.shape != pc.shape:
        raise ValueError("LR and PatchCore percentiles must have the same shape")
    return {
        "lr_percentile": lr,
        "patchcore_percentile": pc,
        "percentile_avg": 0.5 * lr + 0.5 * pc,
        "percentile_lr60": 0.6 * lr + 0.4 * pc,
        "percentile_min": np.minimum(lr, pc),
    }


def _ranking(labels, scores, hard_negative):
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    hard_negative = np.asarray(hard_negative, dtype=np.bool_)
    hard_selection = labels | hard_negative
    return {
        "auroc": binary_auroc(labels.astype(np.int64), scores),
        "auprc": binary_average_precision(labels.astype(np.int64), scores),
        "hard3_auroc": binary_auroc(
            labels[hard_selection].astype(np.int64), scores[hard_selection]
        ),
        "hard3_auprc": binary_average_precision(
            labels[hard_selection].astype(np.int64), scores[hard_selection]
        ),
    }


def _metric_columns(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_percentile_fusion_lovo(
    rows,
    lr_score=DEFAULT_LR_SCORE,
    patchcore_score=DEFAULT_PATCHCORE_SCORE,
    group_column="scan_id",
    target_sensitivity=0.975,
    area_column="final_predicted_area_ratio",
    hard_negative_column="hard_negative_3",
):
    if not rows:
        raise ValueError("input rows are empty")
    required = {
        "gt_present",
        lr_score,
        patchcore_score,
        group_column,
        area_column,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    groups = sorted({row[group_column] for row in rows}, key=str)
    if len(groups) < 2:
        raise ValueError("LOVO evaluation requires at least two groups")

    split_rows = []
    gated_rows = []
    pooled = {
        name: {"labels": [], "scores": [], "predictions": [], "areas": [], "hard": []}
        for name in EVALUATION_NAMES
    }

    for held_out in groups:
        calibration = [row for row in rows if row[group_column] != held_out]
        test = [row for row in rows if row[group_column] == held_out]
        calibration_labels = np.asarray(
            [_as_bool(row["gt_present"]) for row in calibration], dtype=np.bool_
        )
        if not calibration_labels.any() or calibration_labels.all():
            raise ValueError(
                f"calibration split for {held_out!r} needs positives and negatives"
            )
        calibration_lr = np.asarray(
            [float(row[lr_score]) for row in calibration], dtype=np.float64
        )
        calibration_pc = np.asarray(
            [float(row[patchcore_score]) for row in calibration], dtype=np.float64
        )
        null_lr = calibration_lr[~calibration_labels]
        null_pc = calibration_pc[~calibration_labels]
        calibration_streams = {
            "lr_raw": calibration_lr,
            "patchcore_raw": calibration_pc,
            **fuse_percentiles(
                empirical_null_percentile(null_lr, calibration_lr),
                empirical_null_percentile(null_pc, calibration_pc),
            ),
        }

        test_labels = np.asarray(
            [_as_bool(row["gt_present"]) for row in test], dtype=np.bool_
        )
        test_lr = np.asarray([float(row[lr_score]) for row in test], dtype=np.float64)
        test_pc = np.asarray(
            [float(row[patchcore_score]) for row in test], dtype=np.float64
        )
        test_lr_percentile = empirical_null_percentile(null_lr, test_lr)
        test_pc_percentile = empirical_null_percentile(null_pc, test_pc)
        test_streams = {
            "lr_raw": test_lr,
            "patchcore_raw": test_pc,
            **fuse_percentiles(test_lr_percentile, test_pc_percentile),
        }
        calibration_areas = np.asarray(
            [float(row[area_column]) for row in calibration], dtype=np.float64
        )
        calibration_hard = np.asarray(
            [_as_bool(row.get(hard_negative_column, False)) for row in calibration],
            dtype=np.bool_,
        )
        test_areas = np.asarray(
            [float(row[area_column]) for row in test], dtype=np.float64
        )
        test_hard = np.asarray(
            [_as_bool(row.get(hard_negative_column, False)) for row in test],
            dtype=np.bool_,
        )

        for fusion_name in EVALUATION_NAMES:
            calibration_scores = calibration_streams[fusion_name]
            test_scores = test_streams[fusion_name]
            threshold = sensitivity_constrained_threshold(
                calibration_labels,
                calibration_scores,
                target_sensitivity=target_sensitivity,
            )
            calibration_predictions = calibration_scores >= threshold
            test_predictions = test_scores >= threshold
            calibration_metrics = gate_metrics(
                calibration_labels,
                calibration_predictions,
                predicted_area_ratio=calibration_areas,
                hard_negative=calibration_hard,
            )
            test_metrics = gate_metrics(
                test_labels,
                test_predictions,
                predicted_area_ratio=test_areas,
                hard_negative=test_hard,
            )
            split_rows.append({
                "fusion": fusion_name,
                "held_out_group": held_out,
                "threshold": threshold,
                "calibration_null_count": int(null_lr.size),
                **_metric_columns("calibration", calibration_metrics),
                **_metric_columns("test", test_metrics),
            })

            for index, (source, prediction) in enumerate(zip(test, test_predictions)):
                output = dict(source)
                output.update({
                    "held_out_group": held_out,
                    "fusion": fusion_name,
                    "lr_null_percentile": float(test_lr_percentile[index]),
                    "patchcore_null_percentile": float(test_pc_percentile[index]),
                    "fused_presence_score": float(test_scores[index]),
                    "presence_threshold": float(threshold),
                    "gate_present": bool(prediction),
                    "gated_predicted_area_ratio": (
                        float(source[area_column]) if prediction else 0.0
                    ),
                })
                gated_rows.append(output)

            target = pooled[fusion_name]
            target["labels"].extend(test_labels.tolist())
            target["scores"].extend(test_scores.tolist())
            target["predictions"].extend(test_predictions.tolist())
            target["areas"].extend(test_areas.tolist())
            target["hard"].extend(test_hard.tolist())

    summary_rows = []
    for fusion_name in EVALUATION_NAMES:
        values = pooled[fusion_name]
        metrics = gate_metrics(
            values["labels"],
            values["predictions"],
            predicted_area_ratio=values["areas"],
            hard_negative=values["hard"],
        )
        ranking = _ranking(values["labels"], values["scores"], values["hard"])
        thresholds = [
            float(row["threshold"])
            for row in split_rows
            if row["fusion"] == fusion_name
        ]
        summary_rows.append({
            "fusion": fusion_name,
            "n_groups": len(groups),
            "target_calibration_sensitivity": float(target_sensitivity),
            "threshold_mean": float(np.mean(thresholds)),
            "threshold_min": float(np.min(thresholds)),
            "threshold_max": float(np.max(thresholds)),
            **ranking,
            **metrics,
        })
    return split_rows, gated_rows, summary_rows


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--lr-score", default=DEFAULT_LR_SCORE)
    parser.add_argument("--patchcore-score", default=DEFAULT_PATCHCORE_SCORE)
    parser.add_argument("--group-column", default="scan_id")
    parser.add_argument("--target-sensitivity", type=float, default=0.975)
    parser.add_argument("--area-column", default="final_predicted_area_ratio")
    parser.add_argument("--hard-negative-column", default="hard_negative_3")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("presence_percentile_fusion")
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    split_rows, gated_rows, summary_rows = run_percentile_fusion_lovo(
        rows,
        lr_score=args.lr_score,
        patchcore_score=args.patchcore_score,
        group_column=args.group_column,
        target_sensitivity=args.target_sensitivity,
        area_column=args.area_column,
        hard_negative_column=args.hard_negative_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_volume.csv", split_rows)
    _write_csv(args.output_dir / "gated_slices.csv", gated_rows)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary_rows), handle, ensure_ascii=False, indent=2)

    print("\nNull-percentile fusion LOVO")
    print("fusion                 AUROC  hard3  sens   spec   hardSpec FP-red")
    for row in summary_rows:
        print(
            f"{row['fusion']:<22} {row['auroc']:.4f} "
            f"{row['hard3_auroc']:.4f} {row['sensitivity']:.4f} "
            f"{row['specificity']:.4f} "
            f"{row['hard_negative_specificity']:.4f} "
            f"{row['negative_fp_area_reduction']:.4f}"
        )
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
