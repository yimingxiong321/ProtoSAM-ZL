"""Calibrate and evaluate a support-conditioned target-presence gate.

The default protocol is leave-one-volume-out (LOVO).  For each held-out query
volume, the threshold is selected using all other volumes and is then frozen
before evaluating the held-out volume.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from util.presence_gate_calibration import (
    evaluate_fixed_threshold,
    gate_metrics,
    sensitivity_constrained_threshold,
)


TRUE_VALUES = {"1", "true", "yes", "y", "t"}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


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


def _metric_columns(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_lovo(
    rows,
    score_column="topk_mean_probability",
    group_column="scan_id",
    target_sensitivity=0.95,
    area_column="final_predicted_area_ratio",
    hard_negative_column="hard_negative_3",
):
    required = {"gt_present", score_column, group_column, area_column}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    groups = sorted({row[group_column] for row in rows}, key=str)
    if len(groups) < 2:
        raise ValueError("LOVO evaluation requires at least two distinct groups")

    fold_rows = []
    gated_rows = []
    pooled_labels = []
    pooled_predictions = []
    pooled_areas = []
    pooled_hard_negatives = []

    for held_out in groups:
        calibration = [row for row in rows if row[group_column] != held_out]
        test = [row for row in rows if row[group_column] == held_out]
        calibration_labels = np.asarray(
            [_as_bool(row["gt_present"]) for row in calibration], dtype=np.bool_
        )
        calibration_scores = np.asarray(
            [float(row[score_column]) for row in calibration], dtype=np.float64
        )
        threshold = sensitivity_constrained_threshold(
            calibration_labels,
            calibration_scores,
            target_sensitivity=target_sensitivity,
        )

        calibration_areas = np.asarray(
            [float(row[area_column]) for row in calibration], dtype=np.float64
        )
        calibration_hard = np.asarray(
            [_as_bool(row.get(hard_negative_column, False)) for row in calibration],
            dtype=np.bool_,
        )
        calibration_metrics = evaluate_fixed_threshold(
            calibration_labels,
            calibration_scores,
            threshold,
            predicted_area_ratio=calibration_areas,
            hard_negative=calibration_hard,
        )

        test_labels = np.asarray(
            [_as_bool(row["gt_present"]) for row in test], dtype=np.bool_
        )
        test_scores = np.asarray(
            [float(row[score_column]) for row in test], dtype=np.float64
        )
        test_areas = np.asarray(
            [float(row[area_column]) for row in test], dtype=np.float64
        )
        test_hard = np.asarray(
            [_as_bool(row.get(hard_negative_column, False)) for row in test],
            dtype=np.bool_,
        )
        test_predictions = test_scores >= threshold
        test_metrics = gate_metrics(
            test_labels,
            test_predictions,
            predicted_area_ratio=test_areas,
            hard_negative=test_hard,
        )
        fold_rows.append({
            "held_out_group": held_out,
            "threshold": threshold,
            **_metric_columns("calibration", calibration_metrics),
            **_metric_columns("test", test_metrics),
        })

        for source, prediction in zip(test, test_predictions):
            gated = dict(source)
            gated["calibration_held_out_group"] = held_out
            gated["presence_score_column"] = score_column
            gated["presence_threshold"] = threshold
            gated["gate_present"] = bool(prediction)
            gated["gated_predicted_area_ratio"] = (
                float(source[area_column]) if prediction else 0.0
            )
            gated_rows.append(gated)
        pooled_labels.extend(test_labels.tolist())
        pooled_predictions.extend(test_predictions.tolist())
        pooled_areas.extend(test_areas.tolist())
        pooled_hard_negatives.extend(test_hard.tolist())

    pooled_metrics = gate_metrics(
        pooled_labels,
        pooled_predictions,
        predicted_area_ratio=pooled_areas,
        hard_negative=pooled_hard_negatives,
    )
    all_labels = np.asarray(
        [_as_bool(row["gt_present"]) for row in rows], dtype=np.bool_
    )
    all_scores = np.asarray(
        [float(row[score_column]) for row in rows], dtype=np.float64
    )
    all_areas = np.asarray(
        [float(row[area_column]) for row in rows], dtype=np.float64
    )
    all_hard = np.asarray(
        [_as_bool(row.get(hard_negative_column, False)) for row in rows],
        dtype=np.bool_,
    )
    full_validation_threshold = sensitivity_constrained_threshold(
        all_labels,
        all_scores,
        target_sensitivity=target_sensitivity,
    )
    full_validation_metrics = evaluate_fixed_threshold(
        all_labels,
        all_scores,
        full_validation_threshold,
        predicted_area_ratio=all_areas,
        hard_negative=all_hard,
    )
    thresholds = np.asarray([row["threshold"] for row in fold_rows], dtype=np.float64)
    macro_keys = (
        "test_sensitivity",
        "test_specificity",
        "test_hard_negative_specificity",
        "test_negative_fp_area_reduction",
    )
    macro = {}
    for key in macro_keys:
        values = np.asarray([row[key] for row in fold_rows], dtype=np.float64)
        macro[f"{key}_mean"] = float(np.nanmean(values))
        macro[f"{key}_std"] = float(np.nanstd(values))
    summary = {
        "protocol": "leave-one-volume-out",
        "score_column": score_column,
        "group_column": group_column,
        "target_calibration_sensitivity": float(target_sensitivity),
        "n_groups": len(groups),
        "threshold_mean": float(thresholds.mean()),
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "pooled_held_out": pooled_metrics,
        "macro_volume": macro,
        "full_validation_fit": {
            "threshold_for_future_locked_test": full_validation_threshold,
            **full_validation_metrics,
        },
    }
    return fold_rows, gated_rows, summary


def run_fixed_threshold(
    rows,
    threshold,
    score_column="topk_mean_probability",
    area_column="final_predicted_area_ratio",
    hard_negative_column="hard_negative_3",
):
    required = {"gt_present", score_column, area_column}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    labels = np.asarray([_as_bool(row["gt_present"]) for row in rows], dtype=np.bool_)
    scores = np.asarray([float(row[score_column]) for row in rows], dtype=np.float64)
    areas = np.asarray([float(row[area_column]) for row in rows], dtype=np.float64)
    hard = np.asarray(
        [_as_bool(row.get(hard_negative_column, False)) for row in rows],
        dtype=np.bool_,
    )
    predictions = scores >= float(threshold)
    metrics = gate_metrics(
        labels,
        predictions,
        predicted_area_ratio=areas,
        hard_negative=hard,
    )
    gated_rows = []
    for source, prediction in zip(rows, predictions):
        gated = dict(source)
        gated["presence_score_column"] = score_column
        gated["presence_threshold"] = float(threshold)
        gated["gate_present"] = bool(prediction)
        gated["gated_predicted_area_ratio"] = (
            float(source[area_column]) if prediction else 0.0
        )
        gated_rows.append(gated)
    return gated_rows, {
        "protocol": "fixed-threshold-locked-test",
        "score_column": score_column,
        "threshold": float(threshold),
        "test": metrics,
    }


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(fold_rows, summary):
    print("\nLOVO presence-gate calibration")
    print(
        "held_out  threshold  cal_sens  test_sens  test_spec  hard3_spec  fp_area_reduction"
    )
    for row in fold_rows:
        print(
            f"{str(row['held_out_group']):>8}  {row['threshold']:.6f}  "
            f"{row['calibration_sensitivity']:.4f}    "
            f"{row['test_sensitivity']:.4f}     "
            f"{row['test_specificity']:.4f}     "
            f"{row['test_hard_negative_specificity']:.4f}      "
            f"{row['test_negative_fp_area_reduction']:.4f}"
        )
    pooled = summary["pooled_held_out"]
    print("\nPooled held-out result")
    for key in (
        "n_positive",
        "n_negative",
        "sensitivity",
        "specificity",
        "hard_negative_specificity",
        "mean_negative_fp_area_before",
        "mean_negative_fp_area_after",
        "negative_fp_area_reduction",
    ):
        print(f"{key}: {pooled.get(key)}")
    fitted = summary["full_validation_fit"]
    print("\nThreshold fitted on the complete validation CSV")
    print(
        "threshold_for_future_locked_test: "
        f"{fitted['threshold_for_future_locked_test']}"
    )
    print(f"validation_sensitivity: {fitted['sensitivity']}")
    print(f"validation_specificity: {fitted['specificity']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--score", default="topk_mean_probability")
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--group-column", default="scan_id")
    parser.add_argument("--area-column", default="final_predicted_area_ratio")
    parser.add_argument("--hard-negative-column", default="hard_negative_3")
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Skip calibration and evaluate this locked validation-set threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("presence_gate_eval"),
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fixed_threshold is None:
        fold_rows, gated_rows, summary = run_lovo(
            rows,
            score_column=args.score,
            group_column=args.group_column,
            target_sensitivity=args.target_sensitivity,
            area_column=args.area_column,
            hard_negative_column=args.hard_negative_column,
        )
        _write_csv(args.output_dir / "lovo_per_volume.csv", fold_rows)
        _write_csv(args.output_dir / "lovo_gated_slices.csv", gated_rows)
        _print_summary(fold_rows, summary)
    else:
        gated_rows, summary = run_fixed_threshold(
            rows,
            threshold=args.fixed_threshold,
            score_column=args.score,
            area_column=args.area_column,
            hard_negative_column=args.hard_negative_column,
        )
        _write_csv(args.output_dir / "fixed_threshold_gated_slices.csv", gated_rows)
        print("\nFixed locked-threshold test")
        print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
