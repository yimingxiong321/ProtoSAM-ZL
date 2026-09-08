"""Evaluate locked split-conformal target-presence gates across folds.

Use one completed fold as an independent calibration set and one or more
different folds as locked tests.  Target-absent calibration queries form the
null distribution.  Target-present calibration queries are used only to
select the optional high-sensitivity operating alpha.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from util.conformal_presence import (
    combine_conformal_p_values,
    sensitivity_calibrated_alpha,
    upper_tail_p_values,
)
from util.presence_gate_calibration import gate_metrics


TRUE_VALUES = {"1", "true", "yes", "y", "t"}
DEFAULT_SCORES = ("topk_mean_probability", "fg_bg_likelihood_ratio")


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


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


def _metrics(rows, p_values, alpha, area_column, hard_column):
    labels = np.asarray([_as_bool(row["gt_present"]) for row in rows])
    areas = np.asarray([float(row[area_column]) for row in rows])
    hard = np.asarray([_as_bool(row.get(hard_column, False)) for row in rows])
    return gate_metrics(
        labels,
        np.asarray(p_values) <= float(alpha),
        predicted_area_ratio=areas,
        hard_negative=hard,
    )


def evaluate_conformal(
    calibration_rows,
    test_sets,
    score_names=DEFAULT_SCORES,
    fixed_alpha=0.05,
    target_sensitivity=0.975,
    area_column="final_predicted_area_ratio",
    hard_column="hard_negative_3",
):
    required = {"gt_present", area_column, *score_names}
    missing = sorted(required - set(calibration_rows[0]))
    if missing:
        raise ValueError(f"calibration CSV missing columns: {', '.join(missing)}")
    calibration_labels = np.asarray(
        [_as_bool(row["gt_present"]) for row in calibration_rows]
    )
    if not calibration_labels.any() or calibration_labels.all():
        raise ValueError("calibration CSV must contain positive and negative queries")

    calibration_p = {}
    test_p = {name: {} for name, _ in test_sets}
    for score_name in score_names:
        calibration_scores = np.asarray(
            [float(row[score_name]) for row in calibration_rows]
        )
        null_scores = calibration_scores[~calibration_labels]
        calibration_p[score_name] = upper_tail_p_values(
            null_scores, calibration_scores
        )
        for test_name, rows in test_sets:
            scores = np.asarray([float(row[score_name]) for row in rows])
            test_p[test_name][score_name] = upper_tail_p_values(null_scores, scores)

    streams = {name: calibration_p[name] for name in score_names}
    if len(score_names) == 2:
        first, second = score_names
        streams["topk_lr_both"] = combine_conformal_p_values(
            calibration_p[first], calibration_p[second], "both"
        )
        streams["topk_lr_either_bonferroni"] = combine_conformal_p_values(
            calibration_p[first], calibration_p[second], "either_bonferroni"
        )
        for test_name, _ in test_sets:
            test_p[test_name]["topk_lr_both"] = combine_conformal_p_values(
                test_p[test_name][first], test_p[test_name][second], "both"
            )
            test_p[test_name]["topk_lr_either_bonferroni"] = (
                combine_conformal_p_values(
                    test_p[test_name][first],
                    test_p[test_name][second],
                    "either_bonferroni",
                )
            )

    summaries = []
    gated_rows = []
    details = {}
    for stream_name, calibration_values in streams.items():
        calibrated_alpha = sensitivity_calibrated_alpha(
            calibration_values[calibration_labels], target_sensitivity
        )
        details[stream_name] = {
            "fixed_alpha": float(fixed_alpha),
            "sensitivity_calibrated_alpha": calibrated_alpha,
            "calibration_fixed": _metrics(
                calibration_rows,
                calibration_values,
                fixed_alpha,
                area_column,
                hard_column,
            ),
            "calibration_sensitivity_locked": _metrics(
                calibration_rows,
                calibration_values,
                calibrated_alpha,
                area_column,
                hard_column,
            ),
            "tests": {},
        }
        pooled_rows = []
        pooled_values = []
        for test_name, rows in test_sets:
            values = test_p[test_name][stream_name]
            pooled_rows.extend(rows)
            pooled_values.extend(np.asarray(values).tolist())
            fixed_metrics = _metrics(
                rows, values, fixed_alpha, area_column, hard_column
            )
            locked_metrics = _metrics(
                rows, values, calibrated_alpha, area_column, hard_column
            )
            details[stream_name]["tests"][test_name] = {
                "fixed_alpha": fixed_metrics,
                "sensitivity_locked_alpha": locked_metrics,
            }
            summaries.append({
                "score": stream_name,
                "test": test_name,
                "operating_point": "fixed_alpha",
                "alpha": fixed_alpha,
                **fixed_metrics,
            })
            summaries.append({
                "score": stream_name,
                "test": test_name,
                "operating_point": "sensitivity_locked",
                "alpha": calibrated_alpha,
                **locked_metrics,
            })
            for row, p_value in zip(rows, values):
                output = dict(row)
                output["test_set"] = test_name
                output["conformal_score"] = stream_name
                output["absence_null_p_value"] = float(p_value)
                output["fixed_alpha"] = float(fixed_alpha)
                output["fixed_alpha_present"] = bool(p_value <= fixed_alpha)
                output["sensitivity_locked_alpha"] = calibrated_alpha
                output["sensitivity_locked_present"] = bool(
                    p_value <= calibrated_alpha
                )
                gated_rows.append(output)
        pooled_values = np.asarray(pooled_values, dtype=np.float64)
        pooled_fixed = _metrics(
            pooled_rows, pooled_values, fixed_alpha, area_column, hard_column
        )
        pooled_locked = _metrics(
            pooled_rows,
            pooled_values,
            calibrated_alpha,
            area_column,
            hard_column,
        )
        details[stream_name]["pooled_tests"] = {
            "fixed_alpha": pooled_fixed,
            "sensitivity_locked_alpha": pooled_locked,
        }
        summaries.append({
            "score": stream_name,
            "test": "pooled",
            "operating_point": "fixed_alpha",
            "alpha": fixed_alpha,
            **pooled_fixed,
        })
        summaries.append({
            "score": stream_name,
            "test": "pooled",
            "operating_point": "sensitivity_locked",
            "alpha": calibrated_alpha,
            **pooled_locked,
        })
    return summaries, gated_rows, details


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--test-csv", type=Path, action="append")
    parser.add_argument(
        "--source-csv",
        type=Path,
        help="Patient-disjoint mode: split one fold CSV by group instead.",
    )
    parser.add_argument(
        "--calibration-group",
        action="append",
        help="Group assigned to calibration when --source-csv is used.",
    )
    parser.add_argument("--group-column", default="scan_id")
    parser.add_argument("--score", action="append", dest="scores")
    parser.add_argument("--fixed-alpha", type=float, default=0.05)
    parser.add_argument("--target-sensitivity", type=float, default=0.975)
    parser.add_argument("--area-column", default="final_predicted_area_ratio")
    parser.add_argument("--hard-negative-column", default="hard_negative_3")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("presence_conformal_eval")
    )
    args = parser.parse_args()
    if not 0.0 < args.fixed_alpha < 1.0:
        raise ValueError("fixed-alpha must be in (0, 1)")

    if args.source_csv is not None:
        if args.calibration_csv is not None or args.test_csv:
            raise ValueError(
                "use either --source-csv or --calibration-csv/--test-csv"
            )
        if not args.calibration_group:
            raise ValueError("--source-csv requires --calibration-group")
        source_rows = _read_csv(args.source_csv)
        calibration_groups = {str(value) for value in args.calibration_group}
        all_groups = {str(row[args.group_column]) for row in source_rows}
        unknown = calibration_groups - all_groups
        if unknown:
            raise ValueError(f"unknown calibration groups: {sorted(unknown)}")
        calibration_rows = [
            row
            for row in source_rows
            if str(row[args.group_column]) in calibration_groups
        ]
        test_groups = sorted(all_groups - calibration_groups)
        if not test_groups:
            raise ValueError("at least one group must remain for locked testing")
        test_sets = [
            (
                f"group{group}",
                [row for row in source_rows if str(row[args.group_column]) == group],
            )
            for group in test_groups
        ]
    else:
        if args.calibration_csv is None or not args.test_csv:
            raise ValueError(
                "provide --source-csv/--calibration-group or "
                "--calibration-csv/--test-csv"
            )
        calibration_rows = _read_csv(args.calibration_csv)
        test_sets = []
        for index, path in enumerate(args.test_csv):
            rows = _read_csv(path)
            fold = str(rows[0].get("fold", "")).strip()
            test_name = f"fold{fold}" if fold else f"test{index}"
            test_sets.append((test_name, rows))
    summaries, gated_rows, details = evaluate_conformal(
        calibration_rows,
        test_sets,
        score_names=tuple(args.scores or DEFAULT_SCORES),
        fixed_alpha=args.fixed_alpha,
        target_sensitivity=args.target_sensitivity,
        area_column=args.area_column,
        hard_column=args.hard_negative_column,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "conformal_summary.csv", summaries)
    _write_csv(args.output_dir / "conformal_gated_slices.csv", gated_rows)
    with (args.output_dir / "details.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(details), handle, ensure_ascii=False, indent=2)

    print("\nLocked split-conformal evaluation")
    print("score                     test       op                 alpha  sens   spec   hardSpec")
    for row in summaries:
        print(
            f"{row['score']:<25} {row['test']:<10} "
            f"{row['operating_point']:<18} {row['alpha']:.4f} "
            f"{row['sensitivity']:.4f} {row['specificity']:.4f} "
            f"{row['hard_negative_specificity']:.4f}"
        )
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
