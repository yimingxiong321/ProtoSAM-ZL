"""Patient-wise LOVO evaluation of split-conformal presence gates."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_presence_conformal import (
    DEFAULT_SCORES,
    TRUE_VALUES,
    evaluate_conformal,
)
from util.presence_gate_calibration import gate_metrics


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_lovo_conformal(
    rows,
    group_column="scan_id",
    score_names=DEFAULT_SCORES,
    fixed_alpha=0.05,
    target_sensitivity=0.975,
    area_column="final_predicted_area_ratio",
    hard_column="hard_negative_3",
):
    groups = sorted({str(row[group_column]) for row in rows})
    if len(groups) < 2:
        raise ValueError("LOVO conformal evaluation needs at least two groups")
    split_rows = []
    gated_rows = []
    for held_out in groups:
        calibration = [row for row in rows if str(row[group_column]) != held_out]
        test = [row for row in rows if str(row[group_column]) == held_out]
        summaries, gated, _ = evaluate_conformal(
            calibration,
            [(f"group{held_out}", test)],
            score_names=score_names,
            fixed_alpha=fixed_alpha,
            target_sensitivity=target_sensitivity,
            area_column=area_column,
            hard_column=hard_column,
        )
        split_rows.extend(row for row in summaries if row["test"] != "pooled")
        for row in gated:
            row["held_out_group"] = held_out
        gated_rows.extend(gated)

    pooled = []
    by_key = defaultdict(list)
    for row in gated_rows:
        by_key[(row["conformal_score"], "fixed_alpha")].append(row)
        by_key[(row["conformal_score"], "sensitivity_locked")].append(row)
    for (score, operating_point), selected in sorted(by_key.items()):
        prediction_column = (
            "fixed_alpha_present"
            if operating_point == "fixed_alpha"
            else "sensitivity_locked_present"
        )
        labels = np.asarray([_as_bool(row["gt_present"]) for row in selected])
        predictions = np.asarray(
            [_as_bool(row[prediction_column]) for row in selected]
        )
        areas = np.asarray([float(row[area_column]) for row in selected])
        hard = np.asarray([_as_bool(row.get(hard_column, False)) for row in selected])
        metrics = gate_metrics(
            labels,
            predictions,
            predicted_area_ratio=areas,
            hard_negative=hard,
        )
        relevant_splits = [
            row
            for row in split_rows
            if row["score"] == score
            and row["operating_point"] == operating_point
        ]
        pooled.append({
            "score": score,
            "operating_point": operating_point,
            "n_groups": len(groups),
            "alpha_mean": float(np.mean([row["alpha"] for row in relevant_splits])),
            "sensitivity_macro": float(
                np.mean([row["sensitivity"] for row in relevant_splits])
            ),
            "specificity_macro": float(
                np.mean([row["specificity"] for row in relevant_splits])
            ),
            **metrics,
        })
    return split_rows, gated_rows, pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--group-column", default="scan_id")
    parser.add_argument("--score", action="append", dest="scores")
    parser.add_argument("--fixed-alpha", type=float, default=0.05)
    parser.add_argument("--target-sensitivity", type=float, default=0.975)
    parser.add_argument("--area-column", default="final_predicted_area_ratio")
    parser.add_argument("--hard-negative-column", default="hard_negative_3")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("presence_conformal_lovo")
    )
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    split_rows, gated_rows, pooled = run_lovo_conformal(
        rows,
        group_column=args.group_column,
        score_names=tuple(args.scores or DEFAULT_SCORES),
        fixed_alpha=args.fixed_alpha,
        target_sensitivity=args.target_sensitivity,
        area_column=args.area_column,
        hard_column=args.hard_negative_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_volume.csv", split_rows)
    _write_csv(args.output_dir / "gated_slices.csv", gated_rows)
    _write_csv(args.output_dir / "pooled_summary.csv", pooled)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pooled, handle, ensure_ascii=False, indent=2)

    print("\nPatient-wise conformal LOVO summary")
    print("score                     op                 sens   spec   hardSpec FP-red")
    for row in pooled:
        print(
            f"{row['score']:<25} {row['operating_point']:<18} "
            f"{row['sensitivity']:.4f} {row['specificity']:.4f} "
            f"{row['hard_negative_specificity']:.4f} "
            f"{row['negative_fp_area_reduction']:.4f}"
        )
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
