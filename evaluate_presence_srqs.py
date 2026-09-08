"""Compare ProtoSAM probability/LR scores with MedVeriSeg SRQS evidence.

Run this after ``presence_score`` inference has produced a CSV.  Thresholds
are calibrated leave-one-volume-out at the requested sensitivity; no test
labels are used to tune a held-out volume's threshold.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_presence_gate import run_lovo
from util.presence_metrics import ranking_metrics


DEFAULT_SCORES = (
    "topk_mean_probability",
    "probability_srqs_strength",
    "probability_srqs_compactness",
    "probability_srqs_purity",
    "probability_srqs_score",
    "fg_bg_likelihood_ratio",
    "likelihood_srqs_strength",
    "likelihood_srqs_compactness",
    "likelihood_srqs_purity",
    "likelihood_srqs_score",
    "patchcore_candidate_mean_similarity",
    "patchcore_candidate_worst_similarity",
    "patchcore_support_coverage",
    "patchcore_bidirectional_score",
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare_scores(
    rows,
    score_names=DEFAULT_SCORES,
    target_sensitivity=0.975,
    group_column="scan_id",
    area_column="final_predicted_area_ratio",
    hard_negative_column="hard_negative_3",
):
    if not rows:
        raise ValueError("input CSV is empty")
    missing = [score for score in score_names if score not in rows[0]]
    if missing:
        raise ValueError(
            "CSV is missing SRQS columns; rerun presence_score with the updated code: "
            + ", ".join(missing)
        )

    normalized_rows = []
    for source in rows:
        row = dict(source)
        row["gt_present"] = _as_bool(row["gt_present"])
        row[hard_negative_column] = _as_bool(
            row.get(hard_negative_column, False)
        )
        normalized_rows.append(row)

    ranking_all = ranking_metrics(normalized_rows, score_names)
    ranking_hard = ranking_metrics(
        normalized_rows,
        score_names,
        negative_filter=lambda row: bool(row.get(hard_negative_column, False)),
    )
    comparison = []
    details = {}
    for score_name in score_names:
        fold_rows, _, summary = run_lovo(
            normalized_rows,
            score_column=score_name,
            group_column=group_column,
            target_sensitivity=target_sensitivity,
            area_column=area_column,
            hard_negative_column=hard_negative_column,
        )
        pooled = summary["pooled_held_out"]
        fitted = summary["full_validation_fit"]
        comparison.append({
            "score": score_name,
            "auroc": ranking_all[score_name]["auroc"],
            "auprc": ranking_all[score_name]["auprc"],
            "hard3_auroc": ranking_hard[score_name]["auroc"],
            "hard3_auprc": ranking_hard[score_name]["auprc"],
            "lovo_sensitivity": pooled["sensitivity"],
            "lovo_specificity": pooled["specificity"],
            "lovo_hard3_specificity": pooled["hard_negative_specificity"],
            "negative_fp_area_reduction": pooled["negative_fp_area_reduction"],
            "future_locked_threshold": fitted["threshold_for_future_locked_test"],
        })
        details[score_name] = {
            "folds": fold_rows,
            "summary": summary,
        }
    return comparison, details


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--target-sensitivity", type=float, default=0.975)
    parser.add_argument("--group-column", default="scan_id")
    parser.add_argument("--area-column", default="final_predicted_area_ratio")
    parser.add_argument("--hard-negative-column", default="hard_negative_3")
    parser.add_argument(
        "--scores",
        default=",".join(DEFAULT_SCORES),
        help="Comma-separated score columns to compare.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("presence_srqs_eval")
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    score_names = tuple(x.strip() for x in args.scores.split(",") if x.strip())
    comparison, details = compare_scores(
        rows,
        score_names=score_names,
        target_sensitivity=args.target_sensitivity,
        group_column=args.group_column,
        area_column=args.area_column,
        hard_negative_column=args.hard_negative_column,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "score_comparison.csv", comparison)
    with (args.output_dir / "details.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(details), handle, ensure_ascii=False, indent=2)

    print("\nPresence score comparison")
    print("score                            AUROC   hard3  sens    spec    hardSpec FP-red")
    for row in comparison:
        print(
            f"{row['score']:<32} "
            f"{row['auroc']:.4f}  {row['hard3_auroc']:.4f}  "
            f"{row['lovo_sensitivity']:.4f}  {row['lovo_specificity']:.4f}  "
            f"{row['lovo_hard3_specificity']:.4f}   "
            f"{row['negative_fp_area_reduction']:.4f}"
        )
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
