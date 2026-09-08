"""Summarize cross-organ P4 continuity experiments.

Expected layout::

    ROOT/{mri,ct}/{organ}/foldN/summary.csv

P4-2 is paired against P4-0 within every fold, so both methods always use the
same support/query data and the same sensitivity-constrained LOVO protocol.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


BASELINE = "p4_0_raw"
CANDIDATE = "p4_2_raw_anchor"
COMPARISON_METHODS = (
    "p4_1_zscore",
    "p4_2_raw_anchor",
    "p4_2_raw_anchor_gap1",
    "p4_3_zscore_anchor",
    "p4_3_zscore_anchor_gap1",
)
METRICS = (
    "sensitivity",
    "specificity",
    "hard_negative_specificity",
    "negative_fp_area_reduction",
    "mean_negative_fp_area_after",
)
DATASET_NAMES = {"mri": "CHAOST2", "ct": "SABS"}


def _float(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def _write_csv(path, rows):
    if not rows:
        return
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


def load_fold_rows(root):
    rows = []
    for path in sorted(root.glob("*/*/fold*/summary.csv")):
        relative = path.relative_to(root)
        modality, organ, fold_name = relative.parts[:3]
        if modality not in DATASET_NAMES or not fold_name.startswith("fold"):
            continue
        fold = int(fold_name[4:])
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for source in csv.DictReader(handle):
                row = {
                    "dataset": DATASET_NAMES[modality],
                    "modality": modality,
                    "organ": organ,
                    "fold": fold,
                    "method": source["method"],
                    "n_groups": int(source["n_groups"]),
                    "n": int(source["n"]),
                    "n_positive": int(source["n_positive"]),
                    "n_negative": int(source["n_negative"]),
                    "n_hard_negative": int(source["n_hard_negative"]),
                }
                for metric in METRICS:
                    row[metric] = _float(source, metric)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No summaries found under {root}/<modality>/<organ>/fold*/summary.csv"
        )
    return rows


def aggregate_methods(fold_rows):
    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[(row["dataset"], row["modality"], row["organ"], row["method"])].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        dataset, modality, organ, method = key
        item = {
            "dataset": dataset,
            "modality": modality,
            "organ": organ,
            "method": method,
            "folds": len(rows),
            "n_slices": sum(row["n"] for row in rows),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.nanmean(values))
            item[f"{metric}_std"] = float(np.nanstd(values))
        output.append(item)
    return output


def paired_candidate_rows(fold_rows, expected_folds=None, candidate_method=CANDIDATE):
    indexed = {
        (row["dataset"], row["modality"], row["organ"], row["fold"], row["method"]): row
        for row in fold_rows
    }
    tasks = sorted({(row["dataset"], row["modality"], row["organ"]) for row in fold_rows})
    output = []
    for dataset, modality, organ in tasks:
        folds = sorted({
            row["fold"] for row in fold_rows
            if (row["dataset"], row["modality"], row["organ"]) == (dataset, modality, organ)
        })
        pairs = []
        for fold in folds:
            baseline = indexed.get((dataset, modality, organ, fold, BASELINE))
            candidate = indexed.get((dataset, modality, organ, fold, candidate_method))
            if baseline is not None and candidate is not None:
                pairs.append((fold, baseline, candidate))
        if not pairs:
            continue

        item = {
            "dataset": dataset,
            "modality": modality,
            "organ": organ,
            "baseline_method": BASELINE,
            "candidate_method": candidate_method,
            "paired_folds": len(pairs),
            "expected_folds": expected_folds if expected_folds is not None else len(folds),
        }
        for metric in METRICS:
            deltas = np.asarray(
                [candidate[metric] - baseline[metric] for _, baseline, candidate in pairs],
                dtype=np.float64,
            )
            item[f"{metric}_delta"] = float(np.nanmean(deltas))
            # Lower residual FP area is better; every other metric is higher-is-better.
            if metric == "mean_negative_fp_area_after":
                item[f"{metric}_wins"] = int(np.sum(deltas < 0))
            else:
                item[f"{metric}_wins"] = int(np.sum(deltas > 0))

        minimum_wins = math.ceil(len(pairs) * 0.6)
        enough_folds = expected_folds is None or len(pairs) == expected_folds
        item["go_sensitivity"] = item["sensitivity_delta"] >= -0.01
        item["go_specificity"] = (
            item["specificity_delta"] > 0
            and item["specificity_wins"] >= minimum_wins
        )
        item["go_hard_negative"] = item["hard_negative_specificity_delta"] >= 0
        item["go_fp_reduction"] = item["negative_fp_area_reduction_delta"] >= 0
        item["go_complete_folds"] = enough_folds
        item["go"] = all((
            item["go_sensitivity"],
            item["go_specificity"],
            item["go_hard_negative"],
            item["go_fp_reduction"],
            item["go_complete_folds"],
        ))
        output.append(item)
    return output


def macro_candidate_row(paired_rows):
    if not paired_rows:
        return {}
    output = {
        "candidate_method": paired_rows[0]["candidate_method"],
        "tasks": len(paired_rows),
        "tasks_go": sum(bool(row["go"]) for row in paired_rows),
        "paired_folds": sum(row["paired_folds"] for row in paired_rows),
        "specificity_wins": sum(row["specificity_wins"] for row in paired_rows),
    }
    for metric in METRICS:
        values = np.asarray(
            [row[f"{metric}_delta"] for row in paired_rows], dtype=np.float64
        )
        output[f"{metric}_delta_macro"] = float(np.nanmean(values))
    output["all_tasks_go"] = output["tasks_go"] == output["tasks"]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-folds", type=int)
    args = parser.parse_args()

    fold_rows = load_fold_rows(args.root)
    overall = aggregate_methods(fold_rows)
    paired = paired_candidate_rows(fold_rows, args.expected_folds)
    macro = macro_candidate_row(paired)
    paired_all = []
    macro_all = []
    for method in COMPARISON_METHODS:
        method_paired = paired_candidate_rows(
            fold_rows, args.expected_folds, candidate_method=method
        )
        paired_all.extend(method_paired)
        method_macro = macro_candidate_row(method_paired)
        if method_macro:
            macro_all.append(method_macro)
    args.root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.root / "fold_metrics.csv", fold_rows)
    _write_csv(args.root / "overall_summary.csv", overall)
    _write_csv(args.root / "paired_p4_2_vs_p4_0.csv", paired)
    _write_csv(args.root / "macro_p4_2_vs_p4_0.csv", [macro] if macro else [])
    _write_csv(args.root / "paired_all_methods_vs_p4_0.csv", paired_all)
    _write_csv(args.root / "macro_all_methods_vs_p4_0.csv", macro_all)
    with (args.root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe({
                "overall": overall,
                "paired": paired,
                "macro": macro,
                "paired_all_methods": paired_all,
                "macro_all_methods": macro_all,
            }),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\nP4-2 raw-anchor minus P4-0 raw (percentage points)")
    print("dataset   organ   folds  sens    spec    hardSpec FP-red  spec wins  GO")
    for row in paired:
        print(
            f"{row['dataset']:<9} {row['organ']:<7} {row['paired_folds']:>2}  "
            f"{100 * row['sensitivity_delta']:+.2f}  "
            f"{100 * row['specificity_delta']:+.2f}  "
            f"{100 * row['hard_negative_specificity_delta']:+.2f}  "
            f"{100 * row['negative_fp_area_reduction_delta']:+.2f}  "
            f"{row['specificity_wins']}/{row['paired_folds']}       "
            f"{'GO' if row['go'] else 'NO-GO'}"
        )
    if macro:
        print(
            "\nMacro: "
            f"sens={100 * macro['sensitivity_delta_macro']:+.2f} pp, "
            f"spec={100 * macro['specificity_delta_macro']:+.2f} pp, "
            f"hardSpec={100 * macro['hard_negative_specificity_delta_macro']:+.2f} pp, "
            f"FP-red={100 * macro['negative_fp_area_reduction_delta_macro']:+.2f} pp, "
            f"task GO={macro['tasks_go']}/{macro['tasks']}"
        )
    if macro_all:
        print("\nAll P4 variants versus P4-0 (macro percentage points)")
        print("candidate                       sens    spec    hardSpec FP-red  task GO")
        for row in macro_all:
            print(
                f"{row['candidate_method']:<31} "
                f"{100 * row['sensitivity_delta_macro']:+.2f}  "
                f"{100 * row['specificity_delta_macro']:+.2f}  "
                f"{100 * row['hard_negative_specificity_delta_macro']:+.2f}  "
                f"{100 * row['negative_fp_area_reduction_delta_macro']:+.2f}  "
                f"{row['tasks_go']}/{row['tasks']}"
            )
    print(f"\nSaved: {args.root}")


if __name__ == "__main__":
    main()
