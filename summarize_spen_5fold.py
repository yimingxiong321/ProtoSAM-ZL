"""Summarize paired baseline/SPEN Sacred metrics across five folds."""

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev


METRIC_KEYS = {
    "dice": "mar_val_batches_meanDice",
    "iou": "mar_val_al_batches_meanIOU",
    "precision": "mar_val_batches_meanPrec",
    "recall": "mar_val_al_batches_meanRec",
    "boundary_dice": "mar_val_batches_meanBoundaryDice",
    "boundary_recall": "mar_val_batches_meanBoundaryRecall",
    "hd95": "mar_val_batches_meanHD95",
}


def last_metric(metrics, key):
    record = metrics.get(key)
    if record is None:
        return None
    if isinstance(record, dict) and "values" in record:
        values = record["values"]
        return float(values[-1]) if values else None
    if isinstance(record, (int, float)):
        return float(record)
    return None


def collect(root, organ, pipeline_mode):
    selected = {}
    if pipeline_mode != "none":
        matcher_pattern = (
            r"(?:_((?:spen|dense)_[a-z0-9_]+))?_"
            + re.escape(pipeline_mode.lower())
        )
    else:
        # Do not let the generic matcher name consume a coarse-only suffix.
        matcher_pattern = r"(?:_((?:spen|dense)_[a-z0-9_]+(?<!_coarse_only)))?"
    experiment_pattern = re.compile(
        rf"_res_672_{re.escape(organ.lower())}"
        rf"{matcher_pattern}_fold_([0-4])(?:[/\\])"
    )
    for metric_path in root.rglob("metrics.json"):
        text = str(metric_path).lower()
        match = experiment_pattern.search(text)
        if not match:
            continue
        matcher = match.group(1) or "baseline"
        fold = int(match.group(2))
        key = (matcher, fold)
        if key not in selected or metric_path.stat().st_mtime > selected[key].stat().st_mtime:
            selected[key] = metric_path

    rows = []
    for (matcher, fold), path in sorted(selected.items()):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = {"matcher": matcher, "fold": fold, "metrics_file": str(path)}
        row.update({name: last_metric(metrics, key) for name, key in METRIC_KEYS.items()})
        rows.append(row)
    return rows


def print_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["matcher"], []).append(row)
    print("\nFold summary")
    matcher_order = ["baseline"] + sorted(name for name in grouped if name != "baseline")
    for matcher in matcher_order:
        group = grouped.get(matcher, [])
        dice = [row["dice"] for row in group if row["dice"] is not None]
        iou = [row["iou"] for row in group if row["iou"] is not None]
        dice_text = f"Dice={mean(dice):.6f}±{pstdev(dice):.6f}" if dice else "Dice=missing"
        iou_text = f"IoU={mean(iou):.6f}±{pstdev(iou):.6f}" if iou else "IoU=missing"
        print(f"{matcher:10s} folds={len(group)} {dice_text} {iou_text}")

    by_key = {(row["matcher"], row["fold"]): row for row in rows}
    for matcher in (name for name in matcher_order if name != "baseline"):
        deltas = []
        for fold in range(5):
            base = by_key.get(("baseline", fold))
            spen = by_key.get((matcher, fold))
            if base and spen and base["dice"] is not None and spen["dice"] is not None:
                delta = spen["dice"] - base["dice"]
                deltas.append(delta)
                print(f"{matcher} fold {fold}: Dice delta={delta:+.6f}")
        if deltas:
            print(
                f"{matcher} paired mean Dice delta={mean(deltas):+.6f}; "
                f"wins={sum(delta > 0 for delta in deltas)}/{len(deltas)}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("test_mri"))
    parser.add_argument("--organ", default="liver")
    parser.add_argument(
        "--pipeline-mode",
        default="none",
        help="Experiment suffix, e.g. none, coarse_only, or soft_prompt_a05",
    )
    parser.add_argument("--output", type=Path, default=Path("spen_5fold_summary.csv"))
    args = parser.parse_args()

    rows = collect(args.root, args.organ, args.pipeline_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "matcher", "fold", "dice", "iou", "precision", "recall",
                "boundary_dice", "boundary_recall", "hd95", "metrics_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
