"""Aggregate current ProtoSAM ablations across MRI, CT, and Polyp."""

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
MATCHERS = ("dense_fg_hard_attn", "baseline")
PIPELINES = ("soft_prompt_a05", "coarse_only")


def last_metric(metrics, key):
    record = metrics.get(key)
    if isinstance(record, dict):
        values = record.get("values", [])
        return float(values[-1]) if values else None
    return float(record) if isinstance(record, (int, float)) else None


def parse_experiment_name(name):
    match = re.fullmatch(
        r"dinov2_l14_(mri|ct|polyp)_cca_grid_8_res_672_([a-z]+)_(.+)_fold_([0-4])",
        name.lower(),
    )
    if not match:
        return None
    modality, organ, tail, episode = match.groups()
    pipeline = next((item for item in PIPELINES if tail.endswith(item)), None)
    if pipeline is None:
        return None
    matcher_tail = tail[: -(len(pipeline) + 1)] if tail != pipeline else ""
    matcher = matcher_tail or "baseline"
    if matcher not in MATCHERS:
        return None
    dataset = {"mri": "CHAOST2", "ct": "SABS", "polyp": "Polyp"}[modality]
    display_organ = "polyp" if modality == "polyp" else organ
    return dataset, modality, display_organ, matcher, pipeline, int(episode)


def collect(roots):
    selected = {}
    for root in roots:
        if not root.exists():
            continue
        for metric_path in root.rglob("metrics.json"):
            experiment_dir = next(
                (parent for parent in metric_path.parents if parse_experiment_name(parent.name)),
                None,
            )
            if experiment_dir is None:
                continue
            identity = parse_experiment_name(experiment_dir.name)
            if identity not in selected or metric_path.stat().st_mtime > selected[identity].stat().st_mtime:
                selected[identity] = metric_path

    rows = []
    case_rows = []
    for identity, path in sorted(selected.items()):
        dataset, modality, organ, matcher, pipeline, episode = identity
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "dataset": dataset,
            "modality": modality,
            "organ": organ,
            "matcher": matcher,
            "pipeline": pipeline,
            "fold_or_episode": episode,
            "metrics_file": str(path),
        }
        row.update({name: last_metric(metrics, key) for name, key in METRIC_KEYS.items()})
        rows.append(row)
        if dataset == "Polyp":
            dice_prefix = "mar_val_batches_meanDice_"
            for key_name in metrics:
                if not key_name.startswith(dice_prefix):
                    continue
                case = key_name[len(dice_prefix):]
                case_rows.append({
                    "dataset": dataset,
                    "organ": organ,
                    "matcher": matcher,
                    "pipeline": pipeline,
                    "fold_or_episode": episode,
                    "case": case,
                    "dice": last_metric(metrics, key_name),
                    "iou": last_metric(metrics, f"mar_val_batches_meanIOU_{case}"),
                    "metrics_file": str(path),
                })
    return rows, case_rows


def aggregate(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ("dataset", "modality", "organ", "matcher", "pipeline"))
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(("dataset", "modality", "organ", "matcher", "pipeline"), key))
        summary["n"] = len(group)
        for metric in METRIC_KEYS:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_std"] = pstdev(values) if values else None
        summaries.append(summary)
    return summaries


def aggregate_cases(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ("dataset", "organ", "case", "matcher", "pipeline"))
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(("dataset", "organ", "case", "matcher", "pipeline"), key))
        summary["n"] = len(group)
        for metric in ("dice", "iou"):
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_std"] = pstdev(values) if values else None
        summaries.append(summary)
    return summaries


def print_paired_deltas(rows):
    lookup = {
        (r["dataset"], r["organ"], r["pipeline"], r["matcher"], r["fold_or_episode"]): r
        for r in rows
    }
    print("\nPaired Dense - baseline Dice")
    for dataset, organ, pipeline in sorted({(r["dataset"], r["organ"], r["pipeline"]) for r in rows}):
        deltas = []
        for episode in range(5):
            base = lookup.get((dataset, organ, pipeline, "baseline", episode))
            dense = lookup.get((dataset, organ, pipeline, "dense_fg_hard_attn", episode))
            if base and dense and base["dice"] is not None and dense["dice"] is not None:
                deltas.append(dense["dice"] - base["dice"])
        if deltas:
            print(
                f"{dataset:8s} {organ:7s} {pipeline:17s} "
                f"delta={mean(deltas):+.6f} wins={sum(x > 0 for x in deltas)}/{len(deltas)}"
            )


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, default=[Path("test_mri"), Path("test_ct"), Path("test_polyp")])
    parser.add_argument("--output-dir", type=Path, default=Path("generalization_logs"))
    args = parser.parse_args()
    rows, case_rows = collect(args.roots)
    summaries = aggregate(rows)
    case_summaries = aggregate_cases(case_rows)
    run_fields = [
        "dataset", "modality", "organ", "matcher", "pipeline", "fold_or_episode",
        *METRIC_KEYS, "metrics_file",
    ]
    summary_fields = ["dataset", "modality", "organ", "matcher", "pipeline", "n"] + [
        field for metric in METRIC_KEYS for field in (f"{metric}_mean", f"{metric}_std")
    ]
    write_csv(args.output_dir / "runs.csv", rows, run_fields)
    write_csv(args.output_dir / "summary.csv", summaries, summary_fields)
    write_csv(
        args.output_dir / "polyp_cases.csv",
        case_rows,
        ["dataset", "organ", "case", "matcher", "pipeline", "fold_or_episode", "dice", "iou", "metrics_file"],
    )
    write_csv(
        args.output_dir / "polyp_case_summary.csv",
        case_summaries,
        ["dataset", "organ", "case", "matcher", "pipeline", "n", "dice_mean", "dice_std", "iou_mean", "iou_std"],
    )
    print_paired_deltas(rows)
    print(
        f"Saved {len(rows)} runs, {len(summaries)} overall summaries, and "
        f"{len(case_summaries)} Polyp case summaries to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
