"""Evaluate scan-wise normalization and Z-axis continuity for presence gating.

Six patient-wise LOVO branches are evaluated:

P4-0: raw LR with a calibration-only sensitivity-locked threshold.
P4-1: scan-wise Z-score with a calibration-only threshold, no connectivity.
P4-2: raw LR, maximum-score seed, and one connected Z interval.
P4-2-gap1: raw LR with the same anchor and one-slice gaps.
P4-3: scan-wise Z-score with the same anchor, evaluated with zero/one-slice gaps.

The held-out volume never contributes labels or scores to threshold fitting.
The procedure is transductive within a query volume because its complete LR
sequence is used for scan-wise normalization and topology.
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


TRUE_VALUES = {"1", "true", "yes", "y", "t"}
METHODS = (
    "p4_0_raw",
    "p4_1_zscore",
    "p4_2_raw_anchor",
    "p4_2_raw_anchor_gap1",
    "p4_3_zscore_anchor",
    "p4_3_zscore_anchor_gap1",
)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def scanwise_zscore(scores, eps=1e-8):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1D array")
    if not np.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    std = float(scores.std())
    if std <= eps:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / std


def anchor_connected_interval(
    scores,
    z_positions,
    low_threshold,
    raw_scores=None,
    absolute_min=0.1,
    max_gap=0,
):
    """Grow bidirectionally from the maximum score through eligible slices.

    A gap may contain at most ``max_gap`` consecutive ineligible slices and is
    retained only when followed by another eligible, physically adjacent
    slice.  Growth never jumps over missing Z indices.
    """
    scores = np.asarray(scores, dtype=np.float64)
    raw_scores = scores if raw_scores is None else np.asarray(raw_scores, dtype=np.float64)
    z_positions = np.asarray(z_positions, dtype=np.int64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1D array")
    if scores.shape != raw_scores.shape or scores.shape != z_positions.shape:
        raise ValueError("scores, raw_scores, and z_positions must have equal shapes")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    if not np.isfinite(scores).all() or not np.isfinite(raw_scores).all():
        raise ValueError("scores must all be finite")

    keep = np.zeros(scores.size, dtype=np.bool_)
    if float(raw_scores.max()) < float(absolute_min):
        return keep
    seed = int(np.argmax(scores))
    keep[seed] = True

    for direction in (-1, 1):
        index = seed + direction
        while 0 <= index < scores.size:
            previous = index - direction
            if abs(int(z_positions[index]) - int(z_positions[previous])) != 1:
                break
            if scores[index] >= float(low_threshold):
                keep[index] = True
                index += direction
                continue

            gap_indices = []
            probe = index
            while (
                len(gap_indices) < max_gap
                and 0 <= probe < scores.size
                and scores[probe] < float(low_threshold)
            ):
                previous = probe - direction
                if abs(int(z_positions[probe]) - int(z_positions[previous])) != 1:
                    break
                gap_indices.append(probe)
                probe += direction
            if (
                not gap_indices
                or not 0 <= probe < scores.size
                or scores[probe] < float(low_threshold)
                or abs(int(z_positions[probe]) - int(z_positions[probe - direction])) != 1
            ):
                break
            keep[gap_indices] = True
            keep[probe] = True
            index = probe + direction
    return keep


def _prepare_rows(rows, group_column, z_column, input_fusion="lr_raw"):
    if not rows:
        raise ValueError("input rows are empty")
    if "fusion" in rows[0]:
        rows = [row for row in rows if row.get("fusion") == input_fusion]
        if not rows:
            raise ValueError(f"no rows found for fusion={input_fusion!r}")
    seen = set()
    prepared = []
    for row in rows:
        key = (str(row[group_column]), str(row[z_column]), str(row.get("sample_index", "")))
        if key in seen:
            raise ValueError(f"duplicate slice row: {key}")
        seen.add(key)
        prepared.append(dict(row))
    return prepared


def _group_scans(rows, group_column, z_column, score_column):
    scans = {}
    for row in rows:
        scans.setdefault(str(row[group_column]), []).append(row)
    output = {}
    for name, scan_rows in scans.items():
        scan_rows.sort(key=lambda row: float(row[z_column]))
        z = np.asarray([int(float(row[z_column])) for row in scan_rows], dtype=np.int64)
        if len(np.unique(z)) != len(z):
            raise ValueError(f"scan {name!r} has duplicate Z indices")
        raw = np.asarray([float(row[score_column]) for row in scan_rows], dtype=np.float64)
        output[name] = {
            "rows": scan_rows,
            "z": z,
            "raw": raw,
            "zscore": scanwise_zscore(raw),
            "labels": np.asarray([_as_bool(row["gt_present"]) for row in scan_rows]),
        }
    return output


def _flatten(scans, names, field):
    return np.concatenate([np.asarray(scans[name][field]) for name in names])


def _sequence_predictions(scans, names, field, threshold, absolute_min, max_gap):
    output = {}
    for name in names:
        scan = scans[name]
        output[name] = anchor_connected_interval(
            scan[field],
            scan["z"],
            threshold,
            raw_scores=scan["raw"],
            absolute_min=absolute_min,
            max_gap=max_gap,
        )
    return output


def _calibrate_sequence_threshold(
    scans,
    names,
    field,
    target_sensitivity,
    absolute_min,
    max_gap,
):
    values = _flatten(scans, names, field)
    # A value just below the minimum represents the all-eligible limit.
    candidates = np.unique(
        np.concatenate(([np.nextafter(values.min(), -np.inf)], values))
    )[::-1]
    labels = _flatten(scans, names, "labels").astype(np.bool_)
    for threshold in candidates:
        predictions_by_scan = _sequence_predictions(
            scans, names, field, threshold, absolute_min, max_gap
        )
        predictions = np.concatenate([predictions_by_scan[name] for name in names])
        sensitivity = float(predictions[labels].mean()) if labels.any() else float("nan")
        if sensitivity >= target_sensitivity:
            return float(threshold)
    raise ValueError("no sequence threshold reaches target calibration sensitivity")


def _predictions_for_method(scans, names, method, threshold, absolute_min):
    if method == "p4_0_raw":
        return {name: scans[name]["raw"] >= threshold for name in names}
    if method == "p4_1_zscore":
        output = {}
        for name in names:
            scan = scans[name]
            prediction = scan["zscore"] >= threshold
            if float(scan["raw"].max()) < absolute_min:
                prediction = np.zeros_like(prediction)
            output[name] = prediction
        return output
    if method == "p4_2_raw_anchor":
        return _sequence_predictions(scans, names, "raw", threshold, absolute_min, 0)
    if method == "p4_2_raw_anchor_gap1":
        return _sequence_predictions(scans, names, "raw", threshold, absolute_min, 1)
    if method == "p4_3_zscore_anchor":
        return _sequence_predictions(scans, names, "zscore", threshold, absolute_min, 0)
    if method == "p4_3_zscore_anchor_gap1":
        return _sequence_predictions(scans, names, "zscore", threshold, absolute_min, 1)
    raise ValueError(f"unsupported method: {method}")


def _calibrate_method(scans, names, method, target_sensitivity, absolute_min):
    labels = _flatten(scans, names, "labels").astype(np.bool_)
    if method == "p4_0_raw":
        return sensitivity_constrained_threshold(
            labels, _flatten(scans, names, "raw"), target_sensitivity
        )
    if method == "p4_1_zscore":
        return sensitivity_constrained_threshold(
            labels, _flatten(scans, names, "zscore"), target_sensitivity
        )
    if method == "p4_2_raw_anchor":
        return _calibrate_sequence_threshold(
            scans, names, "raw", target_sensitivity, absolute_min, 0
        )
    if method == "p4_2_raw_anchor_gap1":
        return _calibrate_sequence_threshold(
            scans, names, "raw", target_sensitivity, absolute_min, 1
        )
    if method == "p4_3_zscore_anchor":
        return _calibrate_sequence_threshold(
            scans, names, "zscore", target_sensitivity, absolute_min, 0
        )
    if method == "p4_3_zscore_anchor_gap1":
        return _calibrate_sequence_threshold(
            scans, names, "zscore", target_sensitivity, absolute_min, 1
        )
    raise ValueError(f"unsupported method: {method}")


def _metrics(scans, names, predictions, area_column, hard_column):
    labels = _flatten(scans, names, "labels").astype(np.bool_)
    predicted = np.concatenate([predictions[name] for name in names])
    areas = np.concatenate([
        np.asarray([float(row[area_column]) for row in scans[name]["rows"]])
        for name in names
    ])
    hard = np.concatenate([
        np.asarray([_as_bool(row.get(hard_column, False)) for row in scans[name]["rows"]])
        for name in names
    ])
    return gate_metrics(labels, predicted, predicted_area_ratio=areas, hard_negative=hard)


def run_z_continuity_lovo(
    rows,
    score_column="fg_bg_likelihood_ratio",
    group_column="scan_id",
    z_column="z_id",
    target_sensitivity=0.975,
    absolute_min=0.1,
    area_column="final_predicted_area_ratio",
    hard_negative_column="hard_negative_3",
    input_fusion="lr_raw",
):
    required = {"gt_present", score_column, group_column, z_column, area_column}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    prepared = _prepare_rows(rows, group_column, z_column, input_fusion)
    scans = _group_scans(prepared, group_column, z_column, score_column)
    names = sorted(scans, key=str)
    if len(names) < 2:
        raise ValueError("LOVO evaluation requires at least two scans")

    per_volume = []
    gated_rows = []
    pooled = {
        method: {"labels": [], "predictions": [], "areas": [], "hard": []}
        for method in METHODS
    }
    for held_out in names:
        calibration_names = [name for name in names if name != held_out]
        test_names = [held_out]
        for method in METHODS:
            threshold = _calibrate_method(
                scans, calibration_names, method, target_sensitivity, absolute_min
            )
            calibration_predictions = _predictions_for_method(
                scans, calibration_names, method, threshold, absolute_min
            )
            test_predictions = _predictions_for_method(
                scans, test_names, method, threshold, absolute_min
            )
            calibration_metrics = _metrics(
                scans, calibration_names, calibration_predictions, area_column, hard_negative_column
            )
            test_metrics = _metrics(
                scans, test_names, test_predictions, area_column, hard_negative_column
            )
            per_volume.append({
                "method": method,
                "held_out_group": held_out,
                "threshold": threshold,
                "absolute_min": absolute_min,
                **{f"calibration_{key}": value for key, value in calibration_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            })

            scan = scans[held_out]
            predictions = test_predictions[held_out]
            for index, (source, prediction) in enumerate(zip(scan["rows"], predictions)):
                output = dict(source)
                output.update({
                    "continuity_method": method,
                    "held_out_group": held_out,
                    "scanwise_zscore": float(scan["zscore"][index]),
                    "continuity_threshold": float(threshold),
                    "continuity_present": bool(prediction),
                    "continuity_gated_area_ratio": (
                        float(source[area_column]) if prediction else 0.0
                    ),
                })
                gated_rows.append(output)

            target = pooled[method]
            target["labels"].extend(scan["labels"].tolist())
            target["predictions"].extend(predictions.tolist())
            target["areas"].extend(
                [float(row[area_column]) for row in scan["rows"]]
            )
            target["hard"].extend(
                [_as_bool(row.get(hard_negative_column, False)) for row in scan["rows"]]
            )

    summary = []
    for method in METHODS:
        values = pooled[method]
        metrics = gate_metrics(
            values["labels"],
            values["predictions"],
            predicted_area_ratio=values["areas"],
            hard_negative=values["hard"],
        )
        method_rows = [row for row in per_volume if row["method"] == method]
        summary.append({
            "method": method,
            "n_groups": len(names),
            "target_calibration_sensitivity": target_sensitivity,
            "absolute_min": absolute_min,
            "threshold_mean": float(np.mean([row["threshold"] for row in method_rows])),
            "threshold_min": float(np.min([row["threshold"] for row in method_rows])),
            "threshold_max": float(np.max([row["threshold"] for row in method_rows])),
            **metrics,
        })
    return per_volume, gated_rows, summary


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
    parser.add_argument("--score", default="fg_bg_likelihood_ratio")
    parser.add_argument("--group-column", default="scan_id")
    parser.add_argument("--z-column", default="z_id")
    parser.add_argument("--target-sensitivity", type=float, default=0.975)
    parser.add_argument("--absolute-min", type=float, default=0.1)
    parser.add_argument("--area-column", default="final_predicted_area_ratio")
    parser.add_argument("--hard-negative-column", default="hard_negative_3")
    parser.add_argument("--input-fusion", default="lr_raw")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("presence_z_continuity")
    )
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    per_volume, gated_rows, summary = run_z_continuity_lovo(
        rows,
        score_column=args.score,
        group_column=args.group_column,
        z_column=args.z_column,
        target_sensitivity=args.target_sensitivity,
        absolute_min=args.absolute_min,
        area_column=args.area_column,
        hard_negative_column=args.hard_negative_column,
        input_fusion=args.input_fusion,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_volume.csv", per_volume)
    _write_csv(args.output_dir / "gated_slices.csv", gated_rows)
    _write_csv(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2)

    print("\nZ-axis continuity LOVO")
    print("method                         sens   spec   hardSpec FP-red")
    for row in summary:
        print(
            f"{row['method']:<30} {row['sensitivity']:.4f} "
            f"{row['specificity']:.4f} "
            f"{row['hard_negative_specificity']:.4f} "
            f"{row['negative_fp_area_reduction']:.4f}"
        )
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
