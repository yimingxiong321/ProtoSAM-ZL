"""Exact no-op check for the P1 candidate-audit intervention."""

import argparse
import csv


KEY = ("fold", "scan_id", "z_id", "sample_index")


def load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {tuple(row[name] for name in KEY): row for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"duplicate sample keys in {path}")
    return indexed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control")
    parser.add_argument("audit")
    args = parser.parse_args()
    control = load(args.control)
    audit = load(args.audit)
    if set(control) != set(audit):
        missing = len(set(control) - set(audit))
        extra = len(set(audit) - set(control))
        raise SystemExit(f"sample-set mismatch: missing={missing}, extra={extra}")
    mismatches = [
        key for key in control
        if control[key]["prediction_sha256"] != audit[key]["prediction_sha256"]
        or control[key]["predicted_area_pixels"] != audit[key]["predicted_area_pixels"]
    ]
    if mismatches:
        raise SystemExit(
            f"P1 no-op check failed: {len(mismatches)}/{len(control)} masks differ; "
            f"first={mismatches[0]}"
        )
    print(f"P1 no-op check passed: {len(control)} masks are exactly identical")


if __name__ == "__main__":
    main()
