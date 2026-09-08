import csv
import tempfile
import unittest
from pathlib import Path

from summarize_presence_continuity_generalization import (
    aggregate_methods,
    load_fold_rows,
    macro_candidate_row,
    paired_candidate_rows,
)


FIELDS = [
    "method", "n_groups", "n", "n_positive", "n_negative",
    "n_hard_negative", "sensitivity", "specificity",
    "hard_negative_specificity", "negative_fp_area_reduction",
    "mean_negative_fp_area_after",
]


class PresenceContinuitySummaryTest(unittest.TestCase):
    def _write_fold(self, root, fold, sensitivity_delta=0.0):
        path = root / "mri" / "spleen" / f"fold{fold}" / "summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "method": "p4_0_raw", "n_groups": 4, "n": 100,
                "n_positive": 60, "n_negative": 40, "n_hard_negative": 10,
                "sensitivity": 0.97, "specificity": 0.40,
                "hard_negative_specificity": 0.30,
                "negative_fp_area_reduction": 0.20,
                "mean_negative_fp_area_after": 0.08,
            },
            {
                "method": "p4_2_raw_anchor", "n_groups": 4, "n": 100,
                "n_positive": 60, "n_negative": 40, "n_hard_negative": 10,
                "sensitivity": 0.97 + sensitivity_delta, "specificity": 0.50,
                "hard_negative_specificity": 0.40,
                "negative_fp_area_reduction": 0.30,
                "mean_negative_fp_area_after": 0.06,
            },
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_load_aggregate_and_go(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold in range(5):
                self._write_fold(root, fold, sensitivity_delta=-0.005)
            rows = load_fold_rows(root)
            self.assertEqual(len(rows), 10)
            overall = aggregate_methods(rows)
            self.assertEqual(len(overall), 2)
            paired = paired_candidate_rows(rows, expected_folds=5)
            self.assertEqual(len(paired), 1)
            self.assertAlmostEqual(paired[0]["specificity_delta"], 0.10)
            self.assertEqual(paired[0]["specificity_wins"], 5)
            self.assertTrue(paired[0]["go"])
            macro = macro_candidate_row(paired)
            self.assertEqual(macro["tasks_go"], 1)
            self.assertTrue(macro["all_tasks_go"])

    def test_incomplete_folds_are_not_go(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fold(root, 0)
            paired = paired_candidate_rows(load_fold_rows(root), expected_folds=5)
            self.assertFalse(paired[0]["go_complete_folds"])
            self.assertFalse(paired[0]["go"])


if __name__ == "__main__":
    unittest.main()
