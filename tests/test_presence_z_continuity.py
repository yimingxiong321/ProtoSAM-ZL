import unittest

import numpy as np

from evaluate_presence_z_continuity import (
    anchor_connected_interval,
    run_z_continuity_lovo,
    scanwise_zscore,
)


class PresenceZContinuityTest(unittest.TestCase):
    def test_scanwise_zscore(self):
        values = scanwise_zscore([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(values.mean()), 0.0)
        self.assertAlmostEqual(float(values.std()), 1.0)
        np.testing.assert_array_equal(scanwise_zscore([2.0, 2.0]), [0.0, 0.0])

    def test_anchor_removes_island(self):
        keep = anchor_connected_interval(
            [0.8, 0.1, 0.7, 0.9, 0.7, 0.1],
            [0, 1, 2, 3, 4, 5],
            low_threshold=0.5,
            absolute_min=0.1,
        )
        np.testing.assert_array_equal(keep, [False, False, True, True, True, False])

    def test_one_slice_gap_is_filled(self):
        keep = anchor_connected_interval(
            [0.1, 0.7, 0.2, 0.9, 0.8, 0.1],
            [0, 1, 2, 3, 4, 5],
            low_threshold=0.5,
            absolute_min=0.1,
            max_gap=1,
        )
        np.testing.assert_array_equal(keep, [False, True, True, True, True, False])

    def test_fail_safe_and_missing_z_stop_growth(self):
        blocked = anchor_connected_interval(
            [0.01, 0.02], [0, 1], 0.0, absolute_min=0.1
        )
        np.testing.assert_array_equal(blocked, [False, False])
        missing = anchor_connected_interval(
            [0.8, 0.9, 0.8], [0, 1, 3], 0.5, absolute_min=0.1
        )
        np.testing.assert_array_equal(missing, [True, True, False])

    @staticmethod
    def _rows():
        rows = []
        for group_index, group in enumerate(("a", "b", "c")):
            scores = [0.8, 0.1, 0.7, 0.9, 0.7, 0.1]
            labels = [False, False, True, True, True, False]
            for z, (score, label) in enumerate(zip(scores, labels)):
                rows.append({
                    "scan_id": group,
                    "z_id": z,
                    "sample_index": z,
                    "gt_present": label,
                    "fg_bg_likelihood_ratio": score + group_index * 0.001,
                    "final_predicted_area_ratio": 0.1,
                    "hard_negative_3": not label,
                })
        return rows

    def test_lovo_returns_four_ablation_rows(self):
        per_volume, gated, summary = run_z_continuity_lovo(
            self._rows(), target_sensitivity=1.0
        )
        self.assertEqual(len(per_volume), 3 * 4)
        self.assertEqual(len(gated), len(self._rows()) * 4)
        self.assertEqual(len(summary), 4)
        anchored = next(row for row in summary if row["method"] == "p4_2_raw_anchor")
        baseline = next(row for row in summary if row["method"] == "p4_0_raw")
        self.assertGreaterEqual(anchored["specificity"], baseline["specificity"])


if __name__ == "__main__":
    unittest.main()
