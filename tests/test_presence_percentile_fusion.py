import unittest

import numpy as np

from evaluate_presence_percentile_fusion import (
    empirical_null_percentile,
    run_percentile_fusion_lovo,
)


class PresencePercentileFusionTest(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        for group_index, group in enumerate(("a", "b", "c")):
            # Each negative is misleading for only one of the two streams;
            # positives are consistently high in both streams.
            examples = (
                (False, 0.10, 0.85),
                (False, 0.80, 0.10),
                (True, 0.88, 0.86),
                (True, 0.92, 0.90),
            )
            for index, (present, lr, pc) in enumerate(examples):
                rows.append({
                    "scan_id": group,
                    "gt_present": present,
                    "fg_bg_likelihood_ratio": lr + group_index * 0.001,
                    "patchcore_bidirectional_score": pc + group_index * 0.001,
                    "final_predicted_area_ratio": 0.1 + index * 0.01,
                    "hard_negative_3": not present,
                })
        return rows

    def test_empirical_percentile_uses_strict_less_than(self):
        actual = empirical_null_percentile([1.0, 2.0, 2.0, 4.0], [0.0, 2.0, 3.0, 5.0])
        np.testing.assert_allclose(actual, [0.0, 0.25, 0.75, 1.0])

    def test_lovo_outputs_every_stream_and_held_out_row(self):
        rows = self._rows()
        split, gated, summary = run_percentile_fusion_lovo(
            rows, target_sensitivity=1.0
        )
        self.assertEqual(len(split), 3 * 7)
        self.assertEqual(len(gated), len(rows) * 7)
        self.assertEqual(len(summary), 7)
        self.assertEqual(
            {row["fusion"] for row in summary},
            {
                "lr_raw",
                "patchcore_raw",
                "lr_percentile",
                "patchcore_percentile",
                "percentile_avg",
                "percentile_lr60",
                "percentile_min",
            },
        )

    def test_empirical_percentile_is_monotonic(self):
        scores = np.asarray([-2.0, -0.5, 0.0, 0.1, 1.0, 3.0])
        percentiles = empirical_null_percentile([-1.0, 0.0, 0.5, 2.0], scores)
        self.assertTrue(np.all(np.diff(percentiles) >= 0.0))
        self.assertTrue(np.all((0.0 <= percentiles) & (percentiles <= 1.0)))

    def test_min_fusion_rejects_complementary_false_matches(self):
        _, _, summary = run_percentile_fusion_lovo(
            self._rows(), target_sensitivity=1.0
        )
        by_name = {row["fusion"]: row for row in summary}
        self.assertGreaterEqual(
            by_name["percentile_min"]["specificity"],
            by_name["lr_percentile"]["specificity"],
        )


if __name__ == "__main__":
    unittest.main()
