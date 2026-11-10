import unittest
import numpy as np

from zied_quant import feature_coverage_mask


class FeatureCoverageTests(unittest.TestCase):
    def test_keeps_rows_with_at_least_eighty_percent_observed_features(self):
        X = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, np.nan, 3.0, 4.0, 5.0],
            [1.0, np.nan, np.nan, 4.0, 5.0],
        ])
        mask = feature_coverage_mask(X, 0.80)
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_rejects_rows_containing_infinite_values(self):
        X = np.array([[1.0, 2.0, np.inf, 4.0, 5.0]])
        self.assertEqual(feature_coverage_mask(X, 0.80).tolist(), [False])


if __name__ == "__main__":
    unittest.main()
