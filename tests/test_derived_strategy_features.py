import unittest

import pandas as pd

from derived_strategy_features import (
    approve_record, approved_feature_matrix, strategy_feature_name,
)


class DerivedStrategyFeatureTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "version": 1, "kind": "symbolic_rule",
            "conditions": [{"feature": "ret1", "op": "gt", "threshold": 0.01}],
        }
        self.evidence = {
            "train": {"trades": 25, "avg_net": 0.01},
            "validation": {"trades": 22, "avg_net": 0.02},
            "holdout": {"trades": 21, "avg_net": 0.005},
        }

    def test_500_win_loss_gate(self):
        self.assertIsNone(approve_record(self.spec, self.evidence, 499, 500, {"ret1"}))
        self.assertIsNone(approve_record(self.spec, self.evidence, 500, 499, {"ret1"}))

    def test_approved_feature_is_causal_rule_output(self):
        record = approve_record(self.spec, self.evidence, 500, 500, {"ret1"})
        self.assertEqual(record["feature_name"], strategy_feature_name(self.spec))
        frame = pd.DataFrame({"ret1": [-0.01, 0.02, 0.03]})
        result = approved_feature_matrix(frame, {"features": [record]})
        self.assertEqual(result[record["feature_name"]].tolist(), [0.0, 1.0, 1.0])

    def test_negative_holdout_is_not_approved(self):
        evidence = dict(self.evidence)
        evidence["holdout"] = {"trades": 30, "avg_net": -0.001}
        self.assertIsNone(approve_record(self.spec, evidence, 500, 500, {"ret1"}))


if __name__ == "__main__":
    unittest.main()
