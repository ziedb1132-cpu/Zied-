import tempfile
import unittest

from evolution_governance import update_governance, _holdout_quality, performance_risk_summary


class EvolutionGovernanceTests(unittest.TestCase):
    def _candidate(self, score=30.0, holdout_avg=0.01):
        return {
            "id": "candidate-a", "generation": 1, "status": "validation_pass",
            "score": score, "spec": {"conditions": []}, "genome": {},
            "validation": {"trades": 40, "avg_net": 0.012, "profit_factor": 1.3},
            "holdout_observation": {
                "trades": 15, "avg_net": holdout_avg, "profit_factor": 1.4,
            },
            "per_symbol": [
                {"holdout": {"avg_net": 0.01}},
                {"holdout": {"avg_net": 0.02}},
                {"holdout": {"avg_net": 0.005}},
            ],
        }

    def test_realized_payoff_summary_exposes_loss_asymmetry(self):
        report = performance_risk_summary([0.01, 0.02, -0.03, -0.01] * 8)
        self.assertEqual(report["trades"], 32)
        self.assertAlmostEqual(report["avg_win"], 0.015)
        self.assertAlmostEqual(report["avg_loss"], -0.02)
        self.assertAlmostEqual(report["payoff_ratio"], 0.75)
        self.assertAlmostEqual(report["break_even_win_rate"], 0.5714285714)
        self.assertEqual(report["status"], "ok")

    def test_validation_gate_rejects_weak_candidate_even_if_holdout_is_positive(self):
        from evolution_governance import _candidate_from_discovery
        weak = self._candidate()
        weak["validation"]["trades"] = 12
        weak["validation"]["avg_net"] = -0.001
        weak["validation"]["profit_factor"] = 0.9
        result = _candidate_from_discovery({
            "latest_experiments": [weak],
        })
        self.assertIsNone(result)

    def test_validation_gate_accepts_only_sufficient_positive_net_evidence(self):
        from evolution_governance import _candidate_from_discovery
        result = _candidate_from_discovery({
            "latest_experiments": [self._candidate()],
        })
        self.assertIsNotNone(result)
        self.assertTrue(result["validation_gate"]["passed"])

    def test_holdout_rejects_non_positive_expectancy(self):
        audit = _holdout_quality(self._candidate(holdout_avg=-0.001))
        self.assertFalse(audit["passed"])
        self.assertIn("holdout_expectancy_not_positive", audit["reasons"])

    def test_initial_research_champion_uses_validation_only(self):
        with tempfile.TemporaryDirectory() as td:
            report = update_governance(td, discovery={
                "generation": 1, "validation_passed": 1,
                "latest_experiments": [self._candidate()],
            })
            self.assertTrue(report["challenger_promoted"])
            self.assertEqual(report["decision"], "initial_research_champion_validation_only")
            self.assertIn("diagnostic_only", report["holdout_usage"])
            self.assertFalse(report["production_promotion"]["enabled"])

    def test_challenger_selection_is_not_changed_by_reused_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            first = update_governance(td, discovery={
                "generation": 1, "latest_experiments": [self._candidate()],
            })
            original = first["research_champion"]["id"]
            bad = self._candidate(holdout_avg=-0.02)
            bad["id"] = "candidate-b"
            second = update_governance(td, discovery={
                "generation": 2, "latest_experiments": [bad],
            })
            self.assertEqual(second["research_champion"]["id"], original)
            self.assertFalse(second["challenger_promoted"])
            # Identical validation score: negative holdout must not change selection
            # or cause the candidate to be rejected by a reused holdout.
            better_validation = self._candidate(score=31.0, holdout_avg=-0.02)
            better_validation["id"] = "candidate-b"
            better_validation["validation"]["avg_net"] = 0.02
            third = update_governance(td, discovery={
                "generation": 3, "latest_experiments": [better_validation],
            })
            self.assertEqual(third["research_champion"]["id"], "candidate-b")
            self.assertTrue(third["challenger_promoted"])
            self.assertFalse(third["research_champion"]["holdout"]["passed"])

    def test_plateau_increases_next_generation_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            for generation in range(1, 5):
                update_governance(td, discovery={
                    "generation": generation,
                    "latest_experiments": [self._candidate(score=30.0)],
                })
            from pathlib import Path
            import json
            memory = json.loads((Path(td) / "evolution_governance.json").read_text())
            self.assertGreaterEqual(memory["mutation_rate"], 0.60)


if __name__ == "__main__":
    unittest.main()
