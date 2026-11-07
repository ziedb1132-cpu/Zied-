import tempfile
import unittest

from evolution_governance import update_governance, _holdout_quality


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

    def test_holdout_rejects_non_positive_expectancy(self):
        audit = _holdout_quality(self._candidate(holdout_avg=-0.001))
        self.assertFalse(audit["passed"])
        self.assertIn("holdout_expectancy_not_positive", audit["reasons"])

    def test_initial_research_champion_requires_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            report = update_governance(td, discovery={
                "generation": 1, "validation_passed": 1,
                "latest_experiments": [self._candidate()],
            })
            self.assertTrue(report["challenger_promoted"])
            self.assertEqual(report["decision"], "initial_research_champion")
            self.assertFalse(report["production_promotion"]["enabled"])

    def test_failed_challenger_does_not_replace_champion(self):
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
