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

    def _shadow(self, sid="candidate-a", avg=0.012, trades=35, pf=1.4, dd=0.10):
        return {"status": "ok", "selected_by_regime": {"BULL": [{
            "strategy_id": sid, "spec": {"conditions": [{"feature": "rsi", "op": "lt", "threshold": 40}]},
            "trades": trades, "avg_net": avg, "profit_factor": pf,
            "max_drawdown": dd, "win_rate": 0.60,
        }]}}

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

    def test_research_champion_requires_forward_shadow_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            no_shadow = update_governance(td, discovery={
                "generation": 1, "latest_experiments": [self._candidate()],
            })
            self.assertIsNone(no_shadow["research_champion"])
            report = update_governance(td, discovery={
                "generation": 1, "latest_experiments": [self._candidate()],
            }, regime_report=self._shadow())
            self.assertTrue(report["challenger_promoted"])
            self.assertEqual(report["decision"], "initial_shadow_validated_research_champion")
            self.assertEqual(report["research_champion"]["source"], "forward_shadow_paper")
            self.assertIn("BULL", report["ai_strategy_feedback"])
            self.assertTrue(report["ai_strategy_feedback"]["BULL"]["spec"])
            self.assertFalse(report["production_promotion"]["enabled"])

    def test_shadow_gate_rejects_weak_or_under_sampled_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(update_governance(td, regime_report=self._shadow(trades=29))["research_champion"])
            self.assertIsNone(update_governance(td, regime_report=self._shadow(pf=1.10))["research_champion"])
            self.assertIsNone(update_governance(td, regime_report=self._shadow(dd=0.25))["research_champion"])

    def test_shadow_challenger_must_outperform_existing_champion(self):
        with tempfile.TemporaryDirectory() as td:
            update_governance(td, regime_report=self._shadow())
            second = update_governance(td, regime_report=self._shadow(
                sid="candidate-b", avg=0.010, pf=1.3))
            self.assertEqual(second["research_champion"]["id"], "candidate-a")
            self.assertFalse(second["challenger_promoted"])
            third = update_governance(td, regime_report=self._shadow(
                sid="candidate-c", avg=0.020, pf=1.6))
            self.assertEqual(third["research_champion"]["id"], "candidate-c")
            self.assertTrue(third["challenger_promoted"])

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
