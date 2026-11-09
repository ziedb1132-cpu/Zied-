import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zied_quant as zq


class EntryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.original = zq.CFG.get("PROB_MIN")

    def tearDown(self):
        zq.CFG["PROB_MIN"] = self.original

    def test_uses_validated_model_threshold_by_default(self):
        class Brain:
            ready = True
            thr = 0.73

        zq.CFG["PROB_MIN"] = None
        self.assertAlmostEqual(zq.effective_probability_threshold(Brain()), 0.73)

    def test_explicit_probability_threshold_overrides_model(self):
        class Brain:
            ready = True
            thr = 0.73

        zq.CFG["PROB_MIN"] = 0.61
        self.assertAlmostEqual(zq.effective_probability_threshold(Brain()), 0.61)

    def test_unavailable_model_uses_conservative_fallback(self):
        class Brain:
            ready = False
            thr = 0.73

        zq.CFG["PROB_MIN"] = None
        self.assertAlmostEqual(zq.effective_probability_threshold(Brain()), 0.55)

    def test_invalid_override_is_rejected(self):
        class Brain:
            ready = True
            thr = 0.73

        zq.CFG["PROB_MIN"] = 1.5
        with self.assertRaises(ValueError):
            zq.effective_probability_threshold(Brain())


if __name__ == "__main__":
    unittest.main()
