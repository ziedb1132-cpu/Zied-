import os
import tempfile
import unittest

import zied_quant as zq


class SignalOnlyTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = zq.CFG["DB"]
        zq.CFG["DB"] = os.path.join(self.tmp.name, "signals.db")
        self.paper = zq.Paper()

    def tearDown(self):
        self.paper.db.close()
        zq.CFG["DB"] = self.old_db
        self.tmp.cleanup()

    def test_signal_open_does_not_allocate_virtual_cash_or_position(self):
        cash_before = self.paper.cash()
        signal = self.paper.open_tracked_signal(
            "TEST-USDT", 100.0, 0.02, 0.75, 1
        )
        self.assertIsNotNone(signal)
        self.assertEqual(self.paper.cash(), cash_before)
        self.assertEqual(self.paper.open_trades(), [])
        self.assertEqual(self.paper.pending_orders(), [])
        self.assertEqual(len(self.paper.active_tracked_signals()), 1)
        self.assertAlmostEqual(signal["sl"], 97.0)
        self.assertAlmostEqual(signal["tp"], 106.0)

    def test_close_records_net_percent_and_performance(self):
        signal = self.paper.open_tracked_signal(
            "TEST-USDT", 100.0, 0.02, 0.75, 1
        )
        result = self.paper.close_tracked_signal(
            signal["id"], signal["tp"], "take profit"
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["ret_pct"], 5.65, places=5)
        stats = self.paper.tracked_signal_stats()
        self.assertEqual(stats["open"], 0)
        self.assertEqual(stats["closed"], 1)
        self.assertEqual(stats["wins"], 1)
        self.assertAlmostEqual(stats["winrate"], 100.0)


if __name__ == "__main__":
    unittest.main()
