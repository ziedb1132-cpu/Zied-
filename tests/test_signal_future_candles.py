import unittest

import pandas as pd

import zied_quant as zq


class FutureSignalCandlesTests(unittest.TestCase):
    def test_excludes_candle_already_in_progress_when_signal_is_created(self):
        # Signal created at 10:47 UTC. The 10:00 candle contains pre-entry movement.
        candles = pd.DataFrame({
            "ts": [9 * 3600000, 10 * 3600000, 11 * 3600000, 12 * 3600000],
            "h": [101.0, 150.0, 103.0, 104.0],
            "l": [99.0, 80.0, 100.0, 101.0],
            "c": [100.0, 110.0, 102.0, 103.0],
        })
        path = zq.future_candles_for_signal(candles, (10 * 3600 + 47 * 60) )
        self.assertEqual(path["ts"].tolist(), [11 * 3600000, 12 * 3600000])

    def test_signal_exactly_on_hour_boundary_can_use_candle_starting_then(self):
        candles = pd.DataFrame({"ts": [10 * 3600000, 11 * 3600000]})
        path = zq.future_candles_for_signal(candles, 10 * 3600)
        self.assertEqual(path["ts"].tolist(), [10 * 3600000, 11 * 3600000])

    def test_empty_candles_remain_empty(self):
        candles = pd.DataFrame(columns=["ts", "h", "l", "c"])
        self.assertTrue(zq.future_candles_for_signal(candles, 1).empty)


if __name__ == "__main__":
    unittest.main()
