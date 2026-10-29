import json
import logging
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import deep_learning_evolution as dle


class FakePaper:
    def __init__(self):
        self.meta = {}

    def _get(self, key):
        return self.meta.get(key)

    def _set(self, key, value):
        self.meta[key] = str(value)


class WinningPatternDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "paper.db")
        self.zq = SimpleNamespace(
            CFG={"DB": self.db_path, "DIR": self.tmp.name},
            N_FEATURES=300,
            FEATURE_COLS=[f"feature_{i}" for i in range(300)],
            log=logging.getLogger("test_pattern_discovery"),
            tg=lambda _message: None,
        )
        self.paper = FakePaper()
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                CREATE TABLE signals(
                    id INTEGER PRIMARY KEY, ts REAL, feature_ts REAL,
                    feats TEXT, labeled INTEGER, label INTEGER, ret REAL
                )
            """)

    def tearDown(self):
        self.tmp.cleanup()

    def _insert_samples(self, winners=200, losers=30):
        rows = []
        for i in range(winners):
            x = np.zeros(300, dtype=float)
            x[0] = 2.0 + (i % 5) * 0.01
            rows.append((i + 1, i + 1, json.dumps(x.tolist()), 1, 1, 0.02))
        for i in range(losers):
            x = np.zeros(300, dtype=float)
            x[0] = -2.0 - (i % 5) * 0.01
            rows.append((winners + i + 1, winners + i + 1,
                         json.dumps(x.tolist()), 1, 0, -0.02))
        with sqlite3.connect(self.db_path) as db:
            db.executemany(
                "INSERT INTO signals(ts, feature_ts, feats, labeled, label, ret) "
                "VALUES(?,?,?,?,?,?)", rows
            )

    def test_reads_labelled_signals_and_writes_patterns(self):
        self._insert_samples()
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "patterns_updated")
        self.assertEqual(result["winning_samples"], 200)
        self.assertEqual(result["losing_samples"], 30)
        with open(os.path.join(self.tmp.name, "winning_patterns.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["feature_count"], 300)
        self.assertEqual(data["source"], "signals WHERE labeled=1; outcomes and net returns from later-candle evaluation")
        self.assertTrue(data["top_recurring_features"])
        self.assertEqual(data["top_recurring_features"][0]["feature"], "feature_0")

    def test_waits_for_required_winners(self):
        self._insert_samples(winners=199, losers=30)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_200_winners")
        self.assertEqual(result["winning_samples"], 199)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "winning_patterns.json")))

    def test_waits_for_loss_comparison(self):
        self._insert_samples(winners=200, losers=29)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_loss_comparison")
        self.assertEqual(result["losing_samples"], 29)

    def test_removes_stale_patterns_when_history_is_reset(self):
        stale = os.path.join(self.tmp.name, "winning_patterns.json")
        with open(stale, "w", encoding="utf-8") as fh:
            json.dump({"winning_samples": 500}, fh)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_200_winners")
        self.assertFalse(os.path.exists(stale))


if __name__ == "__main__":
    unittest.main()
