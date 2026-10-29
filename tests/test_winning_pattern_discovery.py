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
                    id INTEGER PRIMARY KEY, ts REAL, feature_ts REAL, inst TEXT,
                    feats TEXT, labeled INTEGER, label INTEGER, ret REAL
                )
            """)
            db.execute("""
                CREATE TABLE tracked_signals(
                    signal_id INTEGER, status TEXT, closed_ts REAL
                )
            """)

    def tearDown(self):
        self.tmp.cleanup()

    def _insert_samples(self, winners=500, losers=500):
        rows = []
        for i in range(winners):
            x = np.zeros(300, dtype=float)
            x[0] = 2.0 + (i % 5) * 0.01
            rows.append((i + 1, i + 1, i + 1, f"WIN{i}-USDT",
                         json.dumps(x.tolist()), 1, 1, 0.02))
        for i in range(losers):
            x = np.zeros(300, dtype=float)
            x[0] = -2.0 - (i % 5) * 0.01
            sid = winners + i + 1
            rows.append((sid, sid, sid, f"LOSS{i}-USDT",
                         json.dumps(x.tolist()), 1, 0, -0.02))
        with sqlite3.connect(self.db_path) as db:
            db.executemany(
                "INSERT INTO signals(id, ts, feature_ts, inst, feats, labeled, label, ret) "
                "VALUES(?,?,?,?,?,?,?,?)", rows
            )
            db.executemany(
                "INSERT INTO tracked_signals(signal_id, status, closed_ts) VALUES(?,?,?)",
                [(row[0], "CLOSED", row[1]) for row in rows]
            )

    def test_reads_only_closed_signals_and_writes_patterns(self):
        self._insert_samples()
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "patterns_updated")
        self.assertEqual(result["winning_samples"], 500)
        self.assertEqual(result["losing_samples"], 500)
        with open(os.path.join(self.tmp.name, "winning_patterns.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["feature_count"], 300)
        self.assertEqual(data["source"], "closed tracked signals only; outcomes and net returns from later-candle evaluation")
        self.assertTrue(data["top_recurring_features"])
        self.assertEqual(data["top_recurring_features"][0]["feature"], "feature_0")

    def test_waits_until_500_closed_winners(self):
        self._insert_samples(winners=499, losers=500)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_closed_outcomes")
        self.assertEqual(result["winning_samples"], 499)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "winning_patterns.json")))

    def test_waits_until_500_closed_losers(self):
        self._insert_samples(winners=500, losers=499)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_closed_outcomes")
        self.assertEqual(result["losing_samples"], 499)

    def test_removes_stale_patterns_when_history_is_reset(self):
        stale = os.path.join(self.tmp.name, "winning_patterns.json")
        with open(stale, "w", encoding="utf-8") as fh:
            json.dump({"winning_samples": 500}, fh)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_closed_outcomes")
        self.assertFalse(os.path.exists(stale))


if __name__ == "__main__":
    unittest.main()
