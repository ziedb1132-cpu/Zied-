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
                    feats TEXT, labeled INTEGER, label INTEGER, ret REAL,
                    feature_names TEXT, feature_count INTEGER
                )
            """)
            db.execute("""
                CREATE TABLE tracked_signals(
                    signal_id INTEGER, status TEXT, closed_ts REAL
                )
            """)

    def tearDown(self):
        self.tmp.cleanup()

    def _insert_samples(self, winners=50, losers=50):
        rows = []
        for i in range(winners):
            x = np.zeros(300, dtype=float)
            x[0] = 2.0 + (i % 5) * 0.01
            values = x.tolist()
            if i == 0:
                values[1] = None  # unknown is retained; only this feature statistic skips it
            rows.append((i + 1, i + 1, i + 1, f"WIN{i}-USDT",
                         json.dumps(values), 1, 1, 0.02,
                         json.dumps([f"feature_{j}" for j in range(300)]), 300))
        for i in range(losers):
            x = np.zeros(300, dtype=float)
            x[0] = -2.0 - (i % 5) * 0.01
            sid = winners + i + 1
            rows.append((sid, sid, sid, f"LOSS{i}-USDT",
                         json.dumps(x.tolist()), 1, 0, -0.02,
                         json.dumps([f"feature_{j}" for j in range(300)]), 300))
        with sqlite3.connect(self.db_path) as db:
            db.executemany(
                "INSERT INTO signals(id, ts, feature_ts, inst, feats, labeled, label, ret, feature_names, feature_count) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)", rows
            )
            db.executemany(
                "INSERT INTO tracked_signals(signal_id, status, closed_ts) VALUES(?,?,?)",
                [(row[0], "CLOSED", row[1]) for row in rows]
            )

    def test_reads_only_closed_signals_and_writes_patterns(self):
        self._insert_samples()
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "patterns_updated")
        self.assertEqual(result["winning_samples"], 50)
        self.assertEqual(result["losing_samples"], 50)
        with open(os.path.join(self.tmp.name, "winning_patterns.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["feature_count"], 300)
        self.assertEqual(data["source"], "closed tracked signals only; outcomes and net returns from later-candle evaluation")
        self.assertTrue(data["top_recurring_features"])
        self.assertEqual(data["top_recurring_features"][0]["feature"], "feature_0")

    def test_waits_until_100_total_closed_samples(self):
        self._insert_samples(winners=49, losers=50)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_closed_outcomes")
        self.assertEqual(result["winning_samples"], 49)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "winning_patterns.json")))

    def test_starts_at_100_even_when_one_class_has_only_one_real_sample(self):
        self._insert_samples(winners=99, losers=1)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "patterns_updated")
        self.assertEqual(result["winning_samples"], 99)
        self.assertEqual(result["losing_samples"], 1)

    def test_waits_until_both_outcome_classes_are_present(self):
        self._insert_samples(winners=100, losers=0)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_closed_outcomes")
        self.assertEqual(result["losing_samples"], 0)

    def test_removes_stale_patterns_when_history_is_reset(self):
        stale = os.path.join(self.tmp.name, "winning_patterns.json")
        with open(stale, "w", encoding="utf-8") as fh:
            json.dump({"winning_samples": 50, "losing_samples": 50, "total_usable_samples": 100}, fh)
        result = dle.discover_winning_patterns(self.zq, self.paper)
        self.assertEqual(result["status"], "waiting_for_closed_outcomes")
        self.assertFalse(os.path.exists(stale))


if __name__ == "__main__":
    unittest.main()
