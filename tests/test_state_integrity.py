import sqlite3
import tempfile
import unittest
from pathlib import Path

from state_integrity import check_state


class StateIntegrityTests(unittest.TestCase):
    def test_missing_database_is_allowed_before_first_run(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(check_state(td)["database"], "not_yet_created")

    def test_missing_required_database_fails_when_required(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                check_state(td, require_db=True)

    def test_empty_database_fails_when_historical_data_is_required(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "paper.db"
            conn = sqlite3.connect(db)
            conn.executescript("""
                CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
                CREATE TABLE signals (id INTEGER PRIMARY KEY);
                CREATE TABLE tracked_signals (
                    id INTEGER PRIMARY KEY, status TEXT NOT NULL
                );
            """)
            conn.commit()
            conn.close()
            with self.assertRaisesRegex(RuntimeError, "no historical signals"):
                check_state(td, require_db=True, require_tracked_data=True)

    def test_valid_paper_database_passes_read_only_check(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "paper.db"
            conn = sqlite3.connect(db)
            conn.executescript("""
                CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
                CREATE TABLE signals (id INTEGER PRIMARY KEY);
                CREATE TABLE tracked_signals (
                    id INTEGER PRIMARY KEY, status TEXT NOT NULL
                );
                INSERT INTO tracked_signals(status) VALUES ('OPEN'), ('CLOSED');
            """)
            conn.commit()
            conn.close()
            report = check_state(td, require_db=True)
            self.assertEqual(report["sqlite_check"], "ok")
            self.assertEqual(report["tracked"], 2)
            self.assertEqual(report["open_tracked"], 1)
            self.assertEqual(report["closed_tracked"], 1)


if __name__ == "__main__":
    unittest.main()
