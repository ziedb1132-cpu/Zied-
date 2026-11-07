#!/usr/bin/env python3
"""Read-only integrity checks for the persistent paper-trading SQLite database."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path


def check_state(state_dir, require_db=False):
    root = Path(state_dir)
    db = root / "paper.db"
    if not db.exists():
        if require_db:
            raise RuntimeError("required paper.db is missing")
        return {"status": "ok", "database": "not_yet_created"}
    if db.stat().st_size <= 0:
        raise RuntimeError("paper.db is empty")
    uri = db.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=15)
    try:
        result = conn.execute("PRAGMA quick_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError("SQLite quick_check failed: " + repr(result))
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"meta", "signals", "tracked_signals"}
        missing = sorted(required - tables)
        if missing:
            raise RuntimeError("paper.db missing required tables: " + ", ".join(missing))
        signals = int(conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0])
        tracked = int(conn.execute("SELECT COUNT(*) FROM tracked_signals").fetchone()[0])
        open_count = int(conn.execute(
            "SELECT COUNT(*) FROM tracked_signals WHERE status='OPEN'").fetchone()[0])
        closed_count = int(conn.execute(
            "SELECT COUNT(*) FROM tracked_signals WHERE status='CLOSED'").fetchone()[0])
        if open_count + closed_count > tracked:
            raise RuntimeError("tracked signal counts are inconsistent")
        return {
            "status": "ok", "database": "paper.db",
            "bytes": db.stat().st_size, "signals": signals,
            "tracked": tracked, "open_tracked": open_count,
            "closed_tracked": closed_count, "tables": len(tables),
            "sqlite_check": "ok",
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dir")
    parser.add_argument("--require-db", action="store_true")
    args = parser.parse_args()
    report = check_state(args.state_dir, require_db=args.require_db)
    print("STATE_INTEGRITY_OK " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STATE_INTEGRITY_ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
