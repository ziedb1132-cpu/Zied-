"""Signal-only regression tests: virtual portfolio features must stay removed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zied_quant as zq


def test_paper_has_no_wallet_positions_or_order_ledger(monkeypatch, tmp_path):
    monkeypatch.setitem(zq.CFG, "DIR", str(tmp_path))
    monkeypatch.setitem(zq.CFG, "DB", str(tmp_path / "signals.db"))
    monkeypatch.setattr(zq, "tg", lambda _message: None)

    p = zq.Paper()
    try:
        assert not hasattr(p, "cash")
        assert not hasattr(p, "equity")
        assert not hasattr(p, "place_limit")
        assert not hasattr(p, "pending_orders")
        assert not hasattr(p, "open_trades")
        tables = {r["name"] for r in p.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"signals", "meta", "tracked_signals"} <= tables
        assert not ({"trades", "pending_orders", "experience"} & tables)
    finally:
        p.db.close()
