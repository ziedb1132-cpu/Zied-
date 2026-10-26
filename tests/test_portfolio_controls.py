"""Regression tests for paper LIMIT orders and aggregate portfolio risk."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zied_quant as zq


def make_paper(monkeypatch, tmp_path):
    monkeypatch.setitem(zq.CFG, "DIR", str(tmp_path))
    monkeypatch.setitem(zq.CFG, "DB", str(tmp_path / "paper.db"))
    monkeypatch.setattr(zq, "tg", lambda _message: None)
    return zq.Paper()


def test_limit_reserves_cash_and_fills_only_after_order_exists(monkeypatch, tmp_path):
    p = make_paper(monkeypatch, tmp_path)
    order = p.place_limit("AAA-USDT", 99.0, 0.01, 0.7, 1, np.zeros(150), 5e6, 0.001)
    assert order is not None
    assert p.cash() < zq.CFG["START_BALANCE"]
    assert abs(p.equity() - zq.CFG["START_BALANCE"]) < 1e-6
    assert len(p.pending_orders()) == 1
    assert p.open_trades() == []

    created = time.time() - 600
    p.db.execute("UPDATE pending_orders SET created_ts=?, expires_ts=? WHERE id=?",
                 (created, time.time() + 3600, order["id"]))
    p.db.commit()
    candle_ts = int((created + 60) * 1000)
    monkeypatch.setattr(zq, "get_candles", lambda *_args, **_kwargs: pd.DataFrame({
        "ts": np.array([candle_ts], dtype=np.int64), "l": [98.5], "h": [100.0], "c": [99.0]
    }))

    result = p.check_limit_orders()
    assert result == {"filled": 1, "expired": 0}
    assert p.pending_orders() == []
    trades = p.open_trades()
    assert len(trades) == 1
    assert trades[0]["entry"] == 99.0
    # Reserved cash is converted into a position, not deducted a second time.
    assert abs(p.cash() - (zq.CFG["START_BALANCE"] - order["usdt"])) < 1e-6


def test_expired_limit_refunds_reserved_cash(monkeypatch, tmp_path):
    p = make_paper(monkeypatch, tmp_path)
    order = p.place_limit("BBB-USDT", 50.0, 0.01, 0.7, 1, np.zeros(150), 5e6, 0.001)
    assert order is not None
    cash_after_reserve = p.cash()
    p.db.execute("UPDATE pending_orders SET expires_ts=? WHERE id=?",
                 (time.time() - 1, order["id"]))
    p.db.commit()
    monkeypatch.setattr(zq, "get_candles", lambda *_args, **_kwargs: pd.DataFrame({
        "ts": np.array([], dtype=np.int64), "l": [], "h": [], "c": []
    }))

    result = p.check_limit_orders()
    assert result == {"filled": 0, "expired": 1}
    assert p.pending_orders() == []
    assert abs(p.cash() - zq.CFG["START_BALANCE"]) < 1e-6
    assert p.cash() > cash_after_reserve


def test_portfolio_risk_budget_blocks_an_oversized_new_order(monkeypatch, tmp_path):
    p = make_paper(monkeypatch, tmp_path)
    monkeypatch.setitem(zq.CFG, "MAX_PORTFOLIO_RISK", 0.001)
    before = p.cash()
    order = p.place_limit("CCC-USDT", 100.0, 0.01, 0.7, 1, np.zeros(150), 5e6, 0.001)
    assert order is None
    assert p.pending_orders() == []
    assert p.cash() == before
