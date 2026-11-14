"""Offline tests for the candlestick and headline intelligence overlay."""
import json

import pandas as pd

import candle_news_intelligence as cni


def test_bullish_engulfing_detected():
    candles = pd.DataFrame([
        {"o": 105, "h": 106, "l": 99, "c": 100, "vol": 10},
        {"o": 100, "h": 103, "l": 98, "c": 102, "vol": 12},
        {"o": 99, "h": 108, "l": 98, "c": 107, "vol": 18},
    ])
    names = {p["name"] for p in cni.detect_candlestick_patterns(candles)}
    assert "Bullish Engulfing" in names


def test_bearish_engulfing_detected():
    candles = pd.DataFrame([
        {"o": 100, "h": 106, "l": 99, "c": 105, "vol": 10},
        {"o": 105, "h": 107, "l": 102, "c": 103, "vol": 12},
        {"o": 107, "h": 108, "l": 98, "c": 99, "vol": 18},
    ])
    names = {p["name"] for p in cni.detect_candlestick_patterns(candles)}
    assert "Bearish Engulfing" in names


def test_news_sentiment_uses_coin_specific_headlines():
    headlines = [
        {"title": "Solana adoption surges after major partnership", "sentiment": 0.8},
        {"title": "Bitcoin falls as market risk rises", "sentiment": -0.7},
    ]
    sol, sol_items = cni.news_score("SOL-USDT", headlines)
    btc, btc_items = cni.news_score("BTC-USDT", headlines)
    assert sol > 0 and sol_items
    assert btc < 0 and btc_items


def test_empty_candles_are_neutral():
    assert cni.detect_candlestick_patterns(None) == []
    score, patterns = cni.pattern_score(pd.DataFrame())
    assert score == 0.0 and patterns == []


def test_pattern_feature_matrix_is_causal_and_model_ready():
    candles = pd.DataFrame([
        {"o": 105, "h": 106, "l": 99, "c": 100, "vol": 10},
        {"o": 100, "h": 103, "l": 98, "c": 102, "vol": 12},
        {"o": 99, "h": 108, "l": 98, "c": 107, "vol": 18},
        {"o": 107, "h": 109, "l": 106, "c": 108, "vol": 20},
    ], index=pd.date_range("2026-01-01", periods=4, freq="h"))
    matrix = cni.pattern_feature_matrix(candles)
    column = "candle_pat_bullish_engulfing"
    assert column in matrix.columns
    assert matrix.loc[candles.index[0], column] != matrix.loc[candles.index[2], column]
    assert pd.isna(matrix.loc[candles.index[0], column])  # warm-up is unknown, not fabricated
    assert matrix.loc[candles.index[2], column] == 0.86
    assert matrix.loc[candles.index[2], "candle_pat_count"] >= 1
    # Appending a future candle cannot rewrite already-computed historical rows.
    extended = pd.concat([candles, pd.DataFrame([{"o": 108, "h": 110, "l": 107, "c": 109, "vol": 22}],
                                                  index=[pd.Timestamp("2026-01-01 04:00:00")])])
    extended_matrix = cni.pattern_feature_matrix(extended)
    pd.testing.assert_frame_equal(matrix.iloc[:4], extended_matrix.iloc[:4], check_freq=False)



def test_new_pattern_name_is_persisted_for_next_primary_schema(tmp_path, monkeypatch):
    registry = tmp_path / "candle_pattern_registry.json"
    monkeypatch.setattr(cni, "_PATTERN_REGISTRY_PATH", str(registry))
    monkeypatch.setattr(cni, "PATTERN_NAMES", ("Doji",))
    monkeypatch.setattr(
        cni, "PATTERN_FEATURE_NAMES",
        ("candle_pat_doji", "candle_pat_count")
    )

    cni._register_pattern_name("New Research Pattern")

    saved = json.loads(registry.read_text(encoding="utf-8"))
    assert "New Research Pattern" in saved["patterns"]
    assert "candle_pat_new_research_pattern" in saved["feature_names"]

    # A fresh schema load restores the discovered name before model columns freeze.
    monkeypatch.setattr(cni, "PATTERN_NAMES", ("Doji",))
    monkeypatch.setattr(
        cni, "PATTERN_FEATURE_NAMES",
        ("candle_pat_doji", "candle_pat_count")
    )
    names = cni.configure_pattern_registry(str(registry))
    assert "candle_pat_new_research_pattern" in names



def test_dynamic_sequence_feature_is_causal(monkeypatch):
    definition = {
        "feature_name": "candle_seq_test_3",
        "sequence": ["222", "222", "222"],
        "direction": "long",
        "status": "approved",
    }
    monkeypatch.setattr(cni, "_DISCOVERED_PATTERN_DEFS", [definition])
    monkeypatch.setattr(cni, "PATTERN_FEATURE_NAMES", tuple(cni.PATTERN_FEATURE_NAMES) + ("candle_seq_test_3",))
    candles = pd.DataFrame([
        {"o": 10, "h": 11, "l": 9, "c": 11, "vol": 1},
        {"o": 10, "h": 11, "l": 9, "c": 11, "vol": 1},
        {"o": 10, "h": 11, "l": 9, "c": 11, "vol": 1},
        {"o": 10, "h": 11, "l": 9, "c": 11, "vol": 1},
    ], index=pd.date_range("2026-01-01", periods=4, freq="h"))
    matrix = cni.pattern_feature_matrix(candles)
    assert "candle_seq_test_3" in matrix.columns
    assert matrix.iloc[2]["candle_seq_test_3"] == 1.0
    assert matrix.iloc[3]["candle_seq_test_3"] == 1.0


def test_sequence_discovery_handles_empty_universe():
    result = cni.discover_sequence_patterns({})
    assert result["candidate_sequences"] == 0
    assert result["approved"] == 0
    assert result["horizon_bars"] == 12
