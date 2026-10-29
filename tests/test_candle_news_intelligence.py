"""Offline tests for the candlestick and headline intelligence overlay."""
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
