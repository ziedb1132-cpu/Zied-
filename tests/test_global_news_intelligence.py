"""Offline checks for the 50-source global news intelligence layer."""
import time

import global_news_intelligence as gni


def test_exactly_50_unique_global_sources_are_configured():
    assert len(gni.NEWS_FEEDS) == 50
    assert len({row[0] for row in gni.NEWS_FEEDS}) == 50
    assert len({row[1] for row in gni.NEWS_FEEDS}) == 50


def test_news_features_are_coin_specific_and_bounded():
    now = time.time()
    articles = [
        {
            "title": "Bitcoin ETF inflows surge after institutional adoption",
            "summary": "Bitcoin market sees new investment growth",
            "published_ts": now,
            "sentiment": 0.9,
            "credibility": 1.0,
            "event_risk": 0.0,
            "urgency": 0.0,
            "source": "Example Financial News",
        },
        {
            "title": "Solana network outage and exploit investigation",
            "summary": "Solana users face service risk",
            "published_ts": now,
            "sentiment": -0.8,
            "credibility": 0.8,
            "event_risk": 1.0,
            "urgency": 0.5,
            "source": "Example Crypto News",
        },
    ]
    btc, btc_items = gni.news_features("BTC-USDT", articles)
    sol, sol_items = gni.news_features("SOL-USDT", articles)
    assert len(btc) == len(gni.MODEL_FEATURES) == 10
    assert len(sol) == 10
    assert btc[0] > 0 and btc_items
    assert sol[0] < 0 and sol_items
    assert btc[-1] == sol[-1] == 1.0
    assert all(-1.0 <= value <= 1.0 for value in btc + sol)


def test_missing_news_is_explicitly_unavailable_not_fabricated():
    values, related = gni.news_features("BTC-USDT", [])
    assert values == [0.0] * 10
    assert related == []


def test_news_sentiment_supports_english_and_arabic_terms():
    positive, _, _ = gni._sentiment("Bitcoin adoption surge and ارتفاع")
    negative, _, _ = gni._sentiment("market hack and انهيار")
    assert positive > 0
    assert negative < 0
