#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""50-source global news intelligence and outcome-linked learning for Zied Quant.

News is captured only at signal creation and is paired only with that signal's
subsequent realized closed outcome. Missing news is represented as unavailable,
never fabricated. The auxiliary model is independently validated and only adds
a small bounded ranking adjustment; it cannot bypass the primary AI/risk gates.
"""
import concurrent.futures
import datetime as dt
import email.utils
import html
import json
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import numpy as np
import requests

CACHE_SECONDS = 900
MAX_ARTICLES = 500
MAX_SIGNAL_ARTICLES = 20
MODEL_FEATURES = (
    "sentiment", "bullish_ratio", "bearish_ratio", "source_credibility",
    "source_diversity", "risk_ratio", "urgency", "freshness", "coverage", "available"
)

# These ten columns are appended to the primary Zied Quant model schema.
# The final availability flag distinguishes missing news from neutral sentiment.
NEWS_FEATURE_COLS = tuple("global_news_" + name for name in MODEL_FEATURES)

# Curated publisher feeds and primary institutional sources. A failed source is
# isolated and reported; it never blocks scanning or causes invented headlines.
NEWS_FEEDS = (
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "crypto", 1.00),
    ("Cointelegraph", "https://cointelegraph.com/rss", "crypto", 0.85),
    ("The Block", "https://www.theblock.co/rss.xml", "crypto", 1.00),
    ("Decrypt", "https://decrypt.co/feed", "crypto", 0.90),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/feed", "crypto", 0.85),
    ("DL News", "https://www.dlnews.com/arc/outboundfeeds/rss/", "crypto", 0.90),
    ("CryptoSlate", "https://cryptoslate.com/feed/", "crypto", 0.80),
    ("U.Today", "https://u.today/rss", "crypto", 0.75),
    ("CryptoPotato", "https://cryptopotato.com/feed/", "crypto", 0.75),
    ("BeInCrypto", "https://beincrypto.com/feed/", "crypto", 0.80),
    ("Bitcoin.com News", "https://news.bitcoin.com/feed/", "crypto", 0.85),
    ("Cryptopolitan", "https://www.cryptopolitan.com/feed/", "crypto", 0.70),
    ("The Defiant", "https://thedefiant.io/feed/", "defi", 0.85),
    ("Protos", "https://protos.com/feed/", "crypto", 0.85),
    ("CoinJournal", "https://coinjournal.net/news/feed/", "crypto", 0.70),
    ("NewsBTC", "https://www.newsbtc.com/feed/", "crypto", 0.70),
    ("AMBCrypto", "https://ambcrypto.com/feed/", "crypto", 0.70),
    ("CoinGape", "https://coingape.com/feed/", "crypto", 0.70),
    ("Watcher Guru", "https://watcher.guru/news/feed", "crypto", 0.65),
    ("Crypto Briefing", "https://cryptobriefing.com/feed/", "crypto", 0.80),
    ("Messari", "https://news.google.com/rss/search?q=site%3Amessari.io+crypto&hl=en-US&gl=US&ceid=US%3Aen", "research", 0.90),
    ("Glassnode Insights", "https://insights.glassnode.com/rss/", "research", 0.95),
    ("Santiment", "https://santiment.net/blog/feed/", "research", 0.85),
    ("Coin Center", "https://www.coincenter.org/feed/", "policy", 0.95),
    ("Ethereum Foundation", "https://blog.ethereum.org/en/feed.xml", "protocol", 1.00),
    ("Solana News", "https://solana.com/news/rss.xml", "protocol", 0.95),
    ("Binance Blog", "https://www.binance.com/en/blog/rss", "exchange", 0.90),
    ("Coinbase Blog", "https://www.coinbase.com/blog/rss.xml", "exchange", 0.95),
    ("Kraken Blog", "https://blog.kraken.com/feed", "exchange", 0.90),
    ("OKX Learn", "https://www.okx.com/learn/rss.xml", "exchange", 0.90),
    ("Bloomberg Crypto", "https://feeds.bloomberg.com/crypto/news.rss", "macro", 1.00),
    ("Financial Times Crypto", "https://www.ft.com/crypto?format=rss", "macro", 1.00),
    ("CNBC Crypto", "https://www.cnbc.com/id/100727362/device/rss/rss.html", "macro", 0.95),
    ("Yahoo Finance Crypto", "https://finance.yahoo.com/rss/cryptocurrency", "macro", 0.85),
    ("WSJ Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "macro", 1.00),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/", "macro", 0.90),
    ("CNBC Economy", "https://www.cnbc.com/id/20910258/device/rss/rss.html", "macro", 0.95),
    ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", "central_bank", 1.00),
    ("European Central Bank", "https://www.ecb.europa.eu/rss/press.html", "central_bank", 1.00),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", "world", 0.95),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", "world", 0.90),
    ("Deutsche Welle", "https://rss.dw.com/xml/rss-en-all", "world", 0.90),
    ("Euronews", "https://www.euronews.com/rss", "world", 0.85),
    ("The Guardian World", "https://www.theguardian.com/world/rss", "world", 0.90),
    ("NPR Economy", "https://feeds.npr.org/1017/rss.xml", "macro", 0.90),
    ("TechCrunch Crypto", "https://techcrunch.com/category/cryptocurrency/feed/", "technology", 0.90),
    ("Wired Security", "https://www.wired.com/feed/category/security/latest/rss", "technology", 0.90),
    ("US SEC", "https://www.sec.gov/news/pressreleases.rss", "regulation", 1.00),
    ("Google News Reuters", "https://news.google.com/rss/search?q=site%3Areuters.com+bitcoin+OR+crypto+OR+markets&hl=en-US&gl=US&ceid=US%3Aen", "world", 0.95),
    ("Google News AP", "https://news.google.com/rss/search?q=site%3Aapnews.com+cryptocurrency+OR+bitcoin+OR+economy&hl=en-US&gl=US&ceid=US%3Aen", "world", 0.95),
)
assert len(NEWS_FEEDS) == 50

POSITIVE = {
    "surge", "surges", "surged", "rally", "rallies", "bullish", "breakout",
    "approval", "approved", "adoption", "partnership", "launch", "upgrade",
    "inflow", "inflows", "record", "growth", "gains", "gain", "soars", "soar",
    "rebound", "recovery", "accumulate", "accumulation", "positive", "beat",
    "beats", "investment", "investments", "increased", "cuts", "eases",
    "خفض", "ارتفاع", "موافقة", "نمو", "انتعاش", "إيجابي", "إيجابية"
}
NEGATIVE = {
    "crash", "crashes", "crashed", "bearish", "hack", "hacked", "exploit",
    "lawsuit", "sued", "ban", "banned", "probe", "investigation", "outflow",
    "outflows", "selloff", "sell-off", "plunge", "plunges", "falls", "fall",
    "drop", "drops", "decline", "declines", "liquidation", "liquidations",
    "fraud", "scam", "delisting", "delisted", "warning", "risk", "risks",
    "negative", "miss", "misses", "rejected", "breach", "bankruptcy", "halt",
    "lawsuit", "cuts", "inflation", "raises rates",
    "اختراق", "احتيال", "حظر", "انهيار", "خسائر", "هبوط", "تحقيق", "سلبي", "سلبية"
}
RISK_TERMS = {
    "hack", "hacked", "exploit", "breach", "bankruptcy", "fraud", "scam",
    "liquidation", "ban", "banned", "lawsuit", "insolvency", "default",
    "outage", "attack", "stolen", "investigation", "sanction", "war",
    "اختراق", "احتيال", "إفلاس", "حظر", "هجوم", "حرب", "عقوبات"
}
URGENCY_TERMS = {"breaking", "urgent", "alert", "emergency", "halt", "hack", "exploit", "ban", "crash", "ان عاجل", "عاجل"}
COIN_ALIASES = {
    "BTC": ("bitcoin", "btc"), "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"), "XRP": ("ripple", "xrp"),
    "BNB": ("binance coin", "bnb"), "DOGE": ("dogecoin", "doge"),
    "ADA": ("cardano", "ada"), "TON": ("toncoin", "ton"),
    "TRX": ("tron", "trx"), "LINK": ("chainlink", "link"),
    "AVAX": ("avalanche", "avax"), "DOT": ("polkadot", "dot"),
    "LTC": ("litecoin", "ltc"), "BCH": ("bitcoin cash", "bch"),
    "SUI": ("sui",), "APT": ("aptos", "apt"), "ARB": ("arbitrum", "arb"),
    "OP": ("optimism",), "NEAR": ("near protocol",), "ATOM": ("cosmos",),
    "UNI": ("uniswap",), "AAVE": ("aave",), "SHIB": ("shiba inu",)
}
GENERAL_TERMS = ("crypto", "cryptocurrency", "bitcoin", "ethereum", "digital asset",
                 "digital assets", "crypto market", "sec", "etf", "stablecoin", "blockchain",
                 "interest rate", "federal reserve", "inflation", "central bank")

_LOCK = threading.RLock()
_STATE = {"fetched_at": 0.0, "articles": [], "sources": [], "features": {},
          "model": None, "last_model_fit": 0}
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ZiedQuantNewsResearch/2.0 (+https://github.com/ziedb1132-cpu/Zied-)"})


def _plain(value):
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _timestamp(value):
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    except Exception:
        pass
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    except Exception:
        return 0.0


def _sentiment(text):
    s = re.sub(r"[^a-z0-9$\u0600-\u06ff-]+", " ", str(text).lower())
    tokens = set(s.split())
    pos = sum(1 for word in POSITIVE if word in tokens or word in s)
    neg = sum(1 for word in NEGATIVE if word in tokens or word in s)
    if not pos and not neg:
        return 0.0, 0, 0
    score = float(np.clip((pos - neg) / max(1.0, (pos + neg) ** 0.5), -1.0, 1.0))
    return score, pos, neg


def _parse_feed(feed):
    name, url, category, credibility = feed
    response = _SESSION.get(url, timeout=(4, 8))
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower()
        if tag not in ("item", "entry"):
            continue
        fields = {}
        for child in list(node):
            key = child.tag.rsplit("}", 1)[-1].lower()
            if key == "link":
                link = child.attrib.get("href") or (child.text or "")
            else:
                link = None
            if key not in fields or fields[key] in (None, ""):
                fields[key] = _plain(child.text or "") if key != "link" else link
        title = fields.get("title", "")
        url_value = fields.get("link", "")
        if not title:
            continue
        summary = fields.get("description") or fields.get("summary") or fields.get("content") or ""
        published = (fields.get("pubdate") or fields.get("published") or fields.get("updated")
                     or fields.get("date") or fields.get("dc:date") or "")
        ts = _timestamp(published)
        score, pos, neg = _sentiment(title + " " + summary)
        text_value = (title + " " + summary).lower()
        risk_hits = sum(1 for word in RISK_TERMS if word in text_value)
        urgency_hits = sum(1 for word in URGENCY_TERMS if word in text_value)
        direction = "positive" if score > 0.12 else "negative" if score < -0.12 else "neutral"
        confidence = min(0.98, 0.30 + 0.12 * min(pos + neg, 4) + 0.35 * credibility)
        importance = min(100.0, abs(score) * 48 + min(risk_hits, 3) * 12
                         + min(urgency_hits, 2) * 8 + credibility * 12)
        coins = []
        for symbol, aliases in COIN_ALIASES.items():
            if any(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text_value)
                   for alias in aliases):
                coins.append(symbol)
        rows.append({
            "title": title[:400], "summary": summary[:900], "link": str(url_value)[:700],
            "published": published[:100], "published_ts": ts, "source": name,
            "category": category, "credibility": float(credibility),
            "sentiment": score, "direction": direction, "confidence": round(confidence, 3),
            "impact_score": round(importance, 1), "risk_level": "high" if risk_hits >= 2 else "medium" if risk_hits else "low",
            "coins": coins, "event_risk": min(1.0, risk_hits / 3.0),
            "urgency": min(1.0, urgency_hits / 2.0)
        })
    return rows


def fetch_news(force=False):
    now = time.time()
    with _LOCK:
        if not force and now - _STATE["fetched_at"] < CACHE_SECONDS and _STATE["articles"]:
            return list(_STATE["articles"])
    collected, source_status = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_parse_feed, feed): feed for feed in NEWS_FEEDS}
        for future in concurrent.futures.as_completed(futures):
            feed = futures[future]
            try:
                items = future.result()
                collected.extend(items)
                source_status.append({"name": feed[0], "category": feed[2], "status": "online",
                                      "articles": len(items), "credibility": feed[3]})
            except Exception as exc:
                source_status.append({"name": feed[0], "category": feed[2], "status": "offline",
                                      "articles": 0, "credibility": feed[3],
                                      "error": str(exc)[:180]})
    unique, output = set(), []
    for item in sorted(collected, key=lambda x: x.get("published_ts", 0), reverse=True):
        url = str(item.get("link") or "").strip().lower()
        title_key = re.sub(r"[^a-z0-9]+", " ", item.get("title", "").lower()).strip()
        key = url or title_key
        if not key or key in unique:
            continue
        unique.add(key)
        output.append(item)
    output = output[:MAX_ARTICLES]
    source_status.sort(key=lambda x: (x["status"] != "online", x["name"]))
    with _LOCK:
        _STATE["articles"] = output
        _STATE["sources"] = source_status
        _STATE["fetched_at"] = now
    return list(output)


def news_features(inst, articles=None):
    articles = fetch_news() if articles is None else articles
    base = str(inst).split("-")[0].upper()
    aliases = COIN_ALIASES.get(base, (base.lower(),))
    related = []
    for item in articles:
        text_value = (str(item.get("title", "")) + " " + str(item.get("summary", ""))).lower()
        specific = any(re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", text_value)
                       for a in aliases)
        broad = any(term in text_value for term in GENERAL_TERMS)
        if specific:
            related.append((item, 1.0))
        elif broad:
            related.append((item, 0.20))
    related = sorted(related, key=lambda x: x[0].get("published_ts", 0), reverse=True)[:MAX_SIGNAL_ARTICLES]
    if not related:
        return [0.0] * (len(MODEL_FEATURES) - 1) + [0.0], []
    now = time.time()
    weights, sentiments = [], []
    for item, relevance in related:
        published_ts = float(item.get("published_ts") or 0.0)
        freshness = (math.exp(-max(0.0, now - published_ts) / (36 * 3600))
                     if published_ts > 0 else 0.0)
        # Unknown publication times receive no freshness credit, but may still contribute by source credibility.
        weight = float(item.get("credibility", 0.5)) * relevance * (freshness if published_ts > 0 else 1.0)
        weights.append(weight)
        sentiments.append(float(item.get("sentiment", 0.0)))
    total = max(sum(weights), 1e-9)
    weighted_sentiment = float(np.clip(sum(s*w for s, w in zip(sentiments, weights))/total, -1, 1))
    positive = sum(w for (item, _), w in zip(related, weights) if item.get("sentiment", 0) > 0.12) / total
    negative = sum(w for (item, _), w in zip(related, weights) if item.get("sentiment", 0) < -0.12) / total
    credibility = float(np.mean([x[0].get("credibility", 0.5) for x in related]))
    sources = len({x[0].get("source") for x in related})
    risk = float(np.average([x[0].get("event_risk", 0.0) for x in related], weights=weights))
    urgency = float(np.average([x[0].get("urgency", 0.0) for x in related], weights=weights))
    freshness = float(np.average([math.exp(-max(0, now-float(x[0].get("published_ts") or now))/(36*3600))
                                  for x in related], weights=weights))
    coverage = min(1.0, len(related) / 5.0)
    vector = [weighted_sentiment, positive, negative, credibility,
              min(1.0, sources/8.0), risk, urgency, freshness, coverage, 1.0]
    return [float(np.clip(x, -1, 1)) for x in vector], [x[0] for x in related[:5]]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _fit_logistic(X, y, steps=450):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-6] = 1.0
    Z = np.clip((X-mean)/scale, -6, 6)
    w = np.zeros(Z.shape[1], dtype=np.float64)
    b = 0.0
    for step in range(steps):
        p = _sigmoid(Z @ w + b)
        err = p-y
        rate = 0.08 / (1.0 + step/250.0)
        w -= rate * ((Z.T @ err)/len(y) + 0.025*w)
        b -= rate * float(err.mean())
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "weights": w.tolist(), "bias": float(b)}


def _predict(vector, model):
    if not model or not model.get("ready"):
        return None
    x = np.asarray(vector, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if x.size != len(weights) or not np.isfinite(x).all():
        return None
    z = np.clip((x-mean)/scale, -6, 6)
    return float(_sigmoid(np.dot(z, weights) + float(model.get("bias", 0.0))))


def _auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return 0.5
    return float(np.mean([(a > b) + 0.5*(a == b) for a in pos for b in neg]))


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    os.replace(tmp, path)


def _state_dir(zq):
    return zq.CFG["DIR"]


def _load_model(zq):
    path = os.path.join(_state_dir(zq), "news_model.json")
    try:
        with open(path, encoding="utf-8") as handle:
            model = json.load(handle)
        if model.get("feature_names") == list(MODEL_FEATURES):
            _STATE["model"] = model
    except (OSError, ValueError, TypeError):
        pass


def _write_intelligence(zq, articles):
    sources = list(_STATE.get("sources", []))
    now = time.time()
    stale_cache = False
    if not articles:
        # Preserve the last successful report for display during a temporary
        # publisher outage. Stale items are never used as fresh signal features.
        previous_path = os.path.join(_state_dir(zq), "news_intelligence.json")
        try:
            with open(previous_path, encoding="utf-8") as handle:
                previous = json.load(handle)
            previous_articles = previous.get("articles", [])
            if isinstance(previous_articles, list) and previous_articles:
                articles = previous_articles
                stale_cache = True
        except (OSError, ValueError, TypeError):
            pass
    online = sum(1 for s in sources if s.get("status") == "online")
    fresh = [x for x in articles if now - float(x.get("published_ts") or 0) <= 48*3600]
    sentiment_values = [float(x.get("sentiment", 0)) for x in fresh]
    average = float(np.mean(sentiment_values)) if sentiment_values else 0.0
    report = {
        "schema": 2, "updated_at": now, "stale_cache": stale_cache, "source_count": len(NEWS_FEEDS),
        "online_sources": online, "offline_sources": len(NEWS_FEEDS)-online,
        "article_count": len(articles), "fresh_48h": len(fresh),
        "market_sentiment": round(average, 4),
        "market_direction": "positive" if average > 0.12 else "negative" if average < -0.12 else "neutral",
        "sources": sources, "articles": articles[:50],
        "model": {
            "ready": bool((_STATE.get("model") or {}).get("ready")),
            "samples": int((_STATE.get("model") or {}).get("samples", 0)),
            "validation_auc": (_STATE.get("model") or {}).get("validation_auc"),
            "last_trained_at": (_STATE.get("model") or {}).get("trained_at"),
            "reason": (_STATE.get("model") or {}).get("reason", "awaiting_news_outcomes"),
            "diagnostics": (_STATE.get("model") or {}).get("diagnostics", {}),
            "features": list(MODEL_FEATURES),
            "role": "bounded auxiliary news intelligence; primary AI and risk gates unchanged"
        },
        "method": "RSS/Atom headline and summary analysis; source-weighted lexicon; timestamp decay; exact URL/title deduplication. No missing facts are inferred.",
        "cache_note": "Displaying the last successful feed snapshot because all current feeds returned no items." if stale_cache else None
    }
    _write_json(os.path.join(_state_dir(zq), "news_intelligence.json"), report)


def _persist_signal_snapshot(zq, paper, signal_id, inst):
    articles = fetch_news()
    vector, related = news_features(inst, articles)
    with paper.lock:
        paper.db.execute(
            "CREATE TABLE IF NOT EXISTS news_signal_memory ("
            "signal_id INTEGER PRIMARY KEY, inst TEXT NOT NULL, captured_ts REAL NOT NULL, "
            "features TEXT NOT NULL, article_count INTEGER NOT NULL, sources TEXT NOT NULL)"
        )
        paper.db.execute(
            "INSERT OR REPLACE INTO news_signal_memory(signal_id,inst,captured_ts,features,article_count,sources) "
            "VALUES(?,?,?,?,?,?)",
            (int(signal_id), str(inst), time.time(),
             json.dumps(vector, separators=(",", ":"), allow_nan=False), len(related),
             json.dumps(sorted({x.get("source", "") for x in related}), ensure_ascii=False)))
        paper.db.commit()
    _STATE["features"][str(inst)] = vector


def _news_learning_diagnostics(paper):
    """Count news snapshots against the actual signal foreign key, without inventing history."""
    with paper.lock:
        paper.db.execute(
            "CREATE TABLE IF NOT EXISTS news_signal_memory ("
            "signal_id INTEGER PRIMARY KEY, inst TEXT NOT NULL, captured_ts REAL NOT NULL, "
            "features TEXT NOT NULL, article_count INTEGER NOT NULL, sources TEXT NOT NULL)"
        )
        row = paper.db.execute(
            "SELECT COUNT(*) AS closed_total, "
            "SUM(CASE WHEN n.signal_id IS NOT NULL THEN 1 ELSE 0 END) AS closed_with_snapshot, "
            "SUM(CASE WHEN n.signal_id IS NOT NULL AND n.article_count>0 THEN 1 ELSE 0 END) AS closed_with_articles, "
            "SUM(CASE WHEN n.signal_id IS NOT NULL AND n.article_count=0 THEN 1 ELSE 0 END) AS closed_without_articles, "
            "SUM(CASE WHEN n.signal_id IS NULL THEN 1 ELSE 0 END) AS closed_without_snapshot "
            "FROM tracked_signals t LEFT JOIN news_signal_memory n ON n.signal_id=t.signal_id "
            "WHERE t.status='CLOSED'"
        ).fetchone()
        snapshots = paper.db.execute(
            "SELECT COUNT(*) FROM news_signal_memory"
        ).fetchone()[0]
        open_snapshots = paper.db.execute(
            "SELECT COUNT(*) FROM tracked_signals t JOIN news_signal_memory n "
            "ON n.signal_id=t.signal_id WHERE t.status='OPEN'"
        ).fetchone()[0]
    return {
        "closed_total": int(row["closed_total"] or 0),
        "closed_with_snapshot": int(row["closed_with_snapshot"] or 0),
        "closed_with_articles": int(row["closed_with_articles"] or 0),
        "closed_without_articles": int(row["closed_without_articles"] or 0),
        "closed_without_snapshot": int(row["closed_without_snapshot"] or 0),
        "snapshots_total": int(snapshots or 0),
        "open_with_snapshot": int(open_snapshots or 0)
    }


def _train_news_model(zq, paper, force=False):
    diagnostics = _news_learning_diagnostics(paper)
    with paper.lock:
        rows = paper.db.execute(
            "SELECT n.features,t.ret,t.closed_ts FROM news_signal_memory n "
            "JOIN tracked_signals t ON t.signal_id=n.signal_id "
            "WHERE t.status='CLOSED' AND t.ret IS NOT NULL AND t.ret != 0 "
            "AND n.article_count > 0 ORDER BY t.closed_ts,n.signal_id"
        ).fetchall()
    X, y = [], []
    for row in rows:
        try:
            values = json.loads(row["features"])
            ret = float(row["ret"])
            if len(values) != len(MODEL_FEATURES) or not np.isfinite(values).all() or not np.isfinite(ret):
                continue
            X.append(values)
            y.append(1.0 if ret > 0 else 0.0)
        except (TypeError, ValueError, KeyError):
            continue
    X, y = np.asarray(X, dtype=np.float64).reshape((-1, len(MODEL_FEATURES))), np.asarray(y, dtype=np.float64)
    n = len(y)
    old = _STATE.get("model") or {}
    if not force and old.get("ready") and n - int(old.get("samples", 0)) < 10:
        updated = dict(old)
        updated["diagnostics"] = diagnostics
        _STATE["model"] = updated
        _write_json(os.path.join(_state_dir(zq), "news_model.json"), updated)
        return updated
    reason = ("no_closed_news_linked_signals" if diagnostics["closed_with_articles"] == 0
              else "no_valid_news_training_samples")
    result = {"ready": False, "samples": n, "reason": reason,
              "feature_names": list(MODEL_FEATURES), "trained_at": time.time(),
              "diagnostics": diagnostics}
    if n:
        result["reason"] = "insufficient_or_unvalidated_news_samples"
    if n >= 60 and min(int((y == 1).sum()), int((y == 0).sum())) >= 10:
        split = max(40, int(n * 0.8))
        if n - split >= 15 and min(int((y[split:] == 1).sum()), int((y[split:] == 0).sum())) >= 3:
            validation = _fit_logistic(X[:split], y[:split])
            val_z = np.clip((X[split:] - np.asarray(validation["mean"])) /
                            np.asarray(validation["scale"]), -6, 6)
            val_p = _sigmoid(val_z @ np.asarray(validation["weights"]) + validation["bias"])
            auc = _auc(y[split:], val_p)
            result.update({"validation_auc": round(auc, 4), "validation_samples": int(n-split)})
            if auc >= 0.52:
                result.update(_fit_logistic(X, y))
                result.update({"ready": True, "reason": "chronological_holdout_passed",
                               "validation_auc": round(auc, 4)})
            else:
                result["reason"] = "chronological_holdout_below_baseline"
    _STATE["model"] = result
    _STATE["last_model_fit"] = time.time()
    _write_json(os.path.join(_state_dir(zq), "news_model.json"), result)
    zq.log.info("global news learner: ready=%s samples=%d linked_closed=%d snapshots=%d auc=%s reason=%s",
                result.get("ready"), n, diagnostics["closed_with_articles"],
                diagnostics["snapshots_total"], result.get("validation_auc"), result.get("reason"))
    return result


def install(zq):
    """Install global news aggregation, primary-model features and outcome-linked learning."""
    _load_model(zq)

    # Extend the active primary model schema after deep_learning_evolution has
    # installed its full technical/research feature library. Historical candle
    # rows receive NaN for news: current headlines are never backfilled into the past.
    existing = list(zq.FEATURE_COLS)
    missing = [name for name in NEWS_FEATURE_COLS if name not in existing]
    if missing:
        old_aligned_matrix = zq.aligned_matrix
        zq.FEATURE_COLS = existing + list(NEWS_FEATURE_COLS)
        zq.N_FEATURES = len(zq.FEATURE_COLS)
        zq.CFG["REQUIRED_SIGNAL_FEATURES"] = zq.N_FEATURES

        def aligned_matrix_with_unknown_news(frames, btc):
            matrix = old_aligned_matrix(frames, btc)
            for name in NEWS_FEATURE_COLS:
                matrix[name] = np.nan
            return matrix.reindex(columns=zq.FEATURE_COLS)

        zq.aligned_matrix = aligned_matrix_with_unknown_news

        # Keep prior closed outcomes usable without inventing historical news:
        # zero-valued news fields plus available=0 explicitly mean "not recorded".
        old_rows_to_ds = zq._rows_to_ds
        legacy_feature_count = zq.N_FEATURES - len(NEWS_FEATURE_COLS)

        def rows_to_ds_with_legacy_news(rows, label_key="label", ret_key="ret"):
            adapted = []
            for row in rows:
                try:
                    item = dict(row)
                    values = json.loads(item["feats"])
                    if isinstance(values, list) and len(values) == legacy_feature_count:
                        item["feats"] = json.dumps(
                            values + [0.0] * len(NEWS_FEATURE_COLS),
                            separators=(",", ":"), allow_nan=False
                        )
                    adapted.append(item)
                except (TypeError, ValueError, KeyError):
                    adapted.append(row)
            return old_rows_to_ds(adapted, label_key, ret_key)

        zq._rows_to_ds = rows_to_ds_with_legacy_news

        # Invalidate only feature-schema-dependent caches/models via the existing
        # schema-version mechanism; trading rules and risk settings are untouched.
        import hashlib
        schema = "|".join(zq.FEATURE_COLS).encode("utf-8")
        zq.DATA_VER = 4 + int(hashlib.sha256(schema).hexdigest()[:8], 16)
        zq.log.info("primary model news features installed: added=%d total_features=%d schema=%d",
                    len(NEWS_FEATURE_COLS), zq.N_FEATURES, zq.DATA_VER)
    elif zq.N_FEATURES != len(zq.FEATURE_COLS):
        raise RuntimeError("Primary model news feature schema is inconsistent")

    old_rank = zq.rank_universe

    def rank_with_global_news(X, vol24, *args, **kwargs):
        if X is None or X.empty:
            return old_rank(X, vol24, *args, **kwargs)
        try:
            articles = fetch_news()
        except Exception as exc:
            zq.log.warning("global news fetch unavailable; primary news features marked unknown: %s", exc)
            articles = []

        # Populate the exact same ten news columns consumed by training/inference.
        # Every coin gets an explicit availability value, including coins with no
        # relevant headlines or temporary feed outages.
        feature_map = {}
        vectors = []
        for inst in X.index:
            try:
                vector, _ = news_features(inst, articles)
            except Exception as exc:
                zq.log.debug("news feature extraction unavailable for %s: %s", inst, exc)
                vector = [0.0] * len(NEWS_FEATURE_COLS)
            vector = np.asarray(vector, dtype=np.float64)
            if vector.shape != (len(NEWS_FEATURE_COLS),) or not np.isfinite(vector).all():
                vector = np.zeros(len(NEWS_FEATURE_COLS), dtype=np.float64)
            feature_map[str(inst)] = vector.tolist()
            vectors.append(vector)
        X[list(NEWS_FEATURE_COLS)] = np.vstack(vectors)

        base = old_rank(X, vol24, *args, **kwargs)
        try:
            adjusted = base.copy()
            model = _STATE.get("model")
            for inst in list(base.head(100).index):
                vector = feature_map.get(str(inst), [0.0] * len(NEWS_FEATURE_COLS))
                probability = _predict(vector, model)
                if probability is not None:
                    # Existing bounded news-ranking adjustment is preserved.
                    adjusted.loc[inst] = float(base.loc[inst]) + 0.10 * (probability - 0.5) * 2.0
            _STATE["features"] = feature_map
            _write_intelligence(zq, articles)
            zq.log.info("global news intelligence: sources=%d online=%d articles=%d learner=%s primary_features=%d",
                        len(NEWS_FEEDS), sum(s.get("status") == "online" for s in _STATE["sources"]),
                        len(articles), "ACTIVE" if (model or {}).get("ready") else "COLLECTING",
                        len(NEWS_FEATURE_COLS))
            return adjusted.sort_values(ascending=False)
        except Exception as exc:
            # Do not remove the primary-model feature columns on a reporting error.
            zq.log.warning("global news ranking adjustment skipped safely: %s", exc)
            return base

    zq.rank_universe = rank_with_global_news

    old_create = zq.Paper.create_tracked_signal
    def create_with_news_snapshot(self, row):
        signal = old_create(self, row)
        if signal:
            try:
                _persist_signal_snapshot(zq, self, signal["id"], signal["inst"])
            except Exception as exc:
                zq.log.warning("news snapshot unavailable for signal %s: %s", signal.get("id"), exc)
        return signal
    zq.Paper.create_tracked_signal = create_with_news_snapshot

    old_close = zq.Paper.close_tracked_signal
    def close_and_learn_news(self, *args, **kwargs):
        closed = old_close(self, *args, **kwargs)
        if closed:
            try:
                _train_news_model(zq, self)
            except Exception as exc:
                zq.log.warning("news outcome learning skipped safely: %s", exc)
        return closed
    zq.Paper.close_tracked_signal = close_and_learn_news

    # Publish a useful empty-state report on first startup, without inventing news.
    if not os.path.exists(os.path.join(_state_dir(zq), "news_intelligence.json")):
        try:
            articles = fetch_news()
            _write_intelligence(zq, articles)
        except Exception as exc:
            zq.log.warning("initial news report skipped: %s", exc)
    return zq
