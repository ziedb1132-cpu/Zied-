#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Candlestick-pattern and crypto-news intelligence for Zied Quant.

Patterns/news are research signals, not guarantees. The causal pattern feature
matrix is consumed by deep_learning_evolution for primary-model training and
inference; the optional rank overlay remains separate from portfolio risk checks.
This module never places real orders.
"""
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

PATTERN_VERSION = 1
CACHE_SECONDS = 900
MAX_PATTERN_COINS = 40
NEWS_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://cryptoslate.com/feed/",
)
POSITIVE = {
    "surge", "surges", "surged", "rally", "rallies", "bullish", "breakout",
    "approval", "approved", "adoption", "partnership", "launch", "upgrade",
    "inflow", "inflows", "record", "growth", "gains", "gain", "soars", "soar",
    "support", "rebound", "recovery", "accumulate", "accumulation", "positive",
    "beat", "beats", "buy", "invest", "investment", " ETF "
}
NEGATIVE = {
    "crash", "crashes", "crashed", "bearish", "hack", "hacked", "exploit",
    "lawsuit", "sued", "ban", "banned", "probe", "investigation", "outflow",
    "outflows", "selloff", "sell-off", "plunge", "plunges", "falls", "fall",
    "drop", "drops", "decline", "declines", "liquidation", "liquidations",
    "fraud", "scam", "delisting", "delisted", "warning", "risk", "risks",
    "negative", "miss", "misses", "rejected", "breach", "bankruptcy"
}
GENERAL_TERMS = (
    "crypto", "cryptocurrency", "bitcoin", "ethereum", "digital asset",
    "digital assets", "crypto market", "sec", "etf", "stablecoin", "blockchain"
)

_STATE = {"news_at": 0.0, "news": [], "coin_scores": {}, "patterns": {}}
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ZiedQuantResearchBot/1.0 (RSS headline analysis)"})


def _finite(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _candle_shape(row):
    o, h, l, c = (_finite(row[k]) for k in ("o", "h", "l", "c"))
    span = max(h - l, abs(c) * 1e-12, 1e-12)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "o": o, "h": h, "l": l, "c": c, "span": span, "body": body,
        "upper": max(0.0, upper), "lower": max(0.0, lower),
        "body_pct": body / span, "upper_pct": max(0.0, upper) / span,
        "lower_pct": max(0.0, lower) / span, "bull": c > o, "bear": c < o,
    }


def detect_candlestick_patterns(candles):
    """Return named patterns on closed OHLC candles, with direction and confidence."""
    if candles is None or len(candles) < 3:
        return []
    d = candles.tail(5).reset_index(drop=True)
    cs = [_candle_shape(row) for _, row in d.iterrows()]
    a, b, c = cs[-3], cs[-2], cs[-1]
    p = []
    def add(name, direction, confidence):
        _register_pattern_name(name)
        p.append({"name": name, "direction": direction,
                  "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3)})

    # One-candle patterns.
    if c["body_pct"] <= 0.08:
        add("Doji", "neutral", 0.55)
    if c["body_pct"] <= 0.10 and c["lower_pct"] >= 0.60 and c["upper_pct"] <= 0.12:
        add("Dragonfly Doji", "bullish", 0.78)
    if c["body_pct"] <= 0.10 and c["upper_pct"] >= 0.60 and c["lower_pct"] <= 0.12:
        add("Gravestone Doji", "bearish", 0.78)
    if c["lower"] >= max(c["body"] * 2.2, c["span"] * 0.48) and c["upper"] <= c["span"] * 0.22:
        add("Hammer / Hanging Man", "bullish" if b["bear"] or a["bear"] else "neutral", 0.67)
    if c["upper"] >= max(c["body"] * 2.2, c["span"] * 0.48) and c["lower"] <= c["span"] * 0.22:
        add("Inverted Hammer / Shooting Star", "bearish" if b["bull"] or a["bull"] else "neutral", 0.67)
    if c["body_pct"] >= 0.88 and c["upper_pct"] <= 0.06 and c["lower_pct"] <= 0.06:
        add("Bullish Marubozu" if c["bull"] else "Bearish Marubozu",
            "bullish" if c["bull"] else "bearish", 0.78)
    if c["body_pct"] <= 0.28 and c["upper_pct"] >= 0.25 and c["lower_pct"] >= 0.25:
        add("Spinning Top", "neutral", 0.45)

    # Two-candle reversal / continuation patterns.
    if a["bear"] and c["bull"] and c["o"] <= a["c"] and c["c"] >= a["o"] and c["body"] >= a["body"] * 0.8:
        add("Bullish Engulfing", "bullish", 0.86)
    if a["bull"] and c["bear"] and c["o"] >= a["c"] and c["c"] <= a["o"] and c["body"] >= a["body"] * 0.8:
        add("Bearish Engulfing", "bearish", 0.86)
    if a["bear"] and c["bull"] and c["o"] < a["c"] and c["c"] > (a["o"] + a["c"]) / 2 and c["c"] < a["o"]:
        add("Piercing Line", "bullish", 0.72)
    if a["bull"] and c["bear"] and c["o"] > a["c"] and c["c"] < (a["o"] + a["c"]) / 2 and c["c"] > a["o"]:
        add("Dark Cloud Cover", "bearish", 0.72)
    if a["bear"] and c["bull"] and c["o"] >= min(a["o"], a["c"]) and c["c"] <= max(a["o"], a["c"]):
        add("Bullish Harami", "bullish", 0.62)
    if a["bull"] and c["bear"] and c["o"] >= min(a["o"], a["c"]) and c["c"] <= max(a["o"], a["c"]):
        add("Bearish Harami", "bearish", 0.62)
    if abs(c["h"] - a["h"]) / max(a["h"], 1e-12) < 0.0015:
        add("Tweezer Top", "bearish", 0.60)
    if abs(c["l"] - a["l"]) / max(a["l"], 1e-12) < 0.0015:
        add("Tweezer Bottom", "bullish", 0.60)
    if b["h"] < a["h"] and b["l"] > a["l"] and c["h"] > b["h"] and c["l"] < b["l"]:
        add("Inside Bar Breakout", "bullish" if c["bull"] else "bearish", 0.60)
    if c["h"] > b["h"] and c["l"] < b["l"]:
        add("Outside Bar", "bullish" if c["bull"] else "bearish", 0.55)

    # Three-candle patterns.
    small_b = b["body"] <= max(a["body"], c["body"]) * 0.45
    if a["bear"] and small_b and c["bull"] and c["c"] > (a["o"] + a["c"]) / 2 and c["body"] >= a["body"] * 0.55:
        add("Morning Star", "bullish", 0.84)
    if a["bull"] and small_b and c["bear"] and c["c"] < (a["o"] + a["c"]) / 2 and c["body"] >= a["body"] * 0.55:
        add("Evening Star", "bearish", 0.84)
    if a["bull"] and b["bull"] and c["bull"] and a["c"] < b["c"] < c["c"] and \
       all(x["body_pct"] >= 0.45 for x in (a, b, c)):
        add("Three White Soldiers", "bullish", 0.82)
    if a["bear"] and b["bear"] and c["bear"] and a["c"] > b["c"] > c["c"] and \
       all(x["body_pct"] >= 0.45 for x in (a, b, c)):
        add("Three Black Crows", "bearish", 0.82)
    if len(cs) >= 4:
        x, y = cs[-4], cs[-3]
        if x["bull"] and y["bear"] and b["bull"] and c["bear"] and \
           c["c"] < min(x["l"], b["l"]) and y["body_pct"] > 0.55:
            add("Bearish Kicker-like Reversal", "bearish", 0.58)
        if x["bear"] and y["bull"] and b["bear"] and c["bull"] and \
           c["c"] > max(x["h"], b["h"]) and y["body_pct"] > 0.55:
            add("Bullish Kicker-like Reversal", "bullish", 0.58)

    # Avoid double-counting near-identical detections; keep strongest per name.
    uniq = {}
    for item in p:
        if item["name"] not in uniq or item["confidence"] > uniq[item["name"]]["confidence"]:
            uniq[item["name"]] = item
    return list(uniq.values())


def pattern_score(candles):
    patterns = detect_candlestick_patterns(candles)
    if not patterns:
        return 0.0, patterns
    vals = [p["confidence"] * (1 if p["direction"] == "bullish" else -1 if p["direction"] == "bearish" else 0)
            for p in patterns]
    return float(np.clip(np.mean(vals), -1, 1)), patterns


# Fixed pattern vocabulary shared by live inference and historical training.
PATTERN_NAMES = (
    "Doji", "Dragonfly Doji", "Gravestone Doji", "Hammer / Hanging Man",
    "Inverted Hammer / Shooting Star", "Bullish Marubozu", "Bearish Marubozu",
    "Spinning Top", "Bullish Engulfing", "Bearish Engulfing", "Piercing Line",
    "Dark Cloud Cover", "Bullish Harami", "Bearish Harami", "Tweezer Top",
    "Tweezer Bottom", "Inside Bar Breakout", "Outside Bar", "Morning Star",
    "Evening Star", "Three White Soldiers", "Three Black Crows",
    "Bearish Kicker-like Reversal", "Bullish Kicker-like Reversal",
)

def _pattern_slug(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

PATTERN_FEATURE_NAMES = tuple(
    [f"candle_pat_{_pattern_slug(name)}" for name in PATTERN_NAMES]
    + ["candle_pat_count"]
)
_PATTERN_REGISTRY_PATH = None
_DISCOVERED_PATTERN_DEFS = []
DISCOVERY_MIN_OCCURRENCES = 30
DISCOVERY_MIN_VALIDATION = 10
DISCOVERY_MAX_APPROVED = 40
DISCOVERY_MAX_CANDIDATES = 80000
DISCOVERY_HORIZON = 12
DISCOVERY_ROUND_TRIP_COST = 0.003


def _candle_token(row):
    """Scale-free candle-shape token used by the 3..20-bar sequence miner."""
    x = _candle_shape(row)
    direction = 1 if x["bull"] else (-1 if x["bear"] else 0)
    body_bin = min(3, int(x["body_pct"] * 4))
    wick = x["upper_pct"] - x["lower_pct"]
    wick_bin = 0 if abs(wick) < 0.12 else (1 if wick > 0 else 2)
    return f"{direction + 1}{body_bin}{wick_bin}"


def discover_sequence_patterns(candles_by_symbol, registry_path=None):
    """Mine causal 3..20-candle shapes and approve only chronological OOS edges.

    The sequence ends at bar t; its outcome is measured only from t+1 onward.
    Discovery chooses long/short direction on the earlier 60%; the later 40%
    is used once for validation and never for direction/threshold selection.
    """
    occurrences = {}
    for symbol, frame in (candles_by_symbol or {}).items():
        if not isinstance(frame, pd.DataFrame) or len(frame) < 40:
            continue
        required = {"o", "h", "l", "c"}
        if not required.issubset(frame.columns):
            continue
        data = frame.reset_index(drop=True)
        tokens = [_candle_token(row) for _, row in data.iterrows()]
        close = pd.to_numeric(data["c"], errors="coerce").to_numpy(dtype=float)
        for end in range(19, len(data) - DISCOVERY_HORIZON):
            future = close[end + DISCOVERY_HORIZON]
            current = close[end]
            if not (np.isfinite(current) and current > 0 and np.isfinite(future)):
                continue
            ret = future / current - 1.0
            stamp = frame.index[end] if len(frame.index) else end
            for width in range(3, 21):
                seq = tuple(tokens[end - width + 1:end + 1])
                if seq not in occurrences and len(occurrences) >= DISCOVERY_MAX_CANDIDATES:
                    continue
                bucket = occurrences.setdefault(seq, [])
                if len(bucket) < 500:
                    bucket.append((stamp, str(symbol), float(ret)))

    approved = []
    for seq, events in occurrences.items():
        if len(events) < DISCOVERY_MIN_OCCURRENCES:
            continue
        events.sort(key=lambda item: (item[0], item[1]))
        # Split by unique candle timestamp, never by individual coin rows:
        # coins observed on the same bar cannot leak across the temporal boundary.
        timestamps = sorted({event[0] for event in events})
        split_at = max(1, min(len(timestamps) - 1, int(len(timestamps) * 0.60)))
        cutoff = timestamps[split_at]
        discovery = [event for event in events if event[0] < cutoff]
        validation = [event for event in events if event[0] >= cutoff]
        if len(discovery) < 18 or len(validation) < DISCOVERY_MIN_VALIDATION:
            continue
        # Reject single-coin coincidences; require cross-market recurrence.
        if len({e[1] for e in events}) < 3 or len({e[1] for e in validation}) < 2:
            continue
        # Select orientation using discovery data only.
        long_mean = float(np.mean([e[2] - DISCOVERY_ROUND_TRIP_COST for e in discovery]))
        short_mean = float(np.mean([-e[2] - DISCOVERY_ROUND_TRIP_COST for e in discovery]))
        direction = "long" if long_mean >= short_mean else "short"
        val_net = [((e[2] if direction == "long" else -e[2]) - DISCOVERY_ROUND_TRIP_COST)
                   for e in validation]
        wins = [x for x in val_net if x > 0]
        losses = [x for x in val_net if x < 0]
        mean_net = float(np.mean(val_net))
        win_rate = len(wins) / len(val_net)
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else (99.0 if wins else 0.0)
        # Require positive validation expectancy and a modest edge after costs.
        if mean_net <= 0 or win_rate < 0.52 or profit_factor < 1.10:
            continue
        slug = "_".join(seq)
        feature_name = "candle_seq_" + str(len(seq)) + "_" + slug
        approved.append({
            "name": f"Discovered {len(seq)}-bar sequence {slug[:36]}",
            "feature_name": feature_name, "sequence": list(seq), "length": len(seq),
            "direction": direction, "status": "approved",
            "occurrences": len(events), "discovery_samples": len(discovery),
            "validation_samples": len(validation), "validation_win_rate": win_rate,
            "validation_profit_factor": profit_factor, "validation_mean_net_return": mean_net,
            "round_trip_cost_assumption": DISCOVERY_ROUND_TRIP_COST,
            "horizon_bars": DISCOVERY_HORIZON,
            "last_validated_at": time.time(),
        })
    approved.sort(key=lambda x: (x["validation_mean_net_return"], x["validation_profit_factor"]),
                  reverse=True)
    approved = approved[:DISCOVERY_MAX_APPROVED]
    if registry_path:
        try:
            existing = {}
            with open(registry_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            old_defs = existing.get("discovered_patterns", []) if isinstance(existing, dict) else []
        except (OSError, ValueError, TypeError):
            old_defs = []
        # Keep previously validated definitions for at most seven days while
        # transient candle-fetch failures occur; fresh validations replace them.
        now = time.time()
        merged = {
            p["feature_name"]: p for p in old_defs
            if isinstance(p, dict) and p.get("status") == "approved"
            and now - float(p.get("last_validated_at") or 0) <= 7 * 86400
        }
        merged.update({p["feature_name"]: p for p in approved})
        global _DISCOVERED_PATTERN_DEFS
        _DISCOVERED_PATTERN_DEFS = sorted(
            merged.values(),
            key=lambda p: (float(p.get("last_validated_at") or 0),
                           float(p.get("validation_mean_net_return") or 0)),
            reverse=True
        )[:DISCOVERY_MAX_APPROVED]
        if _PATTERN_REGISTRY_PATH != registry_path:
            configure_pattern_registry(registry_path)
        else:
            _write_pattern_registry()
    return {"candidate_sequences": len(occurrences), "approved": len(approved),
            "minimum_occurrences": DISCOVERY_MIN_OCCURRENCES,
            "horizon_bars": DISCOVERY_HORIZON, "cost_assumption": DISCOVERY_ROUND_TRIP_COST}


def _write_pattern_registry():
    if not _PATTERN_REGISTRY_PATH:
        return
    payload = {
        "version": 1,
        "patterns": list(PATTERN_NAMES),
        "feature_names": list(PATTERN_FEATURE_NAMES),
        "discovered_patterns": list(_DISCOVERED_PATTERN_DEFS),
        "note": "Built-in patterns plus independently validated 3..20-bar sequence definitions; schema changes activate on next process start."
    }
    tmp = _PATTERN_REGISTRY_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(_PATTERN_REGISTRY_PATH) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, allow_nan=False, indent=2)
        os.replace(tmp, _PATTERN_REGISTRY_PATH)
    except OSError:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def configure_pattern_registry(path):
    """Load persistent pattern vocabulary before the primary model schema is built."""
    global PATTERN_NAMES, PATTERN_FEATURE_NAMES, _PATTERN_REGISTRY_PATH, _DISCOVERED_PATTERN_DEFS
    _PATTERN_REGISTRY_PATH = path
    try:
        with open(path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        names = saved.get("patterns", []) if isinstance(saved, dict) else []
        _DISCOVERED_PATTERN_DEFS = [
            p for p in saved.get("discovered_patterns", [])
            if isinstance(p, dict) and p.get("status") == "approved"
            and isinstance(p.get("sequence"), list) and 3 <= len(p["sequence"]) <= 20
            and isinstance(p.get("feature_name"), str)
        ] if isinstance(saved, dict) else []
    except (OSError, ValueError, TypeError):
        names = []
        _DISCOVERED_PATTERN_DEFS = []
    # Keep the built-in vocabulary first and append only valid, unique discoveries.
    merged = list(PATTERN_NAMES)
    for name in names:
        if isinstance(name, str) and name.strip() and len(name) <= 80 and name not in merged:
            merged.append(name)
    PATTERN_NAMES = tuple(merged)
    PATTERN_FEATURE_NAMES = tuple(
        [f"candle_pat_{_pattern_slug(name)}" for name in PATTERN_NAMES]
        + ["candle_pat_count"]
        + [p["feature_name"] for p in _DISCOVERED_PATTERN_DEFS]
    )
    _write_pattern_registry()
    return PATTERN_FEATURE_NAMES


def _register_pattern_name(name):
    """Persist a newly detected pattern for inclusion in the next model schema."""
    global PATTERN_NAMES, PATTERN_FEATURE_NAMES
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        return
    if name in PATTERN_NAMES:
        return
    PATTERN_NAMES = tuple(list(PATTERN_NAMES) + [name])
    PATTERN_FEATURE_NAMES = tuple(
        [f"candle_pat_{_pattern_slug(item)}" for item in PATTERN_NAMES]
        + ["candle_pat_count"]
        + [p["feature_name"] for p in _DISCOVERED_PATTERN_DEFS]
    )
    _write_pattern_registry()


def pattern_feature_matrix(candles):
    """Build causal per-candle pattern inputs using only candles closed at each row.

    Bullish confidence is positive, bearish confidence negative, and neutral
    patterns are represented by the aggregate count without a directional score.
    The first two rows remain NaN because the detector needs at least 3 candles.
    """
    if candles is None or len(candles) == 0:
        return pd.DataFrame(columns=PATTERN_FEATURE_NAMES, index=getattr(candles, "index", None))
    values = np.full((len(candles), len(PATTERN_FEATURE_NAMES)), np.nan, dtype=np.float64)
    name_to_col = {name: i for i, name in enumerate(PATTERN_NAMES)}
    count_col = len(PATTERN_NAMES)
    dynamic_start = count_col + 1
    dynamic_defs = list(_DISCOVERED_PATTERN_DEFS)
    tokens = [_candle_token(row) for _, row in candles.iterrows()]
    dynamic_index = {p["feature_name"]: dynamic_start + i for i, p in enumerate(dynamic_defs)}
    for end in range(2, len(candles)):
        detected = detect_candlestick_patterns(candles.iloc[max(0, end - 4):end + 1])
        values[end, :] = 0.0
        for item in detected:
            col = name_to_col.get(item.get("name"))
            if col is None:
                continue
            confidence = float(item.get("confidence") or 0.0)
            direction = item.get("direction")
            values[end, col] = confidence if direction == "bullish" else (-confidence if direction == "bearish" else 0.0)
        values[end, count_col] = float(len(detected))
        for definition in dynamic_defs:
            seq = definition.get("sequence", [])
            width = len(seq)
            if width <= end + 1 and tokens[end - width + 1:end + 1] == seq:
                col = dynamic_index.get(definition["feature_name"])
                values[end, col] = 1.0 if definition.get("direction") == "long" else -1.0
    return pd.DataFrame(values, index=candles.index, columns=PATTERN_FEATURE_NAMES)


def _symbol_terms(inst):
    base = str(inst).split("-")[0].upper()
    terms = {base, base.lower()}
    aliases = {
        "BTC": ("bitcoin",), "ETH": ("ethereum", "ether"), "SOL": ("solana",),
        "XRP": ("ripple",), "DOGE": ("dogecoin",), "ADA": ("cardano",),
        "TON": ("toncoin", "the open network"), "BNB": ("binance coin", "bnb"),
        "TRX": ("tron",), "LINK": ("chainlink",), "AVAX": ("avalanche",),
        "DOT": ("polkadot",), "LTC": ("litecoin",), "BCH": ("bitcoin cash",),
        "SUI": ("sui",), "APT": ("aptos",), "ARB": ("arbitrum",),
        "OP": ("optimism",), "NEAR": ("near protocol",), "ATOM": ("cosmos",),
        "UNI": ("uniswap",), "AAVE": ("aave",), "SHIB": ("shiba inu",),
    }
    terms.update(aliases.get(base, ()))
    return tuple(sorted((x for x in terms if x), key=len, reverse=True))


def _sentiment(text):
    s = " " + re.sub(r"[^a-z0-9$-]+", " ", str(text).lower()) + " "
    pos = sum(1 for w in POSITIVE if w.strip() in s)
    neg = sum(1 for w in NEGATIVE if w in s)
    if pos + neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / max(2.0, (pos + neg) ** 0.5 * 1.5), -1, 1))


def fetch_news(force=False):
    now = time.time()
    if not force and now - _STATE["news_at"] < CACHE_SECONDS:
        return _STATE["news"]
    entries, seen = [], set()
    for feed in NEWS_FEEDS:
        try:
            resp = _SESSION.get(feed, timeout=8)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:40]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())
                entries.append({"title": title[:400], "link": link[:600], "published": pub[:100],
                                "source": urlparse(feed).netloc, "sentiment": _sentiment(title)})
        except Exception:
            # News is optional: a feed outage must not stop market scanning.
            continue
    entries.sort(key=lambda x: x["published"], reverse=True)
    _STATE["news"], _STATE["news_at"] = entries[:120], now
    return _STATE["news"]


def news_score(inst, headlines=None):
    headlines = fetch_news() if headlines is None else headlines
    terms = _symbol_terms(inst)
    base = str(inst).split("-")[0].upper()
    related = []
    for item in headlines:
        title = item["title"].lower()
        has_coin = any(re.search(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])", title)
                       for term in terms)
        is_general = any(term in title for term in GENERAL_TERMS)
        # Coin-specific stories count most; broad market headlines count at lower weight.
        if has_coin:
            related.append((item["sentiment"], 1.0, item))
        elif is_general and base in {"BTC", "ETH", "SOL", "XRP", "BNB", "TON", "DOGE", "ADA"}:
            related.append((item["sentiment"], 0.25, item))
    if not related:
        return 0.0, []
    vals = [s * w for s, w, _ in related[:12]]
    score = float(np.clip(sum(vals) / max(1.0, sum(w for _, w, _ in related[:12])), -1, 1))
    return score, [x[2] for x in related[:5]]


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp, path)


def install(zq):
    """Install the separate conservative ranking overlay; model inputs are installed by deep_learning_evolution."""
    old_rank = zq.rank_universe

    def rank_with_candle_news(X, vol24, *args, **kwargs):
        # Forward optional ranking context instead of narrowing the wrapped API.
        base_scores = old_rank(X, vol24, *args, **kwargs)
        if X is None or X.empty:
            return base_scores
        headlines = fetch_news()
        top = list(base_scores.head(MAX_PATTERN_COINS).index)
        pattern_rows, news_rows = {}, {}
        adjusted = base_scores.copy()
        for inst in top:
            try:
                candles = zq.get_candles(inst, "1H", 8)
                ps, pats = pattern_score(candles)
                ns, related = news_score(inst, headlines)
                pattern_rows[inst] = {"score": ps, "patterns": pats}
                news_rows[inst] = {"score": ns, "headlines": related}
                # Keep this as a small rank adjustment; AI probability and all risk checks remain.
                adjusted.loc[inst] = float(base_scores.loc[inst]) + 0.18 * ps + 0.12 * ns
            except Exception as exc:
                zq.log.debug("candle/news overlay skipped for %s: %s", inst, exc)
        adjusted = adjusted.sort_values(ascending=False)
        outdir = zq.CFG["DIR"]
        os.makedirs(outdir, exist_ok=True)
        _save_json(os.path.join(outdir, "candle_patterns.json"), {
            "version": PATTERN_VERSION, "updated_at": time.time(), "timeframe": "1H",
            "scanned": len(pattern_rows), "coins": pattern_rows,
            "note": "Candlestick patterns are heuristic context, not guaranteed predictions."
        })
        _save_json(os.path.join(outdir, "news_sentiment.json"), {
            "updated_at": time.time(), "feeds": list(NEWS_FEEDS), "headline_count": len(headlines),
            "coins": news_rows,
            "note": "Headline sentiment is a lightweight keyword heuristic; missing news is neutral."
        })
        _STATE["coin_scores"] = {k: {"pattern": pattern_rows.get(k, {}).get("score", 0.0),
                                     "news": news_rows.get(k, {}).get("score", 0.0)} for k in top}
        zq.log.info("candle/news intelligence: scanned=%d headlines=%d", len(pattern_rows), len(headlines))
        return adjusted

    zq.rank_universe = rank_with_candle_news
    return zq
