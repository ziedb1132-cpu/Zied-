#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upcoming 24-hour economic and crypto event intelligence for Zied Quant.

Only events with a verifiable future timestamp are used. Historical candle rows
remain unknown; the live event snapshot is stored in each signal feature vector.
"""
import datetime as dt
import json
import math
import os
import re
import time
from urllib.parse import urlencode

import numpy as np
import requests

CACHE_SECONDS = 600
WINDOW_SECONDS = 24 * 3600
FEATURE_NAMES = (
    "impact", "count_24h", "hours_to_event", "coin_relevance", "risk",
    "volatility", "confidence", "high_impact", "macro_ratio", "available",
)
EVENT_FEATURE_COLS = tuple("upcoming_event_" + name for name in FEATURE_NAMES)
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CMC_URL = "https://developers.coinmarketcal.com/v1/events"
_STATE = {"fetched_at": 0.0, "events": [], "sources": []}
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ZiedQuantEventIntelligence/1.0"})
_COIN_ALIASES = {
    "BTC": ("bitcoin", "btc"), "ETH": ("ethereum", "eth"),
    "SOL": ("solana", "sol"), "XRP": ("ripple", "xrp"),
    "BNB": ("binance", "bnb"), "DOGE": ("dogecoin", "doge"),
    "ADA": ("cardano", "ada"), "TON": ("toncoin", "ton"),
    "TRX": ("tron", "trx"), "LINK": ("chainlink", "link"),
    "AVAX": ("avalanche", "avax"), "DOT": ("polkadot", "dot"),
    "LTC": ("litecoin", "ltc"), "BCH": ("bitcoin cash", "bch"),
    "SUI": ("sui",), "APT": ("aptos",), "ARB": ("arbitrum",),
    "OP": ("optimism",), "NEAR": ("near protocol",), "ATOM": ("cosmos",),
    "UNI": ("uniswap",), "AAVE": ("aave",), "SHIB": ("shiba inu",),
}
_MACRO_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY"}
_HIGH = {"high", "3", "critical"}
_MEDIUM = {"medium", "2", "moderate"}


def _parse_time(value):
    if isinstance(value, (int, float)):
        stamp = float(value)
        return stamp / 1000.0 if stamp > 10_000_000_000 else stamp
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return 0.0


def _normalise_ff(item):
    title = str(item.get("title") or "").strip()
    currency = str(item.get("country") or item.get("currency") or "").upper()
    stamp = _parse_time(item.get("date") or item.get("timestamp"))
    if not title or not stamp:
        return None
    impact_text = str(item.get("impact") or "").strip().lower()
    impact = 1.0 if impact_text in _HIGH else 0.55 if impact_text in _MEDIUM else 0.25
    return {
        "title": title[:300], "timestamp": stamp, "source": "ForexFactory",
        "type": "macro", "currency": currency, "impact": impact,
        "confidence": 0.88, "url": str(item.get("url") or "")[:500],
    }


def _normalise_cmc(item):
    title = item.get("title") or item.get("name") or ""
    stamp = _parse_time(item.get("date_event") or item.get("date") or item.get("start_date"))
    if isinstance(title, dict):
        title = title.get("en") or next(iter(title.values()), "")
    if not title or not stamp:
        return None
    coins = item.get("coins") or item.get("coin") or []
    if isinstance(coins, dict):
        coins = [coins]
    symbols = []
    for coin in coins if isinstance(coins, list) else []:
        if isinstance(coin, dict):
            symbol = coin.get("symbol") or coin.get("code")
            if symbol:
                symbols.append(str(symbol).upper())
        elif coin:
            symbols.append(str(coin).upper())
    importance = str(item.get("proof") or item.get("source") or "").lower()
    return {
        "title": str(title)[:300], "timestamp": stamp, "source": "CoinMarketCal",
        "type": "crypto", "currency": "", "coins": symbols,
        "impact": 0.8, "confidence": 0.78 if importance else 0.70,
        "url": str(item.get("proof") or item.get("url") or "")[:500],
    }


def fetch_events(force=False):
    now = time.time()
    if not force and now - _STATE["fetched_at"] < CACHE_SECONDS:
        return list(_STATE["events"])
    events, sources = [], []
    try:
        response = _SESSION.get(FF_URL, timeout=(4, 9))
        response.raise_for_status()
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("events", [])
        for item in rows:
            event = _normalise_ff(item)
            if event and now <= event["timestamp"] <= now + WINDOW_SECONDS:
                events.append(event)
        sources.append({"name": "ForexFactory", "status": "online", "events": len(events)})
    except Exception as exc:
        sources.append({"name": "ForexFactory", "status": "offline", "events": 0,
                        "error": str(exc)[:160]})

    # CoinMarketCal is optional: use it only when the owner has configured its
    # official API credentials. No API key is embedded in source code.
    api_key = os.getenv("COINMARKETCAL_API_KEY", "").strip()
    api_code = os.getenv("COINMARKETCAL_API_CODE", "").strip()
    if api_key and api_code:
        try:
            headers = {"x-api-key": api_key, "x-api-code": api_code}
            params = {"dateRangeStart": dt.datetime.fromtimestamp(now, dt.timezone.utc).date().isoformat(),
                      "dateRangeEnd": dt.datetime.fromtimestamp(now + WINDOW_SECONDS, dt.timezone.utc).date().isoformat(),
                      "sortBy": "created_desc", "max": 100}
            response = _SESSION.get(CMC_URL, headers=headers, params=params, timeout=(4, 9))
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("body", payload.get("events", [])) if isinstance(payload, dict) else []
            count = 0
            for item in rows if isinstance(rows, list) else []:
                event = _normalise_cmc(item)
                if event and now <= event["timestamp"] <= now + WINDOW_SECONDS:
                    events.append(event)
                    count += 1
            sources.append({"name": "CoinMarketCal", "status": "online", "events": count})
        except Exception as exc:
            sources.append({"name": "CoinMarketCal", "status": "offline", "events": 0,
                            "error": str(exc)[:160]})
    else:
        sources.append({"name": "CoinMarketCal", "status": "not_configured", "events": 0})

    unique = {}
    for event in events:
        key = (event["source"], event["title"].lower(), int(event["timestamp"] // 60))
        unique[key] = event
    events = sorted(unique.values(), key=lambda x: x["timestamp"])
    _STATE.update({"fetched_at": now, "events": events, "sources": sources})
    return list(events)


def _relevance(inst, event):
    base = str(inst).split("-")[0].upper()
    title = str(event.get("title", "")).lower()
    symbols = set(event.get("coins") or [])
    aliases = _COIN_ALIASES.get(base, (base.lower(),))
    if base in symbols or any(re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", title)
                              for a in aliases):
        return 1.0
    currency = str(event.get("currency") or "").upper()
    if currency == "USD":
        return 0.75
    if currency in _MACRO_CURRENCIES:
        return 0.35
    return 0.15 if event.get("type") == "crypto" else 0.0


def event_features(inst, events=None, now=None):
    events = fetch_events() if events is None else events
    now = time.time() if now is None else float(now)
    relevant = []
    for event in events:
        remaining = float(event.get("timestamp") or 0) - now
        if 0 <= remaining <= WINDOW_SECONDS:
            rel = _relevance(inst, event)
            if rel > 0:
                relevant.append((event, rel, remaining))
    if not relevant:
        available = 1.0 if any(x.get("status") == "online" for x in _STATE["sources"]) else 0.0
        return [0.0] * (len(FEATURE_NAMES) - 1) + [available]
    impacts = [float(e.get("impact", 0.25)) * rel for e, rel, _ in relevant]
    max_impact = max(impacts)
    count = min(1.0, math.log1p(len(relevant)) / math.log(11))
    nearest = min(x[2] for x in relevant) / 3600.0
    time_proximity = max(0.0, min(1.0, 1.0 - nearest / 24.0))
    relevance = float(np.mean([x[1] for x in relevant]))
    risk = max_impact * (0.65 if max_impact >= 0.7 else 0.35)
    volatility = min(1.0, max_impact * (0.6 + 0.4 * count))
    confidence = float(np.average([e.get("confidence", 0.5) for e, _, _ in relevant],
                                  weights=[max(0.05, rel) for _, rel, _ in relevant]))
    high = float(any(float(e.get("impact", 0)) >= 0.95 for e, _, _ in relevant))
    macro_ratio = sum(1 for e, _, _ in relevant if e.get("type") == "macro") / len(relevant)
    vector = [max_impact, count, time_proximity, relevance, risk, volatility,
              confidence, high, macro_ratio, 1.0]
    return [float(np.clip(x, 0.0, 1.0)) for x in vector]


def _write_report(zq, events):
    now = time.time()
    payload = {
        "schema": 1, "updated_at": now, "window_hours": 24,
        "timezone": "UTC", "event_count": len(events),
        "sources": _STATE["sources"],
        "events": [{**event, "hours_until": round((event["timestamp"] - now) / 3600, 2)}
                   for event in events[:100]],
        "feature_names": list(FEATURE_NAMES),
        "note": "Only timestamped future events are included; missing feeds are not fabricated.",
    }
    os.makedirs(zq.CFG["DIR"], exist_ok=True)
    path = os.path.join(zq.CFG["DIR"], "upcoming_events.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    os.replace(tmp, path)


def install(zq):
    existing = list(zq.FEATURE_COLS)
    missing = [name for name in EVENT_FEATURE_COLS if name not in existing]
    if missing:
        old_aligned = zq.aligned_matrix
        zq.FEATURE_COLS = existing + list(EVENT_FEATURE_COLS)
        zq.N_FEATURES = len(zq.FEATURE_COLS)
        zq.CFG["REQUIRED_SIGNAL_FEATURES"] = zq.N_FEATURES

        def aligned_with_event_unknown(frames, btc):
            matrix = old_aligned(frames, btc)
            for name in EVENT_FEATURE_COLS:
                matrix[name] = np.nan
            return matrix.reindex(columns=zq.FEATURE_COLS)
        zq.aligned_matrix = aligned_with_event_unknown

        old_rows_to_ds = zq._rows_to_ds
        prior_feature_count = zq.N_FEATURES - len(EVENT_FEATURE_COLS)
        # Older database rows may predate both the ten news columns and these
        # ten event columns. Adapt them to the complete schema in one step so
        # the news wrapper does not mistake an intermediate vector for current data.
        legacy_without_news_count = prior_feature_count - 10
        def rows_to_ds_with_legacy_events(rows, label_key="label", ret_key="ret"):
            adapted = []
            for row in rows:
                try:
                    item = dict(row)
                    values = json.loads(item["feats"])
                    if isinstance(values, list) and len(values) == legacy_without_news_count:
                        values = values + [0.0] * (10 + len(EVENT_FEATURE_COLS))
                        item["feats"] = json.dumps(values, separators=(",", ":"), allow_nan=False)
                    elif isinstance(values, list) and len(values) == prior_feature_count:
                        item["feats"] = json.dumps(
                            values + [0.0] * len(EVENT_FEATURE_COLS),
                            separators=(",", ":"), allow_nan=False)
                    adapted.append(item)
                except (TypeError, ValueError, KeyError):
                    adapted.append(row)
            return old_rows_to_ds(adapted, label_key, ret_key)
        zq._rows_to_ds = rows_to_ds_with_legacy_events

        import hashlib
        schema = "|".join(zq.FEATURE_COLS).encode("utf-8")
        zq.DATA_VER = 4 + int(hashlib.sha256(schema).hexdigest()[:8], 16)
        zq.log.info("upcoming-event features installed: added=%d total_features=%d schema=%d",
                    len(EVENT_FEATURE_COLS), zq.N_FEATURES, zq.DATA_VER)

    old_rank = zq.rank_universe
    def rank_with_upcoming_events(X, vol24, *args, **kwargs):
        if X is None or X.empty:
            return old_rank(X, vol24, *args, **kwargs)
        try:
            events = fetch_events()
        except Exception as exc:
            zq.log.warning("upcoming event calendar unavailable: %s", exc)
            events = []
        vectors = []
        for inst in X.index:
            try:
                vector = event_features(inst, events)
            except Exception as exc:
                zq.log.debug("event feature extraction failed for %s: %s", inst, exc)
                vector = [0.0] * len(EVENT_FEATURE_COLS)
            vectors.append(vector)
        X[list(EVENT_FEATURE_COLS)] = np.asarray(vectors, dtype=np.float64)
        result = old_rank(X, vol24, *args, **kwargs)
        try:
            _write_report(zq, events)
            zq.log.info("upcoming events: events_24h=%d sources=%s primary_features=%d",
                        len(events), json.dumps(_STATE["sources"], ensure_ascii=False),
                        len(EVENT_FEATURE_COLS))
        except Exception as exc:
            zq.log.warning("upcoming event report write skipped: %s", exc)
        return result
    zq.rank_universe = rank_with_upcoming_events

    # Publish a report even before the first ranking call; failures remain visible.
    try:
        _write_report(zq, fetch_events())
    except Exception as exc:
        zq.log.warning("initial upcoming-event report skipped: %s", exc)
    return zq
