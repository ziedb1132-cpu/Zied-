#!/usr/bin/env python3
"""Create a complete, read-only dashboard snapshot from the completed paper state."""
import json, sqlite3, sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
db_path = root / "paper.db"

def safe_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def rows(conn, sql, params=()):
    try:
        cur = conn.execute(sql, params)
        names = [x[0] for x in cur.description or []]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    except sqlite3.Error:
        return []

data = {
    "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "meta": {}, "summary": {}, "signals": [], "tracked": [],
    "contents": {}, "strategy_lab": None, "evolution": None, "evolution_governance": None
}

# Strategy Lab summary; full research files are included when present in the state directory.
if len(sys.argv) > 3:
    data["strategy_lab"] = safe_json(Path(sys.argv[3])) or None
# Latest compact genetic-evolution report; the full archive remains in durable state.
if db_path.exists():
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    data["meta"] = {str(x["k"]): x["v"] for x in rows(conn, "SELECT k,v FROM meta")}
    data["summary"] = (rows(conn, """SELECT COUNT(*) AS total,
      COALESCE(SUM(CASE WHEN labeled=1 THEN 1 ELSE 0 END),0) AS labeled,
      COALESCE(SUM(CASE WHEN labeled=1 AND label=1 AND ret>0 THEN 1 ELSE 0 END),0) AS wins,
      COALESCE(SUM(CASE WHEN labeled=1 AND (label=0 OR ret<=0) THEN 1 ELSE 0 END),0) AS losses,
      AVG(CASE WHEN labeled=1 THEN ret END) AS avgRet FROM signals""") or [{}])[0]
    data["summary"]["openTracked"] = (rows(conn, "SELECT COUNT(*) n FROM tracked_signals WHERE status='OPEN'") or [{"n":0}])[0]["n"]
    data["summary"]["closedTracked"] = (rows(conn, "SELECT COUNT(*) n FROM tracked_signals WHERE status='CLOSED'") or [{"n":0}])[0]["n"]
    gate = (rows(conn, """SELECT SUM(CASE WHEN s.ret>0 THEN 1 ELSE 0 END) wins,
      SUM(CASE WHEN s.ret<0 THEN 1 ELSE 0 END) losses, COUNT(*) total
      FROM tracked_signals t JOIN signals s ON s.id=t.signal_id
      WHERE t.status='CLOSED' AND s.labeled=1 AND s.ret IS NOT NULL""") or [{}])[0]
    data["summary"].update({"gateWins":gate.get("wins") or 0,"gateLosses":gate.get("losses") or 0,
      "gateTotal":gate.get("total") or 0})
    data["signals"] = rows(conn, "SELECT id,ts,inst,price,atr,prob,rank,labeled,label,ret,hold,feature_ts FROM signals ORDER BY ts DESC LIMIT 10")
    # Export every closed and open tracked signal so dashboard analytics use the full history.
    # Preserve the entry-time market regime in the dashboard snapshot. The regime is
    # stored on tracked_signals when each signal is created; omitting it here made
    # every historical dashboard row appear as "Non renseigné".
    tracked_cols = ("id,signal_id,inst,status,entry,exit_price,sl,tp,prob,prob_source,opened_ts,ret,"
      "closed_ts,outcome,close_reason,hold,regime,entry_score,atr_pct,peak_price,trailing_sl,"
      "trail_active,trail_activation_atr,trail_distance_atr,max_hold_bars,exit_policy,"
      "risk_pct,reward_pct,risk_reward")
    open_tracked = rows(conn, f"SELECT {tracked_cols} FROM tracked_signals WHERE status='OPEN' ORDER BY opened_ts DESC")
    closed_tracked = rows(conn, f"SELECT {tracked_cols} FROM tracked_signals WHERE status='CLOSED' ORDER BY closed_ts DESC, id DESC")
    data["tracked"] = closed_tracked + open_tracked
    conn.close()

# Publish one canonical research report. Its layers contain the strategy,
# discovery-lab, regime-selector, exit-policy and model walk-forward results.
for name in ("meta.json", "comprehensive_backtest.json", "candle_patterns.json", "news_intelligence.json", "news_model.json",
             "feature_quality.json", "feature_importance.json", "feature_drift.json",
             "model_calibration.json", "regime_performance.json"):
    value = safe_json(root / name)
    if value is not None:
        if name == "candle_patterns.json" and isinstance(value, dict):
            coins = value.get("coins")
            if isinstance(coins, dict):
                value["coins"] = {
                    symbol: item for symbol, item in coins.items()
                    if isinstance(item, dict) and (
                        (isinstance(item.get("patterns"), list) and item["patterns"])
                        or any(k not in ("patterns", "score") and v not in (None, "", 0, [], {})
                               for k, v in item.items())
                    )
                }
        data["contents"][name] = value

# Expose the latest autonomous genetic-evolution report in the one-file dashboard.
# The canonical source is the discovery layer inside comprehensive_backtest.json.
comprehensive = data["contents"].get("comprehensive_backtest.json", {})
layers = comprehensive.get("layers", {}) if isinstance(comprehensive, dict) else {}
evolution = layers.get("autonomous_feature_discovery") if isinstance(layers, dict) else None
data["evolution"] = evolution if isinstance(evolution, dict) else None
governance = layers.get("evolution_governance") if isinstance(layers, dict) else None
data["evolution_governance"] = governance if isinstance(governance, dict) else None

out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",",":"), default=str), encoding="utf-8")
tmp.replace(out)
print(f"DASHBOARD_SNAPSHOT_OK bytes={out.stat().st_size} signals={len(data['signals'])} tracked={len(data['tracked'])}")
