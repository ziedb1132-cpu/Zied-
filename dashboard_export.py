#!/usr/bin/env python3
"""Create a compact, read-only dashboard snapshot from the completed paper state."""
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
    "contents": {}
}
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
    tracked_cols = "id,inst,status,entry,exit_price,sl,tp,prob,opened_ts,ret,closed_ts,outcome,close_reason,hold,regime,entry_score"
    open_tracked = rows(conn, f"SELECT {tracked_cols} FROM tracked_signals WHERE status='OPEN' ORDER BY opened_ts DESC")
    closed_tracked = rows(conn, f"SELECT {tracked_cols} FROM tracked_signals WHERE status='CLOSED' ORDER BY closed_ts DESC, id DESC")
    data["tracked"] = closed_tracked + open_tracked
    conn.close()

for name in ("meta.json","strategy_evolution.json","strategy_discovery_lab.json","autonomous_strategy_report.json","strategy_meta_selector_report.json","top_gainer_learning.json","entry_method_memory.json","candle_patterns.json","news_sentiment.json"):
    value = safe_json(root / name)
    if value is not None:
        data["contents"][name] = value
for name in ("meta.json","strategy_evolution.json","strategy_discovery_lab.json","top_gainer_learning.json","entry_method_memory.json","candle_patterns.json","news_sentiment.json"):
    if name in data["contents"]:
        data["contents"][name] = data["contents"][name]
data["contents"]["paper.db"] = "قاعدة البيانات محفوظة في مخزن الحالة؛ لا يتم تنزيلها إلى المتصفح."
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",",":"), default=str), encoding="utf-8")
tmp.replace(out)
print(f"DASHBOARD_SNAPSHOT_OK bytes={out.stat().st_size} signals={len(data['signals'])} tracked={len(data['tracked'])}")
