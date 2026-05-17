import os
import sqlite3
import threading
import logging
import time
import io
from datetime import datetime, timedelta
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
DB  = Path("/tmp/rs.db")

NASDAQ_URL   = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NYSE_URL     = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
BENCHMARK    = "SPY"
TIINGO_TOKEN = os.environ.get("TIINGO_TOKEN", "")
TIINGO_BASE  = "https://api.tiingo.com/tiingo/daily"
BATCH        = 50   # tickers per Tiingo bulk request

_prog = {"pct": 0, "msg": "Idle", "running": False, "error": None}
_lock = threading.Lock()

def set_prog(pct, msg):
    with _lock:
        _prog["pct"] = pct
        _prog["msg"] = msg
    log.info(f"[{pct:.0f}%] {msg}")

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS stocks (
        symbol TEXT PRIMARY KEY, name TEXT, exchange TEXT,
        rs INTEGER, rs_mkt REAL, price REAL,
        d1 REAL, m1 REAL, m3 REAL, y1 REAL, updated TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    con.commit()
    con.close()

# ── Ticker universe ───────────────────────────────────────────────────────────
def get_tickers():
    tickers = []
    for url, exch_map in [
        (NASDAQ_URL, {"": "NASDAQ"}),
        (NYSE_URL,   {"N": "NYSE", "A": "NYSE MKT"}),
    ]:
        try:
            r = requests.get(url, timeout=15)
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            if "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "sym", "Security Name": "name"})
                df["exch"] = "NASDAQ"
            else:
                df = df.rename(columns={"ACT Symbol": "sym", "Security Name": "name"})
                df["exch"] = df["Exchange"].map(exch_map)
                df = df[df["exch"].notna()]
            df = df[df["sym"].notna()]
            df = df[~df["sym"].astype(str).str.contains(r"[.\$\+\^]", regex=True)]
            df = df[df["sym"].astype(str).str.len() <= 5]
            df = df[~df["sym"].astype(str).str.startswith("File")]
            df = df[["sym", "name", "exch"]].drop_duplicates("sym")
            tickers.append(df)
        except Exception as e:
            log.warning(f"Ticker fetch error: {e}")
    if not tickers:
        return pd.DataFrame(columns=["sym","name","exch"])
    return pd.concat(tickers).drop_duplicates("sym").reset_index(drop=True)

# ── RS calculation ────────────────────────────────────────────────────────────
def pct(s, n):
    if len(s) < n + 1: return np.nan
    return (s.iloc[-1] / s.iloc[-n-1]) - 1

def composite(s):
    # IBD formula: 0.4*ROC(63) + 0.2*ROC(126) + 0.2*ROC(189) + 0.2*ROC(252)
    q1 = pct(s, 63)
    q2 = pct(s, 126) if len(s) > 126 else np.nan
    q3 = pct(s, 189) if len(s) > 189 else np.nan
    q4 = pct(s, 252) if len(s) > 252 else np.nan
    vals = [(q1,0.4),(q2,0.2),(q3,0.2),(q4,0.2)]
    good = [(v,w) for v,w in vals if not np.isnan(v)]
    if not good: return np.nan
    tw = sum(w for _,w in good)
    return sum(v*(w/tw) for v,w in good)

# ── Tiingo data fetching ──────────────────────────────────────────────────────
def tiingo_headers():
    return {
        "Authorization": f"Token {TIINGO_TOKEN}",
        "Content-Type": "application/json",
    }

def fetch_tiingo_batch(symbols):
    """
    Fetch end-of-day prices for a list of symbols from Tiingo.
    Returns dict: {symbol: pd.Series of adjClose prices}
    """
    end_date   = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=380)).strftime("%Y-%m-%d")
    result = {}

    for sym in symbols:
        try:
            url = (f"{TIINGO_BASE}/{sym}/prices"
                   f"?startDate={start_date}&endDate={end_date}"
                   f"&resampleFreq=daily&token={TIINGO_TOKEN}")
            r = requests.get(url, headers=tiingo_headers(), timeout=15)

            if r.status_code == 404:
                continue  # ticker not found on Tiingo
            if r.status_code == 429:
                log.warning("Tiingo rate limit hit — sleeping 60s")
                time.sleep(60)
                r = requests.get(url, headers=tiingo_headers(), timeout=15)
            if r.status_code != 200:
                log.debug(f"{sym}: HTTP {r.status_code}")
                continue

            data = r.json()
            if not data:
                continue

            closes = pd.Series(
                [d.get("adjClose") or d.get("close") for d in data],
                index=pd.to_datetime([d["date"] for d in data])
            ).dropna().sort_index()

            if len(closes) >= 63:
                result[sym] = closes

        except Exception as e:
            log.debug(f"{sym} error: {e}")

        time.sleep(0.05)  # polite delay — Tiingo allows ~50 req/hr on free tier

    return result

# ── Full refresh ──────────────────────────────────────────────────────────────
def run_refresh():
    with _lock:
        if _prog["running"]: return
        _prog.update({"running":True,"pct":0,"msg":"Starting…","error":None})
    try:
        _do_refresh()
    except Exception as e:
        log.exception("Refresh error")
        with _lock: _prog["error"] = str(e)
    finally:
        with _lock: _prog["running"] = False

def _do_refresh():
    if not TIINGO_TOKEN:
        set_prog(100, "Error: TIINGO_TOKEN environment variable not set.")
        return

    set_prog(0, "Fetching ticker list…")
    df   = get_tickers()
    syms = df["sym"].tolist()
    set_prog(3, f"Got {len(syms)} tickers. Fetching benchmark (SPY)…")

    bench_data  = fetch_tiingo_batch([BENCHMARK])
    bench_score = composite(bench_data[BENCHMARK]) if BENCHMARK in bench_data else 0.0
    log.info(f"SPY benchmark score: {bench_score:.4f}")

    scores, prices = {}, {}
    total    = len(syms)
    ok_count = 0

    batches = [syms[i:i+BATCH] for i in range(0, total, BATCH)]
    n = len(batches)

    for i, batch in enumerate(batches):
        pct_done = 3 + (i / n) * 86
        set_prog(pct_done, f"Batch {i+1}/{n}  (✓{ok_count} rated so far)…")

        data = fetch_tiingo_batch(batch)

        for sym, closes in data.items():
            scores[sym] = composite(closes)
            prices[sym] = {
                "price": round(float(closes.iloc[-1]), 2),
                "d1":  round(pct(closes, 1)   * 100, 2),
                "m1":  round(pct(closes, 21)  * 100, 2),
                "m3":  round(pct(closes, 63)  * 100, 2),
                "y1":  round(pct(closes, 252) * 100, 2),
            }
            ok_count += 1

        # Tiingo free tier: ~50 requests/hour per token
        # Each batch of 50 = 50 requests, so pause between batches
        time.sleep(2)

    set_prog(90, f"Ranking {ok_count} stocks…")
    if ok_count == 0:
        set_prog(100, "Error: no data returned from Tiingo. Check your API token.")
        return

    series = pd.Series(scores).dropna()
    ranked = (series.rank(pct=True)*98+1).clip(1,99).round(0).astype(int)
    spy_pct = float((series < bench_score).mean()*98+1)

    set_prog(94, "Saving to database…")
    info = df.set_index("sym")
    now  = datetime.utcnow().isoformat()
    rows = []
    for sym, rs in ranked.items():
        r = info.loc[sym] if sym in info.index else pd.Series({"name":"","exch":""})
        p = prices.get(sym, {})
        rows.append((sym, r.get("name",""), r.get("exch",""),
                     int(rs), round(float(rs)-spy_pct, 1),
                     p.get("price"), p.get("d1"), p.get("m1"),
                     p.get("m3"),    p.get("y1"),  now))

    con = sqlite3.connect(DB)
    con.execute("DELETE FROM stocks")
    con.executemany("""INSERT OR REPLACE INTO stocks
        (symbol,name,exchange,rs,rs_mkt,price,d1,m1,m3,y1,updated)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""", rows)
    con.execute("INSERT OR REPLACE INTO meta VALUES('updated',?)", (now,))
    con.execute("INSERT OR REPLACE INTO meta VALUES('total',?)",   (str(len(rows)),))
    con.commit()
    con.close()
    set_prog(100, f"Done! Rated {len(rows)} stocks.")

# ── API routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ratings")
def api_ratings():
    if not DB.exists():
        return jsonify({"rows":[],"total":0,"meta":{}})
    page  = max(1, int(request.args.get("page",1)))
    pp    = min(500, max(10, int(request.args.get("per_page",100))))
    col   = request.args.get("sort","rs")
    dire  = "ASC" if request.args.get("dir","desc")=="asc" else "DESC"
    q     = request.args.get("q","").strip()
    exch  = request.args.get("exchange","all")
    minrs = request.args.get("min_rs")
    maxrs = request.args.get("max_rs")
    minp  = request.args.get("min_price")

    safe = {"symbol","name","exchange","rs","rs_mkt","price","d1","m1","m3","y1"}
    if col not in safe: col = "rs"

    conds, params = ["rs IS NOT NULL"], []
    if q:
        conds.append("(symbol LIKE ? OR name LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if exch and exch != "all":
        conds.append("exchange=?"); params.append(exch)
    if minrs: conds.append("rs>=?"); params.append(float(minrs))
    if maxrs: conds.append("rs<=?"); params.append(float(maxrs))
    if minp:  conds.append("price>=?"); params.append(float(minp))

    where  = " AND ".join(conds)
    offset = (page-1)*pp
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    total = con.execute(f"SELECT COUNT(*) FROM stocks WHERE {where}", params).fetchone()[0]
    rows  = con.execute(
        f"SELECT symbol,name,exchange,rs,rs_mkt,price,d1,m1,m3,y1 "
        f"FROM stocks WHERE {where} ORDER BY {col} {dire} LIMIT ? OFFSET ?",
        params+[pp, offset]).fetchall()
    meta  = dict(con.execute("SELECT k,v FROM meta").fetchall())
    con.close()
    return jsonify({"rows":[dict(r) for r in rows],"total":total,"meta":meta})

@app.route("/api/stock/<sym>")
def api_stock(sym):
    if not DB.exists(): return jsonify({"error":"No data"}), 404
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM stocks WHERE symbol=?", (sym.upper(),)).fetchone()
    con.close()
    return jsonify(dict(row)) if row else (jsonify({"error":"Not found"}), 404)

@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    threading.Thread(target=run_refresh, daemon=True).start()
    return jsonify({"status":"started"})

@app.route("/api/progress")
def api_progress():
    with _lock: return jsonify(dict(_prog))

@app.route("/api/stats")
def api_stats():
    if not DB.exists(): return jsonify({})
    con = sqlite3.connect(DB)
    r = con.execute("""SELECT COUNT(*), AVG(rs),
        SUM(CASE WHEN rs>=90 THEN 1 ELSE 0 END),
        SUM(CASE WHEN rs>=80 THEN 1 ELSE 0 END),
        SUM(CASE WHEN rs_mkt>0 THEN 1 ELSE 0 END)
        FROM stocks WHERE rs IS NOT NULL""").fetchone()
    con.close()
    return jsonify({"total":r[0],"avg_rs":round(r[1] or 0,1),
                    "top10":r[2],"top20":r[3],"above_mkt":r[4]})

# ── Startup ───────────────────────────────────────────────────────────────────
init_db()

sched = BackgroundScheduler()
sched.add_job(run_refresh, CronTrigger(day_of_week="mon-fri", hour=17, minute=0))
sched.start()

con = sqlite3.connect(DB)
has_data = con.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
con.close()
if not has_data:
    log.info("No data — starting first refresh…")
    threading.Thread(target=run_refresh, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
