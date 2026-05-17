import os
import sqlite3
import threading
import logging
import time
import io
import gzip
from datetime import datetime, timedelta, date
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

NASDAQ_URL     = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NYSE_URL       = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
BENCHMARK      = "SPY"
POLYGON_TOKEN  = os.environ.get("POLYGON_TOKEN", "")
POLYGON_BASE   = "https://api.polygon.io"

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

# ── Polygon bulk flat-file download ──────────────────────────────────────────
def get_trading_dates(n_days=380):
    """Get list of recent trading dates to fetch."""
    dates = []
    d = datetime.utcnow().date() - timedelta(days=1)
    while len(dates) < n_days:
        # Skip weekends
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    return sorted(dates)

def fetch_day(trading_date):
    """
    Fetch all US stock prices for a single trading date using
    Polygon's grouped daily endpoint — one call, entire market.
    Returns DataFrame with columns: symbol, close (adjusted)
    """
    ds = trading_date.strftime("%Y-%m-%d")
    url = (f"{POLYGON_BASE}/v2/aggs/grouped/locale/us/market/stocks/{ds}"
           f"?adjusted=true&apiKey={POLYGON_TOKEN}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 403:
            log.error("Polygon API key invalid or unauthorized")
            return None
        if r.status_code == 429:
            log.warning("Polygon rate limit — sleeping 60s")
            time.sleep(60)
            r = requests.get(url, timeout=30)
        if r.status_code != 200:
            log.warning(f"Polygon {ds}: HTTP {r.status_code}")
            return None

        data = r.json()
        if data.get("resultsCount", 0) == 0 or not data.get("results"):
            log.info(f"No results for {ds} (likely market holiday)")
            return None

        df = pd.DataFrame(data["results"])
        # 'T' = ticker, 'c' = close price
        df = df[["T", "c"]].rename(columns={"T": "symbol", "c": "close"})
        df = df.dropna()
        return df

    except Exception as e:
        log.warning(f"Polygon fetch error for {ds}: {e}")
        return None

def build_price_matrix():
    """
    Download ~252 trading days of data using Polygon grouped daily endpoint.
    Returns a DataFrame: rows=dates, columns=symbols, values=adjusted close.
    Each date = 1 API call. Total: ~252 calls for a full year.
    """
    dates = get_trading_dates(380)  # get extra to ensure 252 trading days
    frames = {}
    total = len(dates)

    for i, d in enumerate(dates):
        pct_done = 10 + (i / total) * 70
        if i % 10 == 0:
            set_prog(pct_done, f"Downloading market data: {d}  ({i+1}/{total} days)…")

        df = fetch_day(d)
        if df is not None:
            frames[d] = df.set_index("symbol")["close"]

        # Polygon free tier: 5 calls/minute — stay safe at 4/min
        time.sleep(15)

    if not frames:
        return None

    # Build matrix: index=date, columns=symbol
    matrix = pd.DataFrame(frames).T
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.sort_index()
    log.info(f"Price matrix: {len(matrix)} days x {len(matrix.columns)} symbols")
    return matrix

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
    if not POLYGON_TOKEN:
        set_prog(100, "Error: POLYGON_TOKEN environment variable not set.")
        return

    set_prog(0, "Fetching ticker universe…")
    universe = get_tickers()
    valid_syms = set(universe["sym"].tolist())
    log.info(f"Universe: {len(valid_syms)} tickers")

    set_prog(10, "Downloading market data from Polygon (this takes ~1 hour on free tier)…")
    matrix = build_price_matrix()

    if matrix is None or matrix.empty:
        set_prog(100, "Error: could not download market data from Polygon.")
        return

    set_prog(82, f"Computing RS ratings for {len(matrix.columns)} symbols…")

    # Filter to only NASDAQ/NYSE stocks we care about
    # (Polygon returns all US stocks including OTC etc)
    valid_cols = [c for c in matrix.columns if c in valid_syms]
    matrix = matrix[valid_cols + ([BENCHMARK] if BENCHMARK in matrix.columns else [])]
    log.info(f"Filtered matrix: {len(matrix.columns)} symbols")

    # Compute composite scores
    scores = {}
    for sym in matrix.columns:
        s = matrix[sym].dropna()
        score = composite(s)
        if not np.isnan(score):
            scores[sym] = score

    if not scores:
        set_prog(100, "Error: could not compute any RS scores.")
        return

    # Benchmark
    bench_score = scores.get(BENCHMARK, 0.0)

    # Rank 1-99
    series = pd.Series(scores)
    ranked = (series.rank(pct=True)*98+1).clip(1,99).round(0).astype(int)
    spy_pct = float((series < bench_score).mean()*98+1)

    set_prog(90, "Calculating performance metrics…")

    # Price and performance metrics
    def safe_pct(sym, n):
        s = matrix[sym].dropna()
        v = pct(s, n)
        return round(v*100, 2) if not np.isnan(v) else None

    set_prog(94, "Saving to database…")
    info = universe.set_index("sym")
    now  = datetime.utcnow().isoformat()
    rows = []

    for sym, rs in ranked.items():
        if sym == BENCHMARK:
            continue
        if sym not in valid_syms:
            continue
        r = info.loc[sym] if sym in info.index else pd.Series({"name":"","exch":""})
        last_price = matrix[sym].dropna().iloc[-1] if sym in matrix.columns else None
        rows.append((
            sym,
            r.get("name", ""),
            r.get("exch", ""),
            int(rs),
            round(float(rs) - spy_pct, 1),
            round(float(last_price), 2) if last_price else None,
            safe_pct(sym, 1),
            safe_pct(sym, 21),
            safe_pct(sym, 63),
            safe_pct(sym, 252),
            now
        ))

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
    log.info("No data — will wait for manual refresh trigger.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
