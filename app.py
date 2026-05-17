import os
import sqlite3
import threading
import logging
import time
import io
import json
import csv
from datetime import datetime, timedelta, date
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
DB  = Path("/tmp/rs.db")

NASDAQ_URL    = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NYSE_URL      = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
BENCHMARK     = "SPY"
POLYGON_TOKEN = os.environ.get("POLYGON_TOKEN", "")
POLYGON_BASE  = "https://api.polygon.io"

# We need 252 trading days for the IBD RS formula.
# We ask for 260 to have a small buffer for any missing data.
# Polygon only returns actual trading days so no weekend/holiday noise.
TARGET_TRADING_DAYS = 260

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
    # Price cache: one row per trading day, prices as JSON
    con.execute("""CREATE TABLE IF NOT EXISTS price_cache (
        trade_date TEXT PRIMARY KEY,
        prices     TEXT)""")
    con.commit()
    con.close()

# ── Ticker universe ───────────────────────────────────────────────────────────
# Words that identify non-stock securities to exclude
EXCLUDE_WORDS = [
    "ETF", "FUND", "TRUST", "WARRANT", "WARRANTS", "RIGHT", "RIGHTS",
    "UNIT", "UNITS", "NOTE", "NOTES", "DEBENTURE", "PREFERRED",
    "ACQUISITION", "BLANK CHECK", "TEST", "SYMBOL",
]
EXCLUDE_SUFFIXES = ["W", "R", "U", "Z"]  # warrant, right, unit, temp suffixes

def is_valid_stock(sym, name):
    """Filter out ETFs, warrants, rights, units, and other non-stocks."""
    sym  = str(sym).strip()
    name = str(name).upper()

    # Length check — real stocks are 1-5 chars, but 5-char often warrants
    if not sym or len(sym) > 5:
        return False

    # Common non-stock symbol patterns
    if any(c in sym for c in [".", "$", "+", "^", "-"]):
        return False

    # Warrants/rights often end in W, R (after 4+ char base)
    if len(sym) == 5 and sym[-1] in EXCLUDE_SUFFIXES:
        return False

    # Name-based exclusions
    for word in EXCLUDE_WORDS:
        if word in name:
            return False

    return True

def get_tickers():
    tickers = []
    for url, exch_map in [
        (NASDAQ_URL, {"": "NASDAQ"}),
        (NYSE_URL,   {"N": "NYSE", "A": "NYSE American"}),
    ]:
        try:
            r = requests.get(url, timeout=15)
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            if "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "sym", "Security Name": "name"})
                df["exch"] = "NASDAQ"
                # NASDAQ file has ETF column
                if "ETF" in df.columns:
                    df = df[df["ETF"].astype(str).str.strip() != "Y"]
            else:
                df = df.rename(columns={"ACT Symbol": "sym", "Security Name": "name"})
                df["exch"] = df["Exchange"].map(exch_map)
                df = df[df["exch"].notna()]
                # NYSE file has ETF column too
                if "ETF" in df.columns:
                    df = df[df["ETF"].astype(str).str.strip() != "Y"]

            df = df[df["sym"].notna()]
            df = df[~df["sym"].astype(str).str.startswith("File")]
            df = df[df.apply(lambda r: is_valid_stock(r["sym"], r["name"]), axis=1)]
            df = df[["sym", "name", "exch"]].drop_duplicates("sym")
            tickers.append(df)
        except Exception as e:
            log.warning(f"Ticker fetch error: {e}")

    if not tickers:
        return pd.DataFrame(columns=["sym","name","exch"])
    return pd.concat(tickers).drop_duplicates("sym").reset_index(drop=True)

# ── RS calculation (IBD formula) ──────────────────────────────────────────────
def pct(s, n):
    if len(s) < n + 1: return np.nan
    return (s.iloc[-1] / s.iloc[-n-1]) - 1

def composite(s):
    q1 = pct(s, 63)
    q2 = pct(s, 126) if len(s) > 126 else np.nan
    q3 = pct(s, 189) if len(s) > 189 else np.nan
    q4 = pct(s, 252) if len(s) > 252 else np.nan
    vals = [(q1,0.4),(q2,0.2),(q3,0.2),(q4,0.2)]
    good = [(v,w) for v,w in vals if not np.isnan(v)]
    if not good: return np.nan
    tw = sum(w for _,w in good)
    return sum(v*(w/tw) for v,w in good)

# ── Polygon helpers ───────────────────────────────────────────────────────────
def fetch_day_polygon(trading_date):
    """
    One API call = entire US market prices for that trading day.
    Polygon only has data for real trading days — no weekends or holidays.
    Returns dict {symbol: adj_close} or None.
    """
    ds  = trading_date.strftime("%Y-%m-%d")
    url = (f"{POLYGON_BASE}/v2/aggs/grouped/locale/us/market/stocks/{ds}"
           f"?adjusted=true&apiKey={POLYGON_TOKEN}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 403:
            log.error("Polygon: invalid API key")
            return None
        if r.status_code == 429:
            log.warning("Polygon rate limit — sleeping 65s")
            time.sleep(65)
            r = requests.get(url, timeout=30)
        if r.status_code != 200:
            log.warning(f"Polygon {ds}: HTTP {r.status_code}")
            return None
        data = r.json()
        if not data.get("results"):
            log.info(f"{ds}: no results (market holiday or future date)")
            return None
        # Store both close price and volume for each symbol
        # Format: {symbol: {"c": close, "v": volume}}
        return {
            item["T"]: {"c": item["c"], "v": item.get("v", 0)}
            for item in data["results"]
            if "T" in item and "c" in item
        }
    except Exception as e:
        log.warning(f"Polygon fetch error {ds}: {e}")
        return None

# ── Price cache ───────────────────────────────────────────────────────────────
def load_cached_dates():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT trade_date FROM price_cache ORDER BY trade_date").fetchall()
    con.close()
    return [r[0] for r in rows]

def save_day_to_cache(trade_date_str, prices_dict):
    con = sqlite3.connect(DB)
    con.execute("INSERT OR REPLACE INTO price_cache VALUES (?,?)",
                (trade_date_str, json.dumps(prices_dict)))
    con.commit()
    con.close()

def prune_old_cache(keep_n_days):
    """Keep only the most recent keep_n_days trading days."""
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT trade_date FROM price_cache ORDER BY trade_date DESC").fetchall()
    if len(rows) > keep_n_days:
        cutoff = rows[keep_n_days - 1][0]
        deleted = con.execute(
            "DELETE FROM price_cache WHERE trade_date < ?", (cutoff,)).rowcount
        log.info(f"Pruned {deleted} old cache rows, keeping {keep_n_days} days")
    con.commit()
    con.close()

def load_price_matrix(valid_syms):
    """
    Returns (price_matrix, volume_matrix) both as DataFrames.
    index=dates, columns=symbols
    """
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT trade_date, prices FROM price_cache ORDER BY trade_date").fetchall()
    con.close()
    if not rows:
        return None, None

    price_frames  = {}
    volume_frames = {}
    for trade_date, prices_json in rows:
        data = json.loads(prices_json)
        if data and isinstance(next(iter(data.values()), None), dict):
            # New format: {sym: {"c": price, "v": volume}}
            price_frames[trade_date]  = {s: v["c"] for s, v in data.items()}
            volume_frames[trade_date] = {s: v["v"] for s, v in data.items()}
        else:
            # Old format: {sym: price} — volume unknown
            price_frames[trade_date]  = data
            volume_frames[trade_date] = {}

    price_matrix = pd.DataFrame(price_frames).T
    price_matrix.index = pd.to_datetime(price_matrix.index)
    price_matrix = price_matrix.sort_index()

    vol_matrix = pd.DataFrame(volume_frames).T
    vol_matrix.index = pd.to_datetime(vol_matrix.index)
    vol_matrix = vol_matrix.sort_index()

    keep = [c for c in price_matrix.columns if c in valid_syms or c == BENCHMARK]
    price_matrix = price_matrix[keep]
    vol_keep = [c for c in keep if c in vol_matrix.columns]
    vol_matrix = vol_matrix[vol_keep] if vol_keep else pd.DataFrame()

    return price_matrix, vol_matrix

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
    universe   = get_tickers()
    valid_syms = set(universe["sym"].tolist())
    log.info(f"Universe: {len(valid_syms)} stocks (ETFs and warrants excluded)")

    today    = datetime.utcnow().date()
    # Don't fetch today until after 10pm UTC (~6pm ET, well after market close)
    end_date = today if datetime.utcnow().hour >= 22 else today - timedelta(days=1)

    cached_dates = load_cached_dates()

    if not cached_dates:
        # ── FIRST RUN: download full history ─────────────────────────────────
        # Start from enough days back to get TARGET_TRADING_DAYS
        # Since Polygon skips weekends/holidays naturally, we go back
        # ~1.4x target to be safe (365 calendar days ≈ 252 trading days)
        start_date = today - timedelta(days=370)
        mode = "full"
        log.info(f"First run: downloading from {start_date} to {end_date}")
    else:
        last_cached = date.fromisoformat(cached_dates[-1])
        if last_cached >= end_date:
            set_prog(5, "Cache is current — recomputing ratings…")
            mode = "compute_only"
            start_date = None
        else:
            start_date = last_cached + timedelta(days=1)
            mode = "incremental"
            log.info(f"Incremental: fetching {start_date} to {end_date}")

    # ── Download missing dates ────────────────────────────────────────────────
    if mode in ("full", "incremental"):
        # Build list of weekdays to try
        dates_to_try = []
        d = start_date
        while d <= end_date:
            if d.weekday() < 5:
                dates_to_try.append(d)
            d += timedelta(days=1)

        total = len(dates_to_try)
        if mode == "full":
            set_prog(5, f"First run: downloading ~{total} trading days. This takes ~{total//4} minutes…")
        else:
            set_prog(5, f"Fetching {total} new day(s)…")

        for i, d in enumerate(dates_to_try):
            pct_done = 5 + (i / max(total, 1)) * 72
            set_prog(pct_done, f"Downloading {d}  ({i+1}/{total})…")
            prices = fetch_day_polygon(d)
            if prices:
                save_day_to_cache(d.isoformat(), prices)
            # Polygon free tier: 5 calls/min → 13s between calls
            if i < total - 1:
                time.sleep(13)

        # Keep rolling window at TARGET_TRADING_DAYS
        prune_old_cache(TARGET_TRADING_DAYS)

    # ── Build matrix and compute ratings ─────────────────────────────────────
    set_prog(80, "Loading price matrix…")
    matrix, vol_matrix = load_price_matrix(valid_syms)

    if matrix is None or matrix.empty:
        set_prog(100, "Error: no price data in cache.")
        return

    log.info(f"Matrix: {len(matrix)} trading days x {len(matrix.columns)} symbols")
    set_prog(84, f"Applying IBD-style filters (price ≥ $10, avg vol ≥ 100k)…")

    # ── IBD-style universe filters ────────────────────────────────────────
    MIN_PRICE  = 10.0      # IBD excludes stocks under $10
    MIN_AVG_VOL = 100_000  # IBD excludes thinly traded stocks

    # Latest closing price filter
    last_prices = matrix.iloc[-1]  # most recent day
    price_ok    = set(last_prices[last_prices >= MIN_PRICE].index)

    # Average daily volume filter (last 50 days)
    vol_ok = set()
    if not vol_matrix.empty:
        recent_vol = vol_matrix.tail(50)
        avg_vol    = recent_vol.mean()
        vol_ok     = set(avg_vol[avg_vol >= MIN_AVG_VOL].index)
        # Always keep benchmark regardless
        vol_ok.add(BENCHMARK)
    else:
        # No volume data (old cache) — skip volume filter
        vol_ok = set(matrix.columns)

    filtered_syms = (price_ok & vol_ok) | {BENCHMARK}
    filtered_cols = [c for c in matrix.columns if c in filtered_syms]
    matrix        = matrix[filtered_cols]

    log.info(f"After filters: {len(filtered_cols)} symbols "
             f"(removed {len(valid_syms) - len(filtered_cols)} below $10 or low volume)")

    set_prog(87, f"Computing RS ratings for {len(filtered_cols)} symbols…")

    scores = {}
    for sym in matrix.columns:
        s     = matrix[sym].dropna()
        score = composite(s)
        if not np.isnan(score):
            scores[sym] = score

    if not scores:
        set_prog(100, "Error: could not compute any RS scores.")
        return

    bench_score = scores.get(BENCHMARK, 0.0)
    series      = pd.Series(scores)
    ranked      = (series.rank(pct=True)*98+1).clip(1,99).round(0).astype(int)
    spy_pct     = float((series < bench_score).mean()*98+1)

    set_prog(94, "Saving ratings to database…")
    info = universe.set_index("sym")
    now  = datetime.utcnow().isoformat()
    rows = []

    for sym, rs in ranked.items():
        if sym == BENCHMARK or sym not in valid_syms:
            continue
        r = info.loc[sym] if sym in info.index else pd.Series({"name":"","exch":""})
        s = matrix[sym].dropna()
        last_price = float(s.iloc[-1]) if len(s) else None
        rows.append((
            sym,
            r.get("name",""),
            r.get("exch",""),
            int(rs),
            round(float(rs) - spy_pct, 1),
            round(last_price, 2) if last_price else None,
            round(pct(s,1)*100,   2) if not np.isnan(pct(s,1))   else None,
            round(pct(s,21)*100,  2) if not np.isnan(pct(s,21))  else None,
            round(pct(s,63)*100,  2) if not np.isnan(pct(s,63))  else None,
            round(pct(s,252)*100, 2) if not np.isnan(pct(s,252)) else None,
            now
        ))

    cached_count = len(load_cached_dates())
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM stocks")
    con.executemany("""INSERT OR REPLACE INTO stocks
        (symbol,name,exchange,rs,rs_mkt,price,d1,m1,m3,y1,updated)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""", rows)
    con.execute("INSERT OR REPLACE INTO meta VALUES('updated',?)", (now,))
    con.execute("INSERT OR REPLACE INTO meta VALUES('total',?)",   (str(len(rows)),))
    con.execute("INSERT OR REPLACE INTO meta VALUES('cached_days',?)", (str(cached_count),))
    con.execute("INSERT OR REPLACE INTO meta VALUES('mode',?)", (mode,))
    con.commit()
    con.close()
    set_prog(100, f"Done! Rated {len(rows)} stocks using {cached_count} trading days of data.")

# ── API routes ────────────────────────────────────────────────────────────────
def build_query(request_args):
    """Parse filter params and return (conditions, params) for SQL WHERE."""
    col   = request_args.get("sort","rs")
    dire  = "ASC" if request_args.get("dir","desc")=="asc" else "DESC"
    q     = request_args.get("q","").strip()
    exch  = request_args.get("exchange","all")
    minrs = request_args.get("min_rs")
    maxrs = request_args.get("max_rs")
    minp  = request_args.get("min_price")
    maxp  = request_args.get("max_price")
    miny1 = request_args.get("min_y1")
    minm3 = request_args.get("min_m3")

    safe = {"symbol","name","exchange","rs","rs_mkt","price","d1","m1","m3","y1"}
    if col not in safe: col = "rs"

    conds, params = ["rs IS NOT NULL"], []
    if q:
        conds.append("(symbol LIKE ? OR name LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if exch and exch != "all":
        conds.append("exchange=?"); params.append(exch)
    if minrs: conds.append("rs>=?");    params.append(float(minrs))
    if maxrs: conds.append("rs<=?");    params.append(float(maxrs))
    if minp:  conds.append("price>=?"); params.append(float(minp))
    if maxp:  conds.append("price<=?"); params.append(float(maxp))
    if miny1: conds.append("y1>=?");    params.append(float(miny1))
    if minm3: conds.append("m3>=?");    params.append(float(minm3))
    return col, dire, " AND ".join(conds), params

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ratings")
def api_ratings():
    if not DB.exists():
        return jsonify({"rows":[],"total":0,"meta":{}})
    page = max(1, int(request.args.get("page",1)))
    pp   = min(500, max(10, int(request.args.get("per_page",100))))
    col, dire, where, params = build_query(request.args)
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

@app.route("/api/export")
def api_export():
    """Export current filtered view as CSV."""
    if not DB.exists():
        return "No data", 404
    col, dire, where, params = build_query(request.args)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        f"SELECT symbol,name,exchange,rs,rs_mkt,price,d1,m1,m3,y1 "
        f"FROM stocks WHERE {where} ORDER BY {col} {dire}",
        params).fetchall()
    con.close()

    def generate():
        header = ["Symbol","Company","Exchange","RS Rating","vs SPY",
                  "Price","1D%","1M%","3M%","1Y%"]
        yield ",".join(header) + "\n"
        for r in rows:
            row = [
                r["symbol"],
                f'"{r["name"]}"',
                r["exchange"],
                str(r["rs"]) if r["rs"] is not None else "",
                str(r["rs_mkt"]) if r["rs_mkt"] is not None else "",
                str(r["price"]) if r["price"] is not None else "",
                str(r["d1"]) if r["d1"] is not None else "",
                str(r["m1"]) if r["m1"] is not None else "",
                str(r["m3"]) if r["m3"] is not None else "",
                str(r["y1"]) if r["y1"] is not None else "",
            ]
            yield ",".join(row) + "\n"

    ts = datetime.utcnow().strftime("%Y%m%d")
    filename = f"rs_ratings_{ts}.csv"
    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
