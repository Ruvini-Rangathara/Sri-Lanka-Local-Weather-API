from flask import Flask, request, jsonify
from pathlib import Path
from datetime import date, timedelta
import os, json, time, math
import numpy as np
import pandas as pd
import requests
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from prophet import Prophet
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ============================================================
# CONFIG (with sensible defaults; override via environment)
# ============================================================
BASE = Path(__file__).parent.resolve()
DATA_DIR   = BASE / "data"
CACHE_DIR  = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
for p in [DATA_DIR, CACHE_DIR, MODELS_DIR / "T2M", MODELS_DIR / "PRECIP"]:
    p.mkdir(parents=True, exist_ok=True)

# Sri Lanka bbox (approx)
SL_LAT_MIN = float(os.getenv("SL_LAT_MIN", "5.8"))
SL_LAT_MAX = float(os.getenv("SL_LAT_MAX", "10.1"))
SL_LON_MIN = float(os.getenv("SL_LON_MIN", "79.5"))
SL_LON_MAX = float(os.getenv("SL_LON_MAX", "82.1"))

# POWER daily grid ~0.5°
GRID_STEP  = float(os.getenv("GRID_STEP", "0.5"))

# Shorter history by default (faster). You can set START_DATE=2000-01-01 if you want longer.
START_DATE = os.getenv("START_DATE", "2018-01-01")
# Use yesterday to avoid NRT day that may be incomplete
END_DATE   = os.getenv("END_DATE", (date.today() - timedelta(days=1)).isoformat())

# NASA POWER (daily point)
POWER_URL  = "https://power.larc.nasa.gov/api/temporal/daily/point"
COMMUNITY  = os.getenv("COMMUNITY", "AG")

# Minimal variables (fast): temperature + precipitation (corrected if available)
PARAMS     = os.getenv("POWER_PARAMS", "T2M,PRECTOTCORR")

# Forecast horizon
MAX_HORIZON = int(os.getenv("MAX_HORIZON", "30"))

# Prewarm options (optional)
PREWARM_WORKERS = int(os.getenv("PREWARM_WORKERS", "4"))
PREWARM_SLEEP   = float(os.getenv("PREWARM_SLEEP", "0.05"))

app = Flask(__name__)

# ============================================================
# HTTP session with retries (stability)
# ============================================================
def make_session():
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s
HTTP = make_session()

# ============================================================
# Utility helpers
# ============================================================
def ok(data):         return jsonify({"code": 200, "message": "OK", "data": data})
def bad_request(msg): return jsonify({"code": 400, "message": msg, "data": None}), 400

def snap_to_grid(lat: float, lon: float, step: float = GRID_STEP):
    return round(round(lat / step) * step, 3), round(round(lon / step) * step, 3)

def clamp_bbox(val, vmin, vmax):
    return max(vmin, min(vmax, val))

def cache_path_for(lat: float, lon: float) -> Path:
    return CACHE_DIR / f"power_daily_lat{lat}_lon{lon}_{COMMUNITY}_{START_DATE}_{END_DATE}.csv"

def model_paths(var: str, lat: float, lon: float):
    pdir = MODELS_DIR / var
    model_pkl = pdir / f"model_lat{lat}_lon{lon}.joblib"
    meta_json = pdir / f"meta_lat{lat}_lon{lon}.json"
    return model_pkl, meta_json

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

# ============================================================
# NASA POWER fetch & cache (minimal params, with fallback)
# ============================================================
def _fetch(lat: float, lon: float, params: str) -> dict:
    url = (
        f"{POWER_URL}"
        f"?parameters={params}"
        f"&community={COMMUNITY}"
        f"&start={START_DATE.replace('-','')}&end={END_DATE.replace('-','')}"
        f"&latitude={lat}&longitude={lon}"
        f"&time-standard=UTC"
        f"&format=JSON"
    )
    r = HTTP.get(url, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"POWER {r.status_code}: {r.text[:300]}")
    return r.json().get("properties", {}).get("parameter", {})

def fetch_power_point(lat: float, lon: float) -> pd.DataFrame:
    params_to_try = [PARAMS]
    if "PRECTOTCORR" in PARAMS and "PRECTOT" not in PARAMS:
        params_to_try.append(PARAMS.replace("PRECTOTCORR", "PRECTOT"))
    last_err = None
    for p in params_to_try:
        try:
            j = _fetch(lat, lon, p)
            if "T2M" not in j or len(j["T2M"]) == 0:
                raise RuntimeError("No T2M values returned.")
            df = pd.DataFrame({"date": list(j["T2M"].keys())})
            for k, v in j.items():
                df[k] = [v.get(d, np.nan) for d in df["date"]]
            if "PRECTOTCORR" in df.columns and not pd.isna(df["PRECTOTCORR"]).all():
                df = df.rename(columns={"PRECTOTCORR": "PRECIP"})
            elif "PRECTOT" in df.columns:
                df = df.rename(columns={"PRECTOT": "PRECIP"})
            else:
                df["PRECIP"] = np.nan
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            for col in ["T2M", "PRECIP"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[["date", "T2M", "PRECIP"]]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"POWER fetch failed: {last_err}")

def get_history(lat: float, lon: float) -> pd.DataFrame:
    cpath = cache_path_for(lat, lon)
    if cpath.exists():
        return pd.read_csv(cpath, parse_dates=["date"])
    df = fetch_power_point(lat, lon)
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df.to_csv(cpath, index=False)
    return df

# ============================================================
# Modeling (Prophet-only for speed & simplicity)
# ============================================================
def train_prophet(series: pd.DataFrame):
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(series)
    return m

def train_and_select(df: pd.DataFrame, var: str):
    # Prophet champion; validates on recent holdout
    series = df[["date", var]].dropna().sort_values("date").rename(columns={"date": "ds", var: "y"})
    if series.empty or len(series) < 300:
        raise RuntimeError(f"Not enough history to train {var} (need >= 300 days).")
    last_date = series["ds"].max()
    cutoff = last_date - pd.Timedelta(days=365)
    train = series[series["ds"] <= cutoff].copy()
    test  = series[series["ds"] >  cutoff].copy()
    if len(test) < 30:
        cutoff = last_date - pd.Timedelta(days=180)
        train = series[series["ds"] <= cutoff].copy()
        test  = series[series["ds"] >  cutoff].copy()
    m_prop = train_prophet(train)
    pred_prop = m_prop.predict(test[["ds"]])["yhat"].values
    _rmse = rmse(test["y"].values, pred_prop)
    # Refit on full series
    model = train_prophet(series)
    meta = {
        "algo": "prophet",
        "rmse_prop": float(_rmse),
        "cutoff": str(pd.to_datetime(cutoff).date()),
        "last_date": str(pd.to_datetime(last_date).date()),
        "var": var
    }
    return "prophet", model, meta

def save_model(var: str, lat: float, lon: float, algo: str, model, meta: dict):
    model_pkl, meta_json = model_paths(var, lat, lon)
    joblib.dump(model, model_pkl)
    with open(meta_json, "w") as f:
        json.dump({"algo": algo, **meta, "lat": lat, "lon": lon}, f, indent=2)

def load_model(var: str, lat: float, lon: float):
    model_pkl, meta_json = model_paths(var, lat, lon)
    if model_pkl.exists() and meta_json.exists():
        return joblib.load(model_pkl), json.load(open(meta_json))
    return None, None

def ensure_model(var: str, lat: float, lon: float):
    model, meta = load_model(var, lat, lon)
    hist = get_history(lat, lon)
    hist_last = str(pd.to_datetime(hist["date"].max()).date())
    need = (model is None or meta is None or meta.get("last_date") != hist_last)
    if need:
        algo, model, meta2 = train_and_select(hist, var)
        save_model(var, lat, lon, algo, model, meta2)
        model, meta = load_model(var, lat, lon)
    return model, meta, hist

def forecast_30d(var: str, model, meta: dict):
    horizon = MAX_HORIZON
    last_date = pd.to_datetime(meta["last_date"])
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({"ds": dates})
    pred = model.predict(future)
    out = pd.DataFrame({
        "date": pred["ds"].dt.strftime("%Y-%m-%d"),
        "yhat": pred["yhat"].astype(float),
        "lo": pred["yhat_lower"].astype(float),
        "hi": pred["yhat_upper"].astype(float)
    })
    return out

# ============================================================
# Small-village interpolation (bilinear)
# ============================================================
def _neighbors(lat, lon, step):
    # compute lower grid corner
    lat0 = math.floor(lat / step) * step
    lon0 = math.floor(lon / step) * step
    lat1 = lat0 + step
    lon1 = lon0 + step
    # clamp to bbox; if beyond max, shift one step inward
    lat0 = clamp_bbox(lat0, SL_LAT_MIN, SL_LAT_MAX)
    lat1 = clamp_bbox(lat1, SL_LAT_MIN, SL_LAT_MAX)
    lon0 = clamp_bbox(lon0, SL_LON_MIN, SL_LON_MAX)
    lon1 = clamp_bbox(lon1, SL_LON_MIN, SL_LON_MAX)
    if lat0 == lat1 and lat0 > SL_LAT_MIN: lat0 -= step
    if lon0 == lon1 and lon0 > SL_LON_MIN: lon0 -= step
    return [round(lat0,3), round(lat1,3)], [round(lon0,3), round(lon1,3)]

def _bilinear(lat, lon, lats, lons, q11, q21, q12, q22):
    # lats = [low, high], lons = [low, high]
    dx = (lons[1] - lons[0]) or 1e-12
    dy = (lats[1] - lats[0]) or 1e-12
    x = (lon - lons[0]) / dx
    y = (lat - lats[0]) / dy
    return (1-x)*(1-y)*q11 + x*(1-y)*q21 + (1-x)*y*q12 + x*y*q22

def interpolated_forecast(lat, lon, vars_req):
    # 1) pick the four neighboring POWER grid cells
    lats, lons = _neighbors(lat, lon, GRID_STEP)
    corners = [(lats[0], lons[0]), (lats[0], lons[1]),
               (lats[1], lons[0]), (lats[1], lons[1])]

    # 2) get per-corner forecasts
    per_var = {}
    for var in vars_req:
        corner_fc = {}
        for (la, lo) in corners:
            model, meta, _ = ensure_model(var, la, lo)
            corner_fc[(la, lo)] = forecast_30d(var, model, meta)
        per_var[var] = corner_fc

    # 3) align by date and bilinear-interpolate each column
    dates = per_var[vars_req[0]][corners[0]]["date"].tolist()
    out = []
    for i, d in enumerate(dates):
        row = {"date": d}
        for var in vars_req:
            q11 = per_var[var][corners[0]].iloc[i]  # (lat_low, lon_low)
            q21 = per_var[var][corners[1]].iloc[i]  # (lat_low, lon_high)
            q12 = per_var[var][corners[2]].iloc[i]  # (lat_high, lon_low)
            q22 = per_var[var][corners[3]].iloc[i]  # (lat_high, lon_high)

            yhat = _bilinear(lat, lon, lats, lons, q11["yhat"], q21["yhat"], q12["yhat"], q22["yhat"])
            lo   = _bilinear(lat, lon, lats, lons, q11["lo"],   q21["lo"],   q12["lo"],   q22["lo"])
            hi   = _bilinear(lat, lon, lats, lons, q11["hi"],   q21["hi"],   q12["hi"],   q22["hi"])

            if var == "T2M":
                row.update({"t2m": float(yhat), "t2m_lo": float(lo), "t2m_hi": float(hi), "t2m_model": "interp(4-corner)"})
            elif var == "PRECIP":
                row.update({"precip": float(max(0.0, yhat)),
                            "precip_lo": float(max(0.0, lo)),
                            "precip_hi": float(max(0.0, hi)),
                            "precip_model": "interp(4-corner)"})
        out.append(row)

    return dates[0], dates[-1], out, corners

def interpolated_history(lat, lon, qdate, vars_req):
    # Use historical cached daily values at four corners, then bilinear interpolate
    lats, lons = _neighbors(lat, lon, GRID_STEP)
    corners = [(lats[0], lons[0]), (lats[0], lons[1]),
               (lats[1], lons[0]), (lats[1], lons[1])]
    row = {"date": qdate}
    for var in vars_req:
        vals = []
        for (la, lo) in corners:
            df = get_history(la, lo)
            rec = df[df["date"] == pd.to_datetime(qdate)]
            if rec.empty:
                vals.append((np.nan, la, lo))
            else:
                vals.append((float(rec[var]), la, lo))
        # If any corner missing, we still interpolate with NaNs -> result could be NaN
        qmap = { (la,lo): v for (v, la, lo) in vals }
        v = _bilinear(
            lat, lon, lats, lons,
            qmap.get((lats[0], lons[0]), np.nan),
            qmap.get((lats[0], lons[1]), np.nan),
            qmap.get((lats[1], lons[0]), np.nan),
            qmap.get((lats[1], lons[1]), np.nan),
        )
        if var == "T2M":
            row.update({"t2m": None if np.isnan(v) else float(v), "t2m_model": "interp(4-corner)"})
        elif var == "PRECIP":
            vv = 0.0 if np.isnan(v) else max(0.0, float(v))
            row.update({"precip": vv, "precip_model": "interp(4-corner)"})
    return row, corners

# ============================================================
# Prewarm (parallel, optional)
# ============================================================
def build_snapped_grid():
    lats = sorted({ round(round(x/GRID_STEP)*GRID_STEP, 3)
                    for x in np.arange(SL_LAT_MIN, SL_LAT_MAX + 1e-9, GRID_STEP) })
    lons = sorted({ round(round(x/GRID_STEP)*GRID_STEP, 3)
                    for x in np.arange(SL_LON_MIN, SL_LON_MAX + 1e-9, GRID_STEP) })
    return [(la, lo) for la in lats for lo in lons]

def _prewarm_cell(args):
    lat, lon, var = args
    try:
        ensure_model(var, lat, lon)
        time.sleep(PREWARM_SLEEP)
        return (lat, lon, var, "ok")
    except Exception as e:
        return (lat, lon, var, f"warn: {e}")

def prewarm_models(vars_list=("T2M","PRECIP")):
    grid = build_snapped_grid()
    tasks = [(la, lo, var) for la, lo in grid for var in vars_list]
    print(f"[prewarm] {len(grid)} cells × {len(vars_list)} vars = {len(tasks)} tasks, workers={PREWARM_WORKERS}")
    done = 0
    with ThreadPoolExecutor(max_workers=PREWARM_WORKERS) as ex:
        futures = [ex.submit(_prewarm_cell, t) for t in tasks]
        for f in as_completed(futures):
            done += 1
            if done % 10 == 0:
                lat, lon, var, msg = f.result()
                print(f"[prewarm] {done}/{len(tasks)} last=({lat},{lon},{var}) {msg}")

# ============================================================
# Routes
# ============================================================
@app.route("/health")
def health():
    return ok({
        "status": "healthy",
        "community": COMMUNITY,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "grid_step_deg": GRID_STEP
    })

@app.route("/forecast", methods=["GET"])
def forecast_route():
    # Parse inputs
    try:
        lat = float(request.args.get("lat", ""))
        lon = float(request.args.get("lon", ""))
    except ValueError:
        return bad_request("lat and lon must be numbers")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return bad_request("lat/lon out of range")
    vars_req = request.args.get("vars", "T2M,PRECIP")
    vars_req = [v.strip().upper() for v in vars_req.split(",") if v.strip()]
    allowed = {"T2M", "PRECIP"}
    for v in vars_req:
        if v not in allowed:
            return bad_request(f"Unsupported var '{v}'. Allowed: {sorted(list(allowed))}")
    use_interp = request.args.get("interp", "0") == "1"

    try:
        if use_interp:
            start, end, rows, corners = interpolated_forecast(lat, lon, vars_req)
            result = {
                "location": {
                    "requested": {"lat": lat, "lon": lon},
                    "mode": "interpolated",
                    "neighbors": [{"lat": la, "lon": lo} for (la, lo) in corners]
                },
                "forecast_window": {"start": start, "end": end},
                "daily": rows
            }
        else:
            s_lat, s_lon = snap_to_grid(lat, lon, GRID_STEP)
            series_preds, meta_by_var = {}, {}
            for var in vars_req:
                model, meta, _ = ensure_model(var, s_lat, s_lon)
                meta_by_var[var] = meta if meta else {}
                series_preds[var] = forecast_30d(var, model, meta_by_var[var])
            dates = series_preds[vars_req[0]]["date"].tolist()
            result = {
                "location": {"requested": {"lat": lat, "lon": lon},
                             "snapped": {"lat": s_lat, "lon": s_lon},
                             "mode": "snapped"},
                "forecast_window": {"start": dates[0], "end": dates[-1]},
                "daily": []
            }
            for i, d in enumerate(dates):
                row = {"date": d}
                if "T2M" in series_preds:
                    r = series_preds["T2M"].iloc[i]
                    row.update({"t2m": float(r["yhat"]), "t2m_lo": float(r["lo"]), "t2m_hi": float(r["hi"]),
                                "t2m_model": meta_by_var["T2M"].get("algo", "prophet")})
                if "PRECIP" in series_preds:
                    r = series_preds["PRECIP"].iloc[i]
                    row.update({"precip": float(max(0.0, r["yhat"])),
                                "precip_lo": float(max(0.0, r["lo"])),
                                "precip_hi": float(max(0.0, r["hi"])),
                                "precip_model": meta_by_var["PRECIP"].get("algo", "prophet")})
                result["daily"].append(row)

    except Exception as e:
        return bad_request(f"Training/forecast error: {str(e)}")

    # Optional one-day filter
    qdate = request.args.get("date")
    if qdate:
        result["daily"] = [x for x in result["daily"] if x["date"] == qdate]
        if not result["daily"]:
            return bad_request("Requested date not within the next 30 days")

    return ok(result)

@app.route("/history", methods=["GET"])
def history_route():
    # Historical actuals for one date (from cached POWER), with optional interpolation
    try:
        lat = float(request.args.get("lat",""))
        lon = float(request.args.get("lon",""))
    except ValueError:
        return bad_request("lat and lon must be numbers")
    qdate = request.args.get("date", "")
    if not qdate:
        return bad_request("date is required (YYYY-MM-DD)")
    vars_req = request.args.get("vars", "T2M,PRECIP")
    vars_req = [v.strip().upper() for v in vars_req.split(",") if v.strip()]
    allowed = {"T2M", "PRECIP"}
    for v in vars_req:
        if v not in allowed:
            return bad_request(f"Unsupported var '{v}'. Allowed: {sorted(list(allowed))}")
    use_interp = request.args.get("interp", "0") == "1"

    try:
        if use_interp:
            row, corners = interpolated_history(lat, lon, qdate, vars_req)
            result = {
                "location": {
                    "requested": {"lat": lat, "lon": lon},
                    "mode": "interpolated",
                    "neighbors": [{"lat": la, "lon": lo} for (la, lo) in corners]
                },
                "daily": [row]
            }
        else:
            s_lat, s_lon = snap_to_grid(lat, lon, GRID_STEP)
            df = get_history(s_lat, s_lon)
            rec = df[df["date"] == pd.to_datetime(qdate)]
            if rec.empty:
                return bad_request("No data for that date (check range/format)")
            rec = rec.iloc[0]
            row = {"date": qdate}
            if "T2M" in vars_req:    row["t2m"] = float(rec["T2M"])
            if "PRECIP" in vars_req: row["precip"] = float(max(0.0, rec["PRECIP"]))
            result = {
                "location": {"requested": {"lat": lat, "lon": lon}, "snapped": {"lat": s_lat, "lon": s_lon}, "mode": "snapped"},
                "daily": [row]
            }
    except Exception as e:
        return bad_request(f"History error: {str(e)}")

    return ok(result)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    if os.getenv("PREWARM", "0") == "1":
        vars_env = os.getenv("PREWARM_VARS", "T2M,PRECIP")
        vars_list = tuple(v.strip().upper() for v in vars_env.split(",") if v.strip())
        print("[startup] Pre-warming Sri Lanka grid (with saved models)…")
        prewarm_models(vars_list=vars_list)
        print("[startup] Pre-warm complete.")
    app.run(host="0.0.0.0", port=8000, debug=True)
