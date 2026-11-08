"""
server.py - NaijaFoodWatch public API + web front end (FastAPI).
Run from the project root:   uvicorn server:app --reload     then open http://localhost:8000

Data policy: every price is traced to a source and month; interpolated and estimated values are
labelled; a food gets guidance only if its series passes validation (grade A, B or C).
"""
import json
import sys
import time
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from data.historical_data import NBS_COMMODITY_PRICES, FOOD_INFLATION, RICE_PRICE_ESTIMATE_NOTICE
from data.provenance import validate, FRESH_MAX_MONTHS
from data.wfp_loader import load_series, MARKETS
from src.forecasting_engine import commodity_monthly_forecast
from src.harvest_signal import analyse

COUNTRY = "nigeria"
TTL_SECONDS = 6 * 3600
SOURCE_TEXT = {
    "WFP": "WFP Global Food Prices via the Humanitarian Data Exchange (HDX): retail price per kg, median of reporting markets.",
    "NBS": "Nigeria National Bureau of Statistics, Selected Food Price Watch (monthly bulletins).",
}

PRICES = dict(NBS_COMMODITY_PRICES)
SOURCE = {n: "NBS" for n in PRICES}
try:  # WFP replaces a food only when its latest real month is newer
    for name, ser in load_series(COUNTRY)[0].items():
        if name not in PRICES or max(ser) > max(PRICES[name]):
            PRICES[name], SOURCE[name] = ser, "WFP"
except Exception as exc:
    print("WFP data unavailable, using NBS only:", exc)

QUALITY = {n: validate(s, SOURCE[n], RICE_PRICE_ESTIMATE_NOTICE if SOURCE[n] == "NBS" and "rice" in n.lower() else None)
           for n, s in PRICES.items()}

app = FastAPI(title="NaijaFoodWatch API")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
Role = Literal["grow", "buy"]
Storage = Literal["good", "fair", "poor"]
Weather = Literal["dry", "mixed", "wet"]
_cache: dict = {}


def _signal(crop: str, role: str, storage: str, weather: str) -> dict:
    if crop not in PRICES:
        raise HTTPException(404, "Unknown food")
    ser, q = PRICES[crop], QUALITY[crop]
    base = dict(crop=crop, source=SOURCE[crop], source_text=SOURCE_TEXT[SOURCE[crop]], last_month=q["last"],
                behind=q["behind"], quality=q)
    if q["grade"] == "D":
        why = (f"The latest real price for this food is from {q['last']}, about {q['behind']} months ago, so no guidance is shown."
               if q["behind"] > FRESH_MAX_MONTHS else "This food has too few real price readings or failed data checks, so no guidance is shown.")
        return dict(base, usable=False, action="NO DATA", why=why)
    key = (crop, role, storage, weather)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL_SECONDS:
        return hit[1]
    a = analyse(crop, ser, commodity_monthly_forecast, role, storage, weather)
    fc = a.pop("fc")
    obs = [dict(month=m, price=round(v, 2), type="observed" if m in ser else "interpolated",
                markets=MARKETS.get((crop, m)) if SOURCE[crop] == "WFP" else None)
           for m, v in zip(fc.history_index[-12:], fc.history_values[-12:])]
    a.update(base, usable=True, inflation=FOOD_INFLATION[max(FOOD_INFLATION)], observations=obs,
             history=dict(x=fc.history_index[-12:], y=fc.history_values[-12:]),
             forecast=dict(x=fc.forecast_index, y=fc.forecast_mean, lo=fc.forecast_lower, hi=fc.forecast_upper))
    a = json.loads(json.dumps(a, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
    _cache[key] = (time.time(), a)
    return a


@app.get("/api/foods")
def foods():
    return sorted(PRICES, key=lambda n: (QUALITY[n]["grade"] == "D", n))


@app.get("/api/signal")
def signal(crop: str, role: Role = "grow", storage: Storage = "fair", weather: Weather = "mixed"):
    return _signal(crop, role, storage, weather)


@app.get("/api/watch")
def watch(role: Role = "grow", storage: Storage = "fair", weather: Weather = "mixed"):
    out = []
    for crop in foods():
        s = _signal(crop, role, storage, weather)
        out.append(dict(crop=crop, action=s["action"], last=s.get("last"), outlook=s.get("outlook"),
                        source=s["source"], last_month=s["last_month"], grade=s["quality"]["grade"]))
    return out


@app.get("/api/quality")
def quality():
    """Public data-quality report for every food."""
    return {n: dict(QUALITY[n], source=SOURCE[n]) for n in PRICES}


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")