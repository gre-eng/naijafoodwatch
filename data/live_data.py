"""
live_data.py
=============
LIVE, SELF-UPDATING data fetcher for Nigeria food/CPI inflation.

This is the app's PRIMARY data source. Unlike historical_data.py
(a static fallback), every function here calls a real public API at
runtime. Every time the Streamlit app is opened -- or its cache expires --
it re-fetches current data. When the World Bank publishes a new year's
figures, the app picks them up automatically, with no code edits and no
redeploy needed from you.

DATA SOURCES (all free, no API key required)
------------------------------------------------
World Bank Indicators API v2 -- https://api.worldbank.org/v2/
  - Headline CPI inflation, annual %:      indicator FP.CPI.TOTL.ZG
  - Official exchange rate (NGN per USD):  indicator PA.NUS.FCRF
  Country code used: NGA (ISO3 for Nigeria).
  These are ANNUAL series published with a 1-2 year lag, so the latest
  year (e.g. 2026) is normally absent and comes from the static fallback.

FRED (Federal Reserve Bank of St. Louis) public CSV endpoint --
https://fred.stlouisfed.org/graph/fredgraph.csv
  - Brent crude spot price, daily USD/bbl: series DCOILBRENTEU
  Used two ways: (a) collapsed to an ANNUAL (year-to-date) average for
  the OLS model, and (b) the LATEST DAILY observation, shown as the
  current "spot" level. FRED's Brent series is the EIA spot assessment
  and typically trails the market by a few days; it can also differ from
  the futures settlement quoted in the news by a few dollars.

SPOT USD/NGN -- https://open.er-api.com/v6/latest/USD (primary) and the
  fawazahmed0 currency-api on jsDelivr (backup). Both are free, keyless,
  daily-updated MID-MARKET rates. They are NOT the CBN's official NFEM
  rate and can differ from it by a few percent; the app labels them so.

HONEST LIMITS OF WHAT "AUTOMATIC" CAN MEAN HERE (disclosed in the UI,
not hidden)
-------------------------------------------------------------------------
1. World Bank publishes annually, with roughly a 1-2 year lag. FRED
   Brent data is daily but trails by days. Neither will ever give you
   "today's tomato price" at NBS's monthly N/kg granularity.
2. NBS (Nigeria's statistics office) does not expose a free public
   REST API. Their Selected Food Price Watch bulletins and CPI/food
   inflation reports are PDF/web releases, not a machine-readable
   feed. There is no legitimate free way to make NBS_COMMODITY_PRICES
   or FOOD_INFLATION auto-update the way CPI/exchange rate/oil do --
   the two real options are (a) a paid data vendor that has already
   licensed/scraped it, or (b) a web scraper against NBS's own site,
   which is fragile (breaks silently if NBS changes its page layout)
   and may run against NBS's terms of use. Neither is wired up here;
   this module does not guess or scrape.
3. A market shock (war, pandemic, insecurity) is not itself a number
   any API reports. This module can only ever be as current as the
   numeric series it actually fetches.
4. Because (2) and (3) are real, permanent limits -- not a bug to be
   fixed later -- this module also exposes `data_age_note()`, so the
   app can automatically detect and loudly warn when its static food
   data is going stale, rather than silently presenting old prices as
   current ones.

NOTE ON WORDING: freshness_note(), data_age_note() and spot_note()
deliberately return plain text with no icons/emoji -- the app renders
these inside custom banner divs, not st.success/st.info, precisely so no
default Streamlit icon or emoji character leaks into the UI.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime

import requests
import streamlit as st

WB_BASE = "https://api.worldbank.org/v2/country/NGA/indicator"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

IND_CPI_INFLATION = "FP.CPI.TOTL.ZG"   # headline CPI inflation, annual %
IND_EXCHANGE_RATE = "PA.NUS.FCRF"      # official exchange rate, LCU per USD
FRED_BRENT_CRUDE = "DCOILBRENTEU"      # Brent crude spot, daily USD/bbl

# Free, keyless daily USD rates. Tried in order; first plausible answer wins.
SPOT_FX_SOURCES = [
    ("open.er-api.com", "https://open.er-api.com/v6/latest/USD"),
    ("fawazahmed0 currency-api",
     "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"),
]


def _fred_header_ok(text: str) -> bool:
    """FRED's CSV header was 'DATE' for years and is 'observation_date' now.
    The old check looked for upper-case 'DATE' only, which silently rejects
    the new header and makes the Brent fetch fall back to static data."""
    if not text:
        return False
    first = text.splitlines()[0].lower()
    return "date" in first


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_world_bank_series(indicator: str, start_year: int = 1997, end_year: int = 2026) -> dict[int, float]:
    """
    Fetch a single World Bank indicator series for Nigeria.
    Returns {year: value}. Cached for 12 hours so the app doesn't hammer
    the API on every page interaction, but still re-checks for new data
    automatically within the same day, and on every fresh deploy.
    """
    url = f"{WB_BASE}/{indicator}"
    params = {"format": "json", "per_page": 200, "date": f"{start_year}:{end_year}"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return {}

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return {}

    series: dict[int, float] = {}
    for row in payload[1]:
        year = int(row["date"])
        value = row["value"]
        if value is not None:
            series[year] = round(float(value), 4)
    return dict(sorted(series.items()))


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_fred_annual_average(series_id: str, start_year: int = 1997, end_year: int = 2026) -> dict[int, float]:
    """
    Fetch a daily FRED series as CSV (no API key needed) and collapse it
    to an annual average per calendar year, matching the annual grain
    the rest of this app uses. For the current, still-in-progress year,
    this is the average of whatever days have been published so far --
    which is exactly what a live, honest "year to date" figure should
    be, not a fabricated full-year number.
    """
    params = {
        "id": series_id,
        "cosd": f"{start_year}-01-01",
        "coed": f"{end_year}-12-31",
    }
    try:
        resp = requests.get(FRED_CSV_BASE, params=params, timeout=15)
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException:
        return {}

    if not _fred_header_ok(text):
        return {}

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # header row
    for row in reader:
        if len(row) < 2:
            continue
        date_str, value_str = row[0], row[1]
        if value_str in ("", ".", "NA"):
            continue
        try:
            year = int(date_str[:4])
            value = float(value_str)
        except ValueError:
            continue
        sums[year] = sums.get(year, 0.0) + value
        counts[year] = counts.get(year, 0) + 1

    if not sums:
        return {}
    return {year: round(sums[year] / counts[year], 2) for year in sums}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_brent_spot_latest(lookback_days: int = 45) -> dict:
    """
    Latest available DAILY Brent spot price from FRED (EIA series
    DCOILBRENTEU). Returns {"date": "YYYY-MM-DD", "value": float,
    "source": str} or {} if the fetch fails. Cached for 1 hour.
    """
    today = date.today()
    params = {
        "id": FRED_BRENT_CRUDE,
        "cosd": (today - timedelta(days=lookback_days)).isoformat(),
        "coed": today.isoformat(),
    }
    try:
        resp = requests.get(FRED_CSV_BASE, params=params, timeout=15)
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException:
        return {}

    if not _fred_header_ok(text):
        return {}

    latest = None
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    for row in reader:
        if len(row) < 2 or row[1] in ("", ".", "NA"):
            continue
        try:
            latest = (row[0][:10], float(row[1]))
        except ValueError:
            continue
    if latest is None:
        return {}
    return {"date": latest[0], "value": round(latest[1], 2),
            "source": "EIA Brent spot via FRED (DCOILBRENTEU)"}


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def fetch_usd_ngn_spot() -> dict:
    """
    Latest USD/NGN mid-market rate from a free keyless daily source.
    Returns {"rate": float, "date": "YYYY-MM-DD", "source": str} or {}.
    NOT the CBN official NFEM rate. Cached for 3 hours.
    """
    for name, url in SPOT_FX_SOURCES:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if "rates" in payload:                       # open.er-api.com
                rate = float(payload["rates"]["NGN"])
                stamp = payload.get("time_last_update_utc")
                day = parsedate_to_datetime(stamp).date().isoformat() if stamp else date.today().isoformat()
            else:                                        # fawazahmed0 currency-api
                rate = float(payload["usd"]["ngn"])
                day = str(payload.get("date", date.today().isoformat()))[:10]
            if rate > 0:
                return {"rate": round(rate, 2), "date": day, "source": name}
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue
    return {}


def get_spot_snapshot(brent_fallback: tuple[str, float], ngn_fallback: tuple[str, float]) -> dict:
    """
    Today's spot levels, with an honest live/static flag.

    brent_fallback / ngn_fallback are (date, value) tuples from
    historical_data.py, used only if the live fetch fails or returns a
    value outside a plausibility band (so a bad API response can never
    put an absurd number on screen).

    Returns {"brent": {...}, "ngn": {...}} where each has
    value, date, is_live, source.
    """
    b = fetch_brent_spot_latest()
    if b and 20 <= b["value"] <= 300:
        brent = dict(value=b["value"], date=b["date"], is_live=True, source=b["source"])
    else:
        brent = dict(value=brent_fallback[1], date=brent_fallback[0], is_live=False,
                     source="static fallback (last manually verified)")

    n = fetch_usd_ngn_spot()
    lo, hi = 0.6 * ngn_fallback[1], 1.7 * ngn_fallback[1]
    if n and lo <= n["rate"] <= hi:
        ngn = dict(value=n["rate"], date=n["date"], is_live=True,
                   source=f"{n['source']} (mid-market, not the CBN official rate)")
    else:
        ngn = dict(value=ngn_fallback[1], date=ngn_fallback[0], is_live=False,
                   source="static fallback: CBN NFEM (last manually verified)")
    return {"brent": brent, "ngn": ngn}


def spot_note(item: dict, label: str, prefix: str = "", today: date | None = None) -> str:
    """Plain-text one-liner for a spot item, e.g.
    'Brent crude now: $113.43 per barrel (2026-10-30, live, EIA Brent spot via FRED)'."""
    if today is None:
        today = date.today()
    try:
        age = (today - datetime.strptime(item["date"], "%Y-%m-%d").date()).days
    except ValueError:
        age = None
    age_txt = f", {age} days old" if age is not None and age > 3 else ""
    state = "live" if item["is_live"] else "static"
    unit = "per barrel" if prefix == "$" else "per dollar"
    return f"{label}: {prefix}{item['value']:,.2f} {unit} ({item['date']}, {state}{age_txt}; {item['source']})"


def get_cpi_inflation_live(fallback: dict[int, float]) -> tuple[dict[int, float], bool]:
    """
    Returns (series, is_live). Merges live data over the fallback so any
    year the API is missing (e.g. very latest year not yet published)
    still has a value, while every year the API *does* have overrides
    the static fallback with the freshest number.
    """
    live = fetch_world_bank_series(IND_CPI_INFLATION)
    if not live:
        return fallback, False
    merged = {**fallback, **live}
    return dict(sorted(merged.items())), True


def get_exchange_rate_live(fallback: dict[int, float]) -> tuple[dict[int, float], bool]:
    live = fetch_world_bank_series(IND_EXCHANGE_RATE)
    if not live:
        return fallback, False
    merged = {**fallback, **live}
    return dict(sorted(merged.items())), True


def get_oil_price_live(fallback: dict[int, float]) -> tuple[dict[int, float], bool]:
    """
    Live Brent crude, annual (year-to-date for the current year), from
    FRED's free public CSV endpoint. This is what lets the app react to
    an oil-price shock -- war, OPEC decision, shipping disruption --
    automatically, without anyone touching the code. NOTE: it is an
    AVERAGE of the year so far, so it lags today's spot price after a
    sharp move; see get_spot_snapshot() for the latest daily level.
    """
    live = fetch_fred_annual_average(FRED_BRENT_CRUDE)
    if not live:
        return fallback, False
    merged = {**fallback, **live}
    return dict(sorted(merged.items())), True


def freshness_note(is_live: bool, series: dict[int, float]) -> str:
    """
    Plain-text status line -- no icons or emoji. The caller (app.py)
    decides how to color/frame this inside a banner div.
    """
    if not series:
        return "No data available: the live fetch failed and no fallback series is present."
    latest_year = max(series.keys())
    if is_live:
        return f"live-updated just now, latest year available: {latest_year}"
    return (
        f"live fetch unavailable right now, showing the last verified "
        f"static figures (through {latest_year}); the app will retry "
        f"automatically next time it loads"
    )


def data_age_note(latest_period: str, today: date | None = None, warn_after_months: int = 2) -> tuple[str, bool]:
    """
    For series that CANNOT be live-fetched (NBS commodity prices, NBS
    food inflation) -- automatically compute how stale the newest data
    point is versus today's real date, and return (message, is_stale).

    This is the honest substitute for "automatic updating" on a series
    with no public API: the app can't fetch new NBS numbers by itself,
    but it can always tell you, truthfully and without any manual
    input, exactly how out of date it currently is.

    `latest_period` accepts "YYYY-MM" (monthly) or "YYYY" (annual).
    """
    if today is None:
        today = date.today()

    parts = latest_period.split("-")
    latest_year = int(parts[0])
    latest_month = int(parts[1]) if len(parts) > 1 else 12

    months_stale = (today.year - latest_year) * 12 + (today.month - latest_month)
    is_stale = months_stale > warn_after_months

    if months_stale <= 0:
        return f"Most recent data point: {latest_period} (current).", False
    if not is_stale:
        return f"Most recent data point: {latest_period} ({months_stale} month(s) behind today).", False
    return (
        f"Most recent data point: {latest_period} -- that is {months_stale} months "
        f"behind today's date. This series has no public API (see module "
        f"docstring), so it cannot refresh itself; it needs a manual update "
        f"from the latest NBS bulletin.",
        True,
    )