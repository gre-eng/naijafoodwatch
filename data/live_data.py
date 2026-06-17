"""
live_data.py
=============
LIVE, SELF-UPDATING data fetcher for Nigeria food/CPI inflation.

This is the app's PRIMARY data source. Unlike historical_data.py
(a static fallback), every function here calls a real public API at
runtime. Every time the Streamlit app is opened -- or its 12-hour
cache expires -- it re-fetches current data. When the World Bank
publishes a new year's figures, the app picks them up automatically,
with no code edits and no redeploy needed from you.

DATA SOURCE (free, no API key required)
-----------------------------------------
World Bank Indicators API v2 -- https://api.worldbank.org/v2/
  - Headline CPI inflation, annual %:      indicator FP.CPI.TOTL.ZG
  - Official exchange rate (NGN per USD):  indicator PA.NUS.FCRF

Country code used: NGA (ISO3 for Nigeria).
Reference docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

HONEST LIMITATION (disclosed in the UI, not hidden)
-------------------------------------------------------
The World Bank API publishes annually, with roughly a 1-2 year lag --
it will not give you "today's tomato price" at NBS's monthly ₦/kg
granularity. NBS does not expose a free public REST API; their
Selected Food Price Watch bulletins are PDF/web releases, not a
machine-readable feed. This module is upfront about that boundary:
it auto-updates the macro series (headline CPI, exchange rate) live,
on every run, with zero manual editing -- and falls back to
historical_data.py's verified static figures if the live call fails
for any reason (no internet, API down, rate-limited).
"""

from __future__ import annotations

import requests
import streamlit as st

WB_BASE = "https://api.worldbank.org/v2/country/NGA/indicator"

IND_CPI_INFLATION = "FP.CPI.TOTL.ZG"   # headline CPI inflation, annual %
IND_EXCHANGE_RATE = "PA.NUS.FCRF"      # official exchange rate, LCU per USD


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


def freshness_note(is_live: bool, series: dict[int, float]) -> str:
    if not series:
        return "⚠️ No data available (live fetch failed and no fallback present)."
    latest_year = max(series.keys())
    if is_live:
        return (
            f"🟢 Live data from the World Bank Indicators API "
            f"(fetched this session, cached 12h). Latest year available: {latest_year}."
        )
    return (
        f"🟡 Live fetch unavailable right now — showing last verified "
        f"static figures (through {latest_year}). The app will retry "
        f"the live source automatically next time it loads."
    )