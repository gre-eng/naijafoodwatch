"""
wfp_loader.py
=============
Loads REAL monthly retail food prices from the WFP Global Food Prices database
(via the Humanitarian Data Exchange, HDX) for any country, and converts them to
the same {commodity: {"YYYY-MM": price}} shape your forecasting engine already uses.

Check data completeness first (prints a table):
    python -m data.wfp_loader nigeria
"""
from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HDX = "https://data.humdata.org/api/3/action"
CACHE = Path(__file__).resolve().parent / "cache"
MARKETS: dict = {}  # (commodity, "YYYY-MM") -> number of markets behind that price


def _find_csv_url(country: str) -> str:
    slug = f"wfp-food-prices-for-{country.lower().strip().replace(' ', '-')}"
    pkg = None
    r = requests.get(f"{HDX}/package_show", params={"id": slug}, timeout=30)
    if r.ok and r.json().get("success"):
        pkg = r.json()["result"]
    else:  # fall back to search
        s = requests.get(f"{HDX}/package_search", params={"q": f"wfp food prices {country}", "rows": 10}, timeout=30).json()
        for p in s.get("result", {}).get("results", []):
            if p["name"].startswith("wfp-food-prices-for"):
                pkg = p
                break
    if not pkg:
        raise RuntimeError(f"No WFP food price dataset found on HDX for {country!r}")
    cands = []
    for res in pkg["resources"]:
        name = (res.get("name") or "").lower()
        url = res.get("download_url") or res.get("url") or ""
        fname = url.lower().split("?")[0].rsplit("/", 1)[-1]
        is_csv = (res.get("format") or "").lower() == "csv" or fname.endswith(".csv") or name.endswith(".csv")
        if not is_csv or re.search(r"(^|[_\-. ])qc([_\-. ]|$)", name + " " + fname):
            continue
        cands.append((not ("price" in name or "price" in fname), url))
    if cands:
        return sorted(cands)[0][1]
    found = [(r.get("name"), r.get("format")) for r in pkg["resources"]]
    raise RuntimeError(f"No usable price CSV in dataset {pkg.get('name')!r}. Resources found: {found}")


def fetch_country(country: str = "nigeria", max_age_days: float = 7) -> pd.DataFrame:
    """Download (or reuse a fresh local copy of) the country's WFP price file."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"wfp_{country.lower().replace(' ', '_')}.csv"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return _read(path.read_text(encoding="utf-8"))
    try:
        text = requests.get(_find_csv_url(country), timeout=120).text
        path.write_text(text, encoding="utf-8")
        return _read(text)
    except Exception:
        if path.exists():  # offline or HDX down: use the last saved copy
            return _read(path.read_text(encoding="utf-8"))
        raise


def _read(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df = df[~df["date"].str.startswith("#", na=True)].copy()  # drop the HXL tag row
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m")
    return df.dropna(subset=["price", "month"])


def build_series(df: pd.DataFrame, unit: str = "KG", price_type: str = "Retail", min_months: int = 24):
    """Returns (series, report). series[name]["YYYY-MM"] = median price across markets."""
    d = df[(df["unit"].str.upper() == unit.upper()) & (df["pricetype"].str.lower() == price_type.lower())]
    series, rows = {}, []
    for name, g in d.groupby("commodity"):
        monthly = g.groupby("month")["price"].median()
        if len(monthly) < min_months:
            continue
        series[name] = {m: round(float(v), 2) for m, v in monthly.items()}
        for m, c in g.groupby("month")["market"].nunique().items():
            MARKETS[(name, m)] = int(c)
        first, last = monthly.index.min(), monthly.index.max()
        span = (pd.Period(last) - pd.Period(first)).n + 1
        behind = (pd.Period(time.strftime("%Y-%m")) - pd.Period(last)).n
        rows.append(dict(commodity=name, first=first, last=last, real_months=len(monthly), span_months=span,
                         coverage_pct=round(100 * len(monthly) / span), markets=g["market"].nunique(), months_behind=behind))
    return series, pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)


def load_series(country: str = "nigeria", unit: str = "KG"):
    return build_series(fetch_country(country), unit=unit)


if __name__ == "__main__":
    country = sys.argv[1] if len(sys.argv) > 1 else "nigeria"
    series, report = load_series(country)
    print(f"\nWFP real-price coverage for {country} (retail, per KG):\n")
    print(report.to_string(index=False) if len(report) else "No food reached the 24-month minimum.")
    print("\ncoverage_pct = real months / months between first and last reading (your NBS series was about 13%).")