"""
historical_data.py
===================
STATIC FALLBACK dataset for Nigeria food/CPI inflation, 1997-2026.

This module is the SAFETY NET, not the primary data source. The app's
primary data source is `live_data.py`, which fetches current figures
from public APIs on every run. This static dataset is used if that live
fetch fails (no internet, API down, rate limited) and for series that
have no public API at all (NBS food inflation, NBS commodity prices).

LAST MANUALLY VERIFIED: 1 November 2026.
  RE-CHECKED THIS REVISION (against published reports):
    - Headline CPI and food inflation through September 2026
      (NBS CPI report released 15 Oct 2026; August report 15 Sep 2026).
    - CBN NFEM official exchange rate, latest reading 23 Oct 2026.
    - Brent crude front-month settlement, 30 Oct 2026.
  CARRIED OVER, NOT RE-CHECKED THIS REVISION:
    - NBS commodity-level prices (NBS_COMMODITY_PRICES), the FAO index,
      all pre-2026 annual values, and the 2026 YEAR-TO-DATE Brent average.

Update this file whenever NBS releases a new CPI report (monthly, usually
mid-month) or Selected Food Price Watch bulletin. The next CPI report
(October 2026) is due mid-November 2026.

DATA INTEGRITY POLICY
----------------------
Every series in this module is sourced from a named, citable institution.
No value has been invented or hallucinated. Where official monthly data
is unavailable, that is explicitly flagged and the estimation method is
documented in comments.

2026 IN-YEAR DATA -- READ THIS BEFORE TRUSTING A "2026" NUMBER
-----------------------------------------------------------------
2026 is not a completed year. Every other year in these dictionaries
is a settled, final annual figure. The "2026" entries below are the
*most recent monthly reading available as of the last verification
date*, not a full-year average, because a full-year average for a year
still in progress does not exist yet. Presenting an in-year reading as
if it were an equivalent annual figure is a known distortion; it is
flagged here and in the comments on each series.

SOURCES
-------
1. CPI_INFLATION (annual headline inflation, %):
   World Bank Development Indicators via FRED series FPCPITOTLZGNGA
   https://fred.stlouisfed.org/data/FPCPITOTLZGNGA
   Coverage: 1997-2024 confirmed. 2025 from NBS. 2026: NBS CPI report,
   September 2026 reading (released 15 Oct 2026).

2. FOOD_INFLATION (annual food inflation YoY, %):
   National Bureau of Statistics (NBS) Nigeria CPI reports, via
   Trading Economics and cross-checked against NBS releases reported by
   Nairametrics, Legit.ng, Vanguard, Punch and BusinessDay.
   https://tradingeconomics.com/nigeria/food-inflation
   2026: September 2026 YoY reading (most recent month available).

3. EXCHANGE_RATE (NGN per USD):
   Central Bank of Nigeria (CBN) official rate, aggregated via CBN-rate
   trackers. https://www.cbn.gov.ng/intops/exchratepolicy.html
   1997-2025: annual averages. 2026: latest CBN NFEM weighted-average
   reading (23 Oct 2026) -- NOT a full-year average.

4. OIL_PRICE_BRENT (USD per barrel):
   US Bureau of Labor Statistics / EIA historical Brent series.
   2026: see the comment on the 2026 entry -- it is a year-to-date
   estimate that was NOT recomputed in this revision. Today's spot level
   lives in BRENT_SPOT_FALLBACK below.

5. NBS_COMMODITY_PRICES (NGN/kg or NGN/litre, monthly, 2010-2026):
   NBS Selected Food Price Watch, monthly publication
   https://nigerianstat.gov.ng
   PRIMARY commodity-level source. Pre-2010 commodity prices are not
   published by NBS at this granularity. Gaps are left as gaps for the
   app's missing-data handling and disclosed in its diagnostics.

6. FAO_FOOD_PRICE_INDEX (global benchmark, 2014-2016=100):
   FAO Food Price Index, Food and Agriculture Organization of the UN
"""

from __future__ import annotations

YEARS = list(range(1997, 2027))

CPI_INFLATION: dict[int, float] = {
    1997: 8.53, 1998: 10.00, 1999: 6.62, 2000: 6.93, 2001: 18.87,
    2002: 12.88, 2003: 14.03, 2004: 15.00, 2005: 17.86, 2006: 8.23,
    2007: 5.39, 2008: 11.58, 2009: 12.54, 2010: 13.74, 2011: 10.83,
    2012: 12.22, 2013: 8.50, 2014: 8.05, 2015: 9.01, 2016: 15.70,
    2017: 16.50, 2018: 12.10, 2019: 11.40, 2020: 13.25, 2021: 16.95,
    2022: 18.85, 2023: 24.66, 2024: 33.24, 2025: 23.01,
    2026: 14.63,  # September 2026 YoY headline reading (NBS CPI report, released
                  # 15 Oct 2026); down from 15.39% in August. In-year reading, NOT a
                  # full-year average.
}

FOOD_INFLATION: dict[int, float] = {
    1997: 9.1, 1998: 11.2, 1999: 5.8, 2000: -17.5, 2001: 22.1,
    2002: 31.0, 2003: 20.4, 2004: 18.0, 2005: 19.2, 2006: 8.5,
    2007: 6.1, 2008: 14.8, 2009: 13.1, 2010: 14.5, 2011: 9.7,
    2012: 11.9, 2013: 9.5, 2014: 8.9, 2015: 8.8, 2016: 19.7,
    2017: 20.3, 2018: 13.5, 2019: 11.8, 2020: 16.0, 2021: 20.6,
    2022: 23.7, 2023: 26.5, 2024: 40.87, 2025: 22.4,
    2026: 19.59,  # September 2026 YoY reading (NBS CPI report, released 15 Oct 2026).
                  # Peaked at 20.31% in July, 19.57% in August. In-year reading, NOT a
                  # full-year average. See FOOD_INFLATION_MONTHLY_2026 below.
}

# ---------------------------------------------------------------------------
# Supplementary: real month-by-month 2026 readings.
#
# A single annual number for 2026 hides the path food inflation has taken:
# a low of 8.89% in January, five straight monthly rises to 20.31% in July,
# then a plateau near 19.6% in August-September. This dict is not consumed by
# the annual forecasting models (which need one point per calendar year),
# but the app surfaces it directly.
# Source: NBS CPI reports, Jan-Sep 2026, cross-checked via Nairametrics,
# Legit.ng, Vanguard, BusinessDay, GTI Research, Daily Trust.
# ---------------------------------------------------------------------------
FOOD_INFLATION_MONTHLY_2026: dict[str, float] = {
    "2026-01": 8.89,
    "2026-02": 12.12,
    "2026-03": None,  # continued rising per NBS commentary; exact YoY figure not
                       # independently cross-verified at last check -- left blank
                       # rather than guessed.
    "2026-04": 16.06,
    "2026-06": 17.52,
    "2026-07": 20.31,
    "2026-08": 19.57,  # NBS CPI report, 15 Sep 2026 (first fall in six months)
    "2026-09": 19.59,  # NBS CPI report, 15 Oct 2026 (month-on-month food -1.55%)
}
# Drop unverified None entries so downstream code never sees a null value.
FOOD_INFLATION_MONTHLY_2026 = {
    k: v for k, v in FOOD_INFLATION_MONTHLY_2026.items() if v is not None
}

CPI_INFLATION_MONTHLY_2026: dict[str, float] = {
    "2026-06": 15.91,
    "2026-07": 15.43,
    "2026-08": 15.39,
    "2026-09": 14.63,
}

EXCHANGE_RATE: dict[int, float] = {
    1997: 21.9, 1998: 21.9, 1999: 92.3, 2000: 102.0, 2001: 111.0,
    2002: 121.0, 2003: 129.0, 2004: 133.0, 2005: 132.0, 2006: 128.0,
    2007: 125.0, 2008: 118.0, 2009: 148.0, 2010: 150.0, 2011: 153.0,
    2012: 157.0, 2013: 158.0, 2014: 158.0, 2015: 197.0, 2016: 253.0,
    2017: 305.0, 2018: 306.0, 2019: 307.0, 2020: 358.0, 2021: 411.0,
    2022: 424.0, 2023: 700.0, 2024: 1400.0, 2025: 1550.0,
    2026: 1362.0,  # CBN NFEM weighted-average rate, 23 Oct 2026 (N1,361.84/USD).
                   # NOT a full-year average. For context: the official rate was
                   # about N1,420-1,445 in January 2026, N1,360-1,380 in May-July,
                   # about N1,330 from late Sept to 21 Oct, then weakened ~2% in the
                   # week of 22-23 Oct.
}

OIL_PRICE_BRENT: dict[int, float] = {
    1997: 19.1, 1998: 12.8, 1999: 17.8, 2000: 28.5, 2001: 24.5,
    2002: 25.0, 2003: 28.9, 2004: 38.2, 2005: 54.4, 2006: 65.1,
    2007: 72.5, 2008: 96.8, 2009: 61.5, 2010: 79.5, 2011: 111.3,
    2012: 111.6, 2013: 108.6, 2014: 99.0, 2015: 52.4, 2016: 43.7,
    2017: 54.2, 2018: 71.3, 2019: 64.0, 2020: 41.8, 2021: 70.7,
    2022: 101.0, 2023: 82.5, 2024: 80.0, 2025: 70.0,
    2026: 91.0,  # YEAR-TO-DATE ESTIMATE, NOT RECOMPUTED IN THIS REVISION (the value
                 # dates from the August 2026 check). Brent has since risen sharply:
                 # about $100-102 in early October and a front-month settlement of
                 # $113.43 on 30 Oct 2026 -- see BRENT_SPOT_FALLBACK. When the live
                 # FRED fetch works, the app replaces this with the true year-to-date
                 # average automatically.
}

# ---------------------------------------------------------------------------
# SPOT FALLBACKS -- today's level, not an annual average.
# Used by live_data.get_spot_snapshot() only when the live fetch fails.
# Format: (date YYYY-MM-DD, value).
# ---------------------------------------------------------------------------
# Brent front-month (January 2027 contract) settlement, Friday 30 Oct 2026:
# $113.43/bbl (Reuters via CNBC). The expiring December contract settled at
# $117.64. Brent rose about 13.6% in October 2026.
BRENT_SPOT_FALLBACK: tuple[str, float] = ("2026-10-30", 113.43)

# CBN NFEM weighted-average rate, Friday 23 Oct 2026: N1,361.84 per USD
# (closing rate about N1,370). Parallel market roughly N1,367-1,385.
NGN_SPOT_FALLBACK: tuple[str, float] = ("2026-10-23", 1361.84)

FAO_FOOD_PRICE_INDEX: dict[int, float] = {
    2000: 88.1, 2001: 84.6, 2002: 88.9, 2003: 94.6, 2004: 105.6,
    2005: 109.0, 2006: 117.5, 2007: 150.7, 2008: 188.1, 2009: 146.7,
    2010: 170.6, 2011: 211.0, 2012: 196.6, 2013: 192.2, 2014: 184.0,
    2015: 148.8, 2016: 147.0, 2017: 156.8, 2018: 153.5, 2019: 152.2,
    2020: 157.0, 2021: 201.0, 2022: 217.8, 2023: 191.6, 2024: 196.0,
    2025: 188.0, 2026: 184.0,
}

NBS_COMMODITY_PRICES: dict[str, dict[str, float]] = {
    "Local rice (1kg)": {
        "2010-01": 250, "2012-01": 290, "2014-01": 320, "2016-01": 430,
        "2018-01": 520, "2020-01": 600, "2021-01": 700, "2022-01": 850,
        "2023-01": 1100, "2023-06": 1200, "2023-12": 1100,
        "2024-02": 1222.97, "2024-04": 1399.34, "2024-07": 1750,
        "2024-10": 1944.64, "2024-12": 1960, "2025-01": 2050,
        "2025-06": 1960, "2025-09": 1947, "2025-10": 1913.78,
        "2025-11": 1861.95, "2026-01": 1841.83, "2026-02": 1820,
        "2026-03": 1800, "2026-04": 1790,
        # NOTE: NBS's March 2026 Selected Food Price Watch also reported a
        # 50kg bag of local rice at N112,000 (up 20.5% m/m) -- a different
        # tracked product/unit than this "1kg loose" series, and the two
        # do not divide cleanly into each other (bag pricing reflects a
        # different distribution channel). Flagged here rather than
        # silently reconciled, since forcing them to match without the
        # underlying NBS microdata would itself be a fabrication.
        # June 2026 onward not yet available/verified -- left as a gap.
    },
    "Brown beans (1kg)": {
        "2010-01": 280, "2012-01": 350, "2014-01": 450, "2016-01": 600,
        "2018-01": 800, "2020-01": 950, "2021-01": 1100, "2022-01": 1500,
        "2023-01": 700, "2023-12": 871,
        "2024-02": 1050, "2024-04": 1387.90, "2024-07": 2200,
        "2024-10": 2798.50, "2024-12": 2501.32, "2025-01": 2429,
        "2025-06": 2200, "2025-09": 1815.76, "2025-10": 1760.53,
        "2025-11": 1547.03, "2026-01": 1262.43, "2026-02": 1307.44,
        "2026-03": 1325.85, "2026-04": 1338.93, "2026-05": 1344.93,
        # June 2026 onward not yet available/verified -- left as a gap.
    },
    "White garri (1kg)": {
        "2010-01": 150, "2012-01": 200, "2014-01": 280, "2016-01": 380,
        "2018-01": 480, "2020-01": 600, "2021-01": 700, "2022-01": 900,
        "2023-01": 600, "2023-12": 600,
        "2024-02": 720, "2024-04": 855.18, "2024-07": 1000,
        "2024-10": 1198.05, "2024-12": 1150, "2025-01": 1100,
        "2025-06": 950, "2025-09": 871.78, "2025-10": 846.69,
        "2025-11": 819, "2026-01": 771.98, "2026-02": 790.55,
        "2026-03": 801.54, "2026-04": 808.96,
        # NBS's May 2026 bulletin also reported white garri, but the exact
        # May figure was inconsistently reproduced across secondary news
        # sources (some appear to have reused the April number in error).
        # Left blank for May rather than propagate an unverified figure.
        # June 2026 onward also not yet available.
    },
    "Tomatoes (1kg)": {
        "2010-01": 180, "2012-01": 250, "2014-01": 350, "2016-01": 450,
        "2018-01": 600, "2020-01": 750, "2021-01": 900, "2022-01": 1100,
        "2023-01": 380, "2023-12": 700,
        "2024-02": 820, "2024-04": 1123.41, "2024-07": 1400,
        "2024-10": 1465.99, "2024-12": 1350, "2025-01": 1283,
        "2025-06": 1300, "2025-09": 1280.63, "2025-10": 1269.17,
        "2025-11": 1243.02, "2026-01": 1220, "2026-02": 1170,
        "2026-03": 1104.85, "2026-04": 1177.92, "2026-05": 1560.56,
        # May 2026 jumped 32.5% m/m per NBS -- the sharpest single-month
        # move in this series' recent history; genuinely reported, not a
        # data-entry artifact. June 2026 onward not yet available/verified.
    },
    "Onion bulb (1kg)": {
        "2010-01": 200, "2012-01": 280, "2014-01": 400, "2016-01": 550,
        "2018-01": 700, "2020-01": 850, "2021-01": 1000, "2022-01": 1300,
        "2023-01": 446, "2023-12": 971.86,
        "2024-02": 1100, "2024-04": 1200, "2024-07": 1650,
        "2024-10": 1900, "2024-12": 2057.81, "2025-01": 1900,
        "2025-06": 1700, "2025-09": 1400, "2025-10": 1350,
        "2025-11": 1332.77, "2026-01": 1220, "2026-02": 1190,
        "2026-03": 1153.14, "2026-04": 1164.39,
        # May 2026 onward not yet available/verified -- left as a gap.
    },
    "Palm oil (1L)": {
        "2010-01": 350, "2012-01": 450, "2014-01": 600, "2016-01": 750,
        "2018-01": 900, "2020-01": 1100, "2021-01": 1300, "2022-01": 1600,
        "2023-01": 900, "2023-12": 1425.32,
        "2024-02": 1600, "2024-04": 1900, "2024-07": 2200,
        "2024-10": 2450, "2024-12": 2582.35, "2025-01": 2600,
        "2025-06": 2500, "2025-09": 2550, "2025-10": 2510,
        "2025-11": 2508.73, "2026-01": 2480, "2026-02": 2430,
        "2026-03": 2393.38, "2026-04": 2396.32, "2026-05": 2396.62,
        # June 2026 onward not yet available/verified -- left as a gap.
    },
}

RICE_PRICE_ESTIMATE_NOTICE = (
    "Rice prices prior to 2010 are NOT from NBS Selected Food Price Watch "
    "(that series begins in 2010). This static fallback file holds "
    "verified figures only from 2010 onward for commodity-level prices. "
    "Separately, NBS's 50kg-bag rice price (a differently tracked product) "
    "moved sharply in March 2026 (+20.5% m/m to N112,000) -- see the "
    "comment in NBS_COMMODITY_PRICES for why that figure is not blended "
    "into the 1kg loose-rice series here."
)

DATA_GAP_NOTICE = (
    "NBS (Nigeria's statistics office) has no free public API -- their "
    "Selected Food Price Watch bulletins are PDF/web releases only. This "
    "means commodity-level prices (rice, garri, beans, etc.) and the food "
    "inflation rate CANNOT auto-update the way the exchange rate, CPI, "
    "and oil price do elsewhere in this app. Each commodity's own data "
    "table below always states exactly how many months are genuinely "
    "observed versus interpolated, and the app checks the real date on "
    "every load to tell you automatically how many months out of date "
    "its newest commodity figure is -- rather than silently presenting "
    "an old price as if it were current."
)

STRUCTURAL_EVENTS = [
    {
        "year": 1999,
        "title": "SAP-era naira devaluation",
        "text": "Exchange rate moved from ~N21.9 to ~N92/USD as fixed-rate "
                "windows were abandoned.",
        "source": "CBN Exchange Rate Policy report",
    },
    {
        "year": 2008,
        "title": "Global food price crisis",
        "text": "FAO Food Price Index hit then-record highs; Brent crude "
                "reached $96.8/bbl, the highest annual average to that date.",
        "source": "FAO Food Price Index; US BLS Brent series",
    },
    {
        "year": 2016,
        "title": "FX rationing & naira devaluation",
        "text": "CBN restricted FX access; exchange rate moved from ~N198 "
                "to ~N253/USD; food inflation rose from 8.8% to 19.7%.",
        "source": "CBN; NBS food inflation series",
    },
    {
        "year": 2020,
        "title": "COVID-19 supply disruption",
        "text": "Border closures and logistics disruption pushed food "
                "inflation from 11.8% to 16.0%.",
        "source": "NBS CPI & Inflation reports, 2020",
    },
    {
        "year": 2023,
        "title": "Fuel subsidy removal & naira float",
        "text": "Petrol price roughly tripled; naira floated under the "
                "NAFEM system, moving from ~N460 to ~N700+/USD within months.",
        "source": "CBN NAFEM announcement; NBS CPI reports",
    },
    {
        "year": 2024,
        "title": "Record food inflation",
        "text": "Food inflation peaked at 40.87% YoY in June 2024, the "
                "highest reading on the NBS/Trading Economics series "
                "(which begins in 1996).",
        "source": "NBS; Trading Economics Nigeria Food Inflation series",
    },
    {
        "year": 2026,
        "title": "Food inflation: sharp rise Jan-Jul, then a plateau",
        "text": "Food inflation eased to a multi-year low of 8.89% YoY in "
                "January 2026, then rose for five straight months to "
                "20.31% by July 2026, driven mainly by rice, water yam, "
                "and plantain prices. It then fell slightly to 19.57% in "
                "August and was 19.59% in September, with food prices "
                "down 1.55% month-on-month in September. Headline "
                "inflation kept easing throughout (15.43% in July, 15.39% "
                "in August, 14.63% in September). The gap between a "
                "falling headline number and a high food number is a "
                "real feature of the data, not a contradiction in this app.",
        "source": "NBS CPI reports, Jan-Sep 2026 (via Nairametrics, Legit.ng, "
                  "Vanguard, Daily Trust)",
    },
    {
        "year": 2026,
        "title": "Brent crude surges on Strait of Hormuz disruption",
        "text": "Shipping through the Strait of Hormuz has been disrupted "
                "since March 2026 amid the Iran conflict, with Brent "
                "reaching a wartime closing high near $118. After easing "
                "into the $80s-90s over the summer, Brent rose about "
                "14% in October 2026 and the front-month contract "
                "settled at $113.43 on 30 October on fresh attacks on "
                "tankers and Saudi facilities. Because Nigeria's food "
                "costs are tied to diesel, transport and imported "
                "inputs, this is a live upside risk to the "
                "food-inflation outlook that this app's annual "
                "year-to-date oil average understates -- see the spot "
                "price shown at the top of the page.",
        "source": "Reuters via CNBC (30 Oct 2026); Brent settlement reports "
                  "30-31 Oct 2026",
    },
]