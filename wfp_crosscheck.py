"""
wfp_crosscheck.py - can WFP's CURRENT Nigerian prices help estimate NBS prices for months NBS has not published?
Everything printed is computed from the live WFP data (downloaded through the HDX API by data/wfp_loader.py). No numbers are typed in.
Run:   python wfp_crosscheck.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
PAIRS = {"Meat (beef)": ["Beef, boneless"], "Fish": ["Mackerel, frozen", "Titus fish, frozen", "Catfish, fresh", "Tilapia, fresh"]}


def run(wfp_df=None, nbs=None):
    from data.official_prices import get_official_prices
    from data.wfp_loader import fetch_country, _read  # noqa: F401
    nbs = nbs or {k.split(" (")[0]: pd.Series(v) for k, v in get_official_prices().items()}
    df = wfp_df if wfp_df is not None else fetch_country("nigeria")
    r = df[(df.unit.str.upper() == "KG") & (df.pricetype.str.lower() == "retail")]
    out = []
    for w, foods in PAIRS.items():
        g = r[r.commodity == w]
        if g.empty:
            print(f"WFP has no '{w}' rows."); continue
        wp = g.groupby("month").price.median().sort_index()
        mk = g.groupby("month").market.nunique()
        print(f"\nWFP '{w}': {wp.index.min()} to {wp.index.max()}, markets per month in the last 6 months: {mk.tail(6).tolist()}")
        for f in foods:
            if f not in nbs:
                continue
            n = nbs[f].sort_index()
            both = pd.concat([np.log(n).rename("n"), np.log(wp).rename("w")], axis=1, join="inner")
            d3 = both.diff(3).dropna()
            if len(d3) < 8:
                print(f"  {f}: only {len(d3)} overlapping 3-month windows, too few"); continue
            b, a = np.polyfit(d3.w, d3.n, 1)
            r2 = np.corrcoef(d3.w, d3.n)[0, 1] ** 2
            err_m = np.mean([abs(d3.n.iloc[i] - (np.polyfit(np.delete(d3.w.values, i), np.delete(d3.n.values, i), 1) @ [d3.w.iloc[i], 1])) for i in range(len(d3))])
            err_0 = np.mean(np.abs(d3.n))
            ok = r2 >= 0.3 and err_m < err_0
            print(f"  vs NBS {f:20s} windows={len(d3):2d}  slope={b:5.2f}  R2={r2:4.2f}  error with WFP {err_m * 100:4.1f}%  vs no-change {err_0 * 100:4.1f}%  -> {'USABLE' if ok else 'not usable'}")
            if ok:
                last = both.index[-1]
                for m in wp.index[wp.index > last]:
                    out.append(dict(food=f, month=m, estimate=round(float(np.exp(both.n.iloc[-1] + b * (np.log(wp[m]) - both.w.iloc[-1]))), 2),
                                    last_official=round(float(n.iloc[-1]), 2), basis=f"WFP {w} movement, slope {b:.2f}", label="MODEL_ESTIMATE, not NBS data"))
    if out:
        pd.DataFrame(out).to_csv("data/wfp_based_estimates.csv", index=False)
        print("\nWrote data/wfp_based_estimates.csv")
    else:
        print("\nNo WFP-based estimate written: WFP did not pass the test for any food.")


if __name__ == "__main__":
    run()