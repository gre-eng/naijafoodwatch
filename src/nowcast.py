"""
nowcast.py - model ESTIMATES for months NBS has not yet published (e.g. Jun-Sep 2026).
Framework: (1) harmonized official NBS series (data/official_prices.py), (2) a hierarchical model where each food's
drift and momentum are partially pooled across all foods, (3) rolling-origin back-testing, (4) prediction intervals
taken from the model's OWN past errors. Nothing here is NBS data, and every output is labelled MODEL_ESTIMATE.
Run:   python -m src.nowcast
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.official_prices import get_official_prices

H, MIN_TRAIN = 8, 12


def shrink(est, se):
    """Partial pooling (empirical Bayes): noisy estimates move toward the group mean. Returns shrunk values, sds, mean, spread."""
    w = 1 / (se ** 2 + 1e-12)
    mu = np.sum(w * est) / np.sum(w)
    tau2 = max(0.0, np.var(est, ddof=1) - np.mean(se ** 2))
    B = tau2 / (tau2 + se ** 2 + 1e-12)
    return mu + B * (est - mu), np.sqrt(B * se ** 2), mu, np.sqrt(tau2)

# Monthly food-inflation (YoY %) from your historical_data.py (Jan-Apr, Jun, Jul 2026). Used only as a sanity check.
CPI_FOOD_YOY = {"2026-01": 8.89, "2026-02": 12.12, "2026-03": 14.31, "2026-04": 16.06}


def panel(prices: dict) -> pd.DataFrame:
    """Log prices on an unbroken monthly calendar. Uses the latest run of consecutive months (a gap would fake a price jump)."""
    df = pd.DataFrame({k.split(" (")[0]: pd.Series(v) for k, v in prices.items()}).sort_index()
    df.index = pd.PeriodIndex(df.index, freq="M")
    df = df.reindex(pd.period_range(df.index.min(), df.index.max(), freq="M"))
    full = (df.notna().mean(axis=1) >= 0.9).values
    start = len(full)
    while start > 0 and full[start - 1]:
        start -= 1
    df = df.iloc[start:].dropna(axis=1)
    df.index = df.index.astype(str)
    return np.log(df)


def fit(logs: pd.DataFrame, method: str):
    d = logs.diff().dropna()
    mu0, phi0, sd = d.mean(), pd.Series(0.0, index=d.columns), d.std()
    if method == "naive":
        return pd.Series(0.0, index=d.columns), phi0, sd
    T = len(d)
    mu = pd.Series(0.0, index=d.columns) if method == "ar_only" else pd.Series(shrink(mu0.values, (sd / np.sqrt(T)).values)[0], index=d.columns)
    phis, ses = [], []
    for c in d.columns:
        x, y = (d[c] - mu[c]).values[:-1], (d[c] - mu[c]).values[1:]
        b = float(x @ y / (x @ x + 1e-12))
        phis.append(b)
        ses.append(np.sqrt(max(np.var(y - b * x), 1e-12) / (x @ x + 1e-12)))
    phi = pd.Series(np.clip(shrink(np.array(phis), np.array(ses))[0], 0, 0.9), index=d.columns)
    return mu, phi, sd


def predict(logs, mu, phi, h):
    d_last = logs.diff().iloc[-1]
    return pd.DataFrame({k: sum(mu + phi ** j * (d_last - mu) for j in range(1, k + 1)) for k in range(1, h + 1)})


def backtest(logs: pd.DataFrame, method: str):
    rows = []
    for o in range(MIN_TRAIN - 1, len(logs) - 1):
        tr = logs.iloc[:o + 1]
        mu, phi, sd = fit(tr, method)
        pr = predict(tr, mu, phi, H)
        for h in range(1, min(H, len(logs) - 1 - o) + 1):
            act = logs.iloc[o + h] - tr.iloc[-1]
            for f in logs.columns:
                rows.append((method, f, h, act[f] - pr.loc[f, h], (act[f] - pr.loc[f, h]) / sd[f], pr.loc[f, h], act[f], sd[f]))
    return pd.DataFrame(rows, columns=["method", "food", "h", "err", "z", "pred", "act", "sd"])


def build(prices: dict, horizon: int = H) -> dict:
    """What the chatbot needs: central estimate and calibrated 80% range per food and month ahead, the back-tested typical error
    per horizon, and the signal strength above which a price DIRECTION has been right 70%+ of the time in the back-test."""
    logs = panel(prices)
    if len(logs) < MIN_TRAIN + 4:
        raise ValueError(f"Only {len(logs)} consecutive official months are loaded (I need at least {MIN_TRAIN + 4}). Add the missing NBS reports to data/nbs_reports.")
    bts = {m: backtest(logs, m) for m in ("naive", "ar_only")}
    mae = {m: bt.assign(a=bt.err.abs()).groupby("h").a.mean() for m, bt in bts.items()}
    best = min(mae, key=lambda m: mae[m].mean())
    bt = bts[best]
    counts = bt.groupby("h").size()
    horizon = max([h for h in range(1, horizon + 1) if h in counts.index and counts[h] >= 30 and all(counts.get(k, 0) >= 30 for k in range(1, h + 1))] or [1])
    mu, phi, sd = fit(logs, best)
    pr = predict(logs, mu, phi, horizon)
    z = bt.groupby("h").z.quantile([0.10, 0.90]).unstack()
    foods = {}
    for f in logs.columns:
        base = float(logs[f].iloc[-1])
        foods[f] = dict(c=[float(np.exp(base + pr.loc[f, h])) for h in range(1, horizon + 1)],
                        lo=[float(np.exp(base + pr.loc[f, h] + z.loc[h, 0.10] * sd[f])) for h in range(1, horizon + 1)],
                        hi=[float(np.exp(base + pr.loc[f, h] + z.loc[h, 0.90] * sd[f])) for h in range(1, horizon + 1)])
    b3 = bt[(bt.h == 3) & (bt.pred != 0)].assign(strength=lambda d: d.pred.abs() / d.sd, hit=lambda d: np.sign(d.pred) == np.sign(d.act))
    tau = next((float(t) for t in b3.strength.quantile([0.0, 0.25, 0.5, 0.75, 0.9]) if (b3.strength >= t).sum() >= 60 and b3[b3.strength >= t].hit.mean() >= 0.70), None)
    return dict(H=horizon, method=best, foods=foods, sd=sd.to_dict(), tau=tau, mae={int(h): float(v) for h, v in mae[best].items()}, months=len(logs))


def run():
    prices = get_official_prices()
    logs = panel(prices)
    bt = pd.concat([backtest(logs, m) for m in ("naive", "ar_only", "hier_drift_ar")])
    mae = bt.assign(a=bt.err.abs() * 100).pivot_table(index="method", columns="h", values="a", aggfunc="mean").round(1)
    print(f"BACK-TEST: average error in % of price, by months ahead ({len(logs)} months, {logs.shape[1]} foods, rolling origins):\n{mae.to_string()}\n")
    best = mae.mean(axis=1).idxmin()
    print(f"Best method on past data: {best}\n")
    mu, phi, sd = fit(logs, best)
    pr = predict(logs, mu, phi, H)
    z = bt[bt.method == best].groupby("h").z.quantile([0.10, 0.90]).unstack()
    cov = {h: float(((bt[(bt.method == best) & (bt.h == h)].z >= z.loc[h, 0.10]) & (bt[(bt.method == best) & (bt.h == h)].z <= z.loc[h, 0.90])).mean()) for h in z.index}
    last = pd.Period(logs.index[-1])
    rows = []
    for f in logs.columns:
        for h in range(1, H + 1):
            c = logs[f].iloc[-1] + pr.loc[f, h]
            rows.append(dict(food=f, month=str(last + h), estimate=round(float(np.exp(c)), 2), low80=round(float(np.exp(c + z.loc[h, 0.10] * sd[f])), 2),
                             high80=round(float(np.exp(c + z.loc[h, 0.90] * sd[f])), 2), last_official=round(float(np.exp(logs[f].iloc[-1])), 2),
                             method=best, label="MODEL_ESTIMATE, not NBS data"))
    out = pd.DataFrame(rows)
    out.to_csv("data/nowcast_jun_oct_2026.csv", index=False)
    # sanity check against the national food-inflation signal
    yoy = {m: float(np.median(np.exp(logs.loc[m] - logs.loc[str(pd.Period(m) - 12)]) - 1) * 100) for m in CPI_FOOD_YOY}
    print("SANITY CHECK: median YoY change of these 24 foods vs official food inflation (%):")
    print(pd.DataFrame({"these_foods_median": yoy, "official_food_inflation": CPI_FOOD_YOY}).round(1).to_string(), "\n")
    return out, mae, best


if __name__ == "__main__":
    out, mae, best = run()
    show = ["Brown beans", "Local rice", "White garri", "Tomatoes", "Onions", "Yam tuber", "White maize", "Palm oil"]
    for m in sorted(out.month.unique()):
        print(f"--- {m}: estimate (80% range), last official in brackets ---")
        for f in show:
            r = out[(out.food == f) & (out.month == m)].iloc[0]
            print(f"  {f:14s} N{r.estimate:>7,.0f}  (N{r.low80:,.0f} to N{r.high80:,.0f})   [May: N{r.last_official:,.0f}]")