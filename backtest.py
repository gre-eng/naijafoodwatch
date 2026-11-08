"""
backtest.py - Would the guidance have beaten simple rules on past prices?
For each food, for past months, it uses ONLY earlier data to produce a signal, then checks what the
real price did over the next 3 months. Run from the project root:   python backtest.py
Results are saved to backtest_results.csv. Needs the same data as server.py.
"""
import pandas as pd

from src.forecasting_engine import commodity_monthly_forecast
from src.harvest_signal import analyse, STORAGE_COST_PER_MONTH, H


def _ym(s):
    y, m = map(int, s.split("-"))
    return y * 12 + m


def _plus(s, n):
    t = _ym(s) - 1 + n
    return f"{t // 12}-{t % 12 + 1:02d}"


def backtest_food(name, series, forecast_fn=commodity_monthly_forecast, role="grow", storage="fair",
                  weather="mixed", min_train=24, max_origins=12):
    months = sorted(series)
    carry = STORAGE_COST_PER_MONTH[storage] * H
    origins = [m for m in months[min_train - 1:] if _plus(m, H) in series][-max_origins:]
    rows = []
    for o in origins:
        train = {m: series[m] for m in months if m <= o}          # no peeking at the future
        a = analyse(name, train, forecast_fn, role, storage, weather)
        real = series[_plus(o, H)] / series[o] - 1                  # what actually happened
        if role == "grow":   # storing pays only if the price rose by more than storage cost
            gain = real - carry if a["action"] == "STORE" else 0.0
            right = (real > carry) == (a["action"] == "STORE")
            base_store = real - carry
        else:                # buying now pays if prices rose; waiting pays if they fell
            gain = real if a["action"] == "BUY" else 0.0
            right = (real > 0) == (a["action"] == "BUY")
            base_store = real
        rows.append(dict(origin=o, action=a["action"], real_change=real, followed_gain=gain, always_act_gain=base_store, right=right))
    return rows


def run(role="grow"):
    from server import PRICES, QUALITY
    out = []
    for name, ser in PRICES.items():
        rows = backtest_food(name, ser, role=role)
        if len(rows) < 4:
            continue
        d = pd.DataFrame(rows)
        out.append(dict(food=name, grade=QUALITY[name]["grade"], tests=len(d), signal_right_pct=round(100 * d.right.mean()),
                        avg_gain_following_signal_pct=round(100 * d.followed_gain.mean(), 1),
                        avg_gain_always_store_or_buy_pct=round(100 * d.always_act_gain.mean(), 1),
                        avg_gain_never_act_pct=0.0))
    res = pd.DataFrame(out)
    res.to_csv("backtest_results.csv", index=False)
    print(res.to_string(index=False) if len(res) else "Not enough history to back-test.")
    print("\nThe guidance only earns trust where 'following signal' beats BOTH 'always store/buy' and 'never act' consistently.")
    return res


if __name__ == "__main__":
    run()