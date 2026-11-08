from datetime import datetime

from data.official_prices import get_official_prices
from src.nowcast import panel, build

prices = get_official_prices()
print(f"Foods loaded: {len(prices)}\n")

print("Newest official month per food:")
for name, series in prices.items():
    print(f"  {name:32s} {max(series)}  ({len(series)} months)")

logs = panel(prices)
L, F = len(logs), logs.shape[1]
print(f"\nConsecutive run: {logs.index[0]} to {logs.index[-1]}  ({L} months, {F} foods)")

nc = build(prices)
H = nc["H"]

now = datetime.now()
last_y, last_m = map(int, logs.index[-1].split("-"))
gap = (now.year * 12 + now.month) - (last_y * 12 + last_m)

print(f"H (months I can validate ahead): {H}")
print(f"gap (months since newest price): {gap}")
if gap + 3 <= H:
    print("OK: the chat should be able to estimate now.")
else:
    print(f"NOT OK: need gap + 3 <= H, but {gap} + 3 = {gap + 3} > {H}. Add newer NBS reports.")