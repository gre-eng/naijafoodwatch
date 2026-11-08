"""
nbs_loader.py - reads the official NBS "Selected Food Price Watch" monthly Excel tables.

Each monthly report contains, per item: the national average price for that month, the previous
month and the same month a year earlier, plus six zone averages and the highest/lowest state.
Drop every monthly file (the .zip or the .xlsx) into data/nbs_reports/ and this builds a monthly
price history. Download them from: https://microdata.nigerianstat.gov.ng/index.php/catalog/162

Check what was read:   python -m data.nbs_loader
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import openpyxl

REPORTS = Path(__file__).resolve().parent / "nbs_reports"
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
# Map an NBS report label to the name already used in historical_data.py so the histories join up.
# Add more once you know your old names (e.g. "Garri white": "Garri white (1kg)").
ALIASES = {"Beans Brown": "Brown beans (1kg)", "Garri white": "White garri (1kg)",
           "Tomatoes, fresh": "Tomatoes (1kg)", "Onions, fresh": "Onion bulb (1kg)"}
# Not aliased on purpose: rice (NBS tracks several grades; your "Local rice (1kg)" matches none of them)
# and palm oil (NBS labels it 75cl, your file says 1 litre: confirm the unit first).
# NBS renamed items in 2025. These pairs were matched by IDENTICAL prices for Dec 2024 (old-name report vs the
# year-ago column of the new-name Dec 2025 report), so the two names are the same item.
LABEL_ALIASES = {
    "Agric eggs(medium size price of one)": "Agric hen eggs,",
    "Beans brown,sold loose": "Beans Brown",
    "Beans:white black eye. sold loose": "Beans white",
    "Beef,boneless": "Beef Boneless",
    "Catfish (obokun) fresh": "Cat fresh fish",
    "Chicken Feet": "Chicken feet",
    "Evaporated tinned milk(peak), 170g": "Tin Milk-Evaporated, Peak Milk, 150g",
    "Frozen chicken": "Chicken meat (Frozen)",
    "Gari white,sold loose": "Garri white",
    "Gari yellow,sold loose": "Garri Yellow",
    "Irish potato": "Irish Potatoe",
    "Mackerel : frozen": "Mackerel, Frozen",
    "Maize grain white sold loose": "Maize (Corn) Grains (White) sold loose",
    "Onion bulb": "Onions, fresh",
    "Palm oil: 1 bottle,specify bottle": "Palm oil, 75cl",
    "Plantain(ripe)": "Plantain (ripe)",
    "Plantain(unripe)": "Plantain (unripe)",
    "Rice local sold loose": "Rice Local, short-Grained",
    "Rice,imported high quality sold loose": "Rice Long-Grained (Imported)",
    "Sweet potato": "Sweet potatoes",
    "Tilapia fish (epiya) fresh": "Tilapia fresh fish (Epiya)",
    "Titus:frozen": "Titus, frozen",
    "Tomato": "Tomatoes, fresh",
    "Vegetable oil:1 bottle,specify bottle": "Vegetable Oil, 75cl",
    "Wheat flour: prepacked (golden penny 2kg)": "Wheat Flour, Prepacked (2kg)",
    "Yam tuber": "Yam Tuber"
}


def _norm(label: str) -> str:
    return LABEL_ALIASES.get(label.strip(), label.strip())


_NOT_PER_KG = re.compile(r"\bcl\b|\d\s?cl|crate|piece|tin milk|bread|eggs|bottle|\d+\s?g\b|2kg|oil", re.I)


BOTTLE_OK = {"Palm oil, 75cl", "Vegetable Oil, 75cl"}   # price is per 75cl bottle (NBS table label)


def per_kg(label: str) -> bool:
    return not _NOT_PER_KG.search(label)


def _ym(header: str) -> str:
    """'Average of May-26' -> '2026-05'"""
    mon, yy = header.split()[-1].split("-")
    return f"{2000 + int(yy)}-{MONTHS[mon[:3].lower()]:02d}"


def parse_workbook(wb) -> dict:
    ws, zs = wb.worksheets[0], (wb.worksheets[1] if len(wb.worksheets) > 1 else None)
    head = [c.value for c in ws[1]]
    yago, prev, cur = (_ym(h) for h in head[1:4])
    zones = {}
    if zs:
        zh = [c.value for c in zs[1]][1:7]
        for r in zs.iter_rows(min_row=2, values_only=True):
            if r[0]:
                zones[_norm(r[0])] = {z: float(v) for z, v in zip(zh, r[1:7]) if isinstance(v, (int, float))}
    items = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if not r[0] or not isinstance(r[3], (int, float)):
            continue
        label = _norm(r[0])
        items[label] = dict(yago=r[1], prev=r[2], cur=r[3], highest=r[6], lowest=r[7], zones=zones.get(label, {}))
    return dict(month=cur, prev=prev, yago=yago, items=items)


def _open_all(folder: Path):
    for f in sorted(folder.rglob("*")):
        if f.suffix.lower() == ".xlsx":
            yield f.name, openpyxl.load_workbook(f, data_only=True)
        elif f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    if n.lower().endswith(".xlsx") and not n.startswith("__MACOSX"):
                        yield f"{f.name}:{n}", openpyxl.load_workbook(io.BytesIO(z.read(n)), data_only=True)


def load_reports(folder: Path = REPORTS):
    """Returns (series, zones, files). series[label]["YYYY-MM"] = national average price per kg.
    A month reported in its own report beats the same month quoted as a comparison in a later one."""
    best, zones, files = {}, {}, []
    folder.mkdir(exist_ok=True)
    for name, wb in _open_all(folder):
        try:
            rep = parse_workbook(wb)
        except Exception as exc:
            print(f"Skipped {name}: {exc}")
            continue
        files.append((name, rep["month"]))
        for label, it in rep["items"].items():
            if not (per_kg(label) or label in BOTTLE_OK):
                continue
            for month, price, prio in ((rep["yago"], it["yago"], 1), (rep["prev"], it["prev"], 1), (rep["month"], it["cur"], 2)):
                if isinstance(price, (int, float)) and price > 0 and best.get((label, month), (0, 0))[0] <= prio:
                    best[(label, month)] = (prio, round(float(price), 2))
            if it["zones"] and (label not in zones or rep["month"] > zones[label]["month"]):
                zones[label] = dict(month=rep["month"], zones={k.title(): round(v, 2) for k, v in it["zones"].items()},
                                    highest=it["highest"], lowest=it["lowest"])
    series = {}
    for (label, month), (_, price) in best.items():
        series.setdefault(label, {})[month] = price
    return series, zones, files


if __name__ == "__main__":
    series, zones, files = load_reports()
    print(f"Read {len(files)} report(s): {files}\n")
    for label, s in sorted(series.items()):
        print(f"{label:45s} {len(s):3d} months  {min(s)} to {max(s)}")
    print("\nTip: more monthly files = longer history. Each file adds this month, last month and the same month last year.")