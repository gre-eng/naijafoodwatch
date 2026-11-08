"""
official_prices.py - the chatbot's price source: official NBS Selected Food Price Watch reports only.
Reads every report in data/nbs_reports/ (see nbs_loader.py). No estimates, no interpolation, no legacy values.
"""
from __future__ import annotations

from data.nbs_loader import load_reports

NAMES = {  # NBS label -> friendly name (units are in the brackets). Order = tie-break order for typed names.
    "Rice Local, short-Grained": "Local rice (1kg)", "Rice Long-Grained (Imported)": "Imported rice (1kg)",
    "Beans Brown": "Brown beans (1kg)", "Beans white": "White beans (1kg)",
    "Garri white": "White garri (1kg)", "Garri Yellow": "Yellow garri (1kg)",
    "Maize (Corn) Grains (White) sold loose": "White maize (1kg)", "Yam Tuber": "Yam tuber (1kg)",
    "Tomatoes, fresh": "Tomatoes (1kg)", "Onions, fresh": "Onions (1kg)",
    "Irish Potatoe": "Irish potatoes (1kg)", "Sweet potatoes": "Sweet potatoes (1kg)",
    "Plantain (ripe)": "Ripe plantain (1kg)", "Plantain (unripe)": "Unripe plantain (1kg)",
    "Palm oil, 75cl": "Palm oil (75cl bottle)", "Vegetable Oil, 75cl": "Vegetable oil (75cl bottle)",
    "Beef Boneless": "Beef, boneless (1kg)", "Chicken meat (Frozen)": "Chicken, frozen (1kg)",
    "Chicken Wings": "Chicken wings (1kg)", "Chicken feet": "Chicken feet (1kg)",
    "Mackerel, Frozen": "Mackerel, frozen (1kg)", "Titus, frozen": "Titus fish, frozen (1kg)",
    "Cat fresh fish": "Catfish, fresh (1kg)", "Tilapia fresh fish (Epiya)": "Tilapia, fresh (1kg)",
}


def get_official_prices(min_months: int = 24) -> dict:
    """{friendly name: {"YYYY-MM": price}} for foods with at least min_months of official prices."""
    series = load_reports()[0]
    return {NAMES.get(k, f"{k} (1kg)"): {m: v[m] for m in sorted(v)} for k, v in sorted(series.items(), key=lambda kv: list(NAMES).index(kv[0]) if kv[0] in NAMES else 99)
            if len(v) >= min_months}