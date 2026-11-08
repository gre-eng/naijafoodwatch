"""
provenance.py - validates a price series and grades how far it can be trusted.
Grades: A high, B good, C indicative only, D not reliable (no guidance is shown).
Estimated data (e.g. the rice notice in historical_data.py) can never score above C.
"""
from __future__ import annotations

import time

MIN_REAL_MONTHS = 24
FRESH_MAX_MONTHS = 6
JUMP_LIMIT = 0.50      # month-to-month change above this is flagged for review
LABELS = {"A": "High", "B": "Good", "C": "Indicative only", "D": "Not reliable"}


def _ym(s: str) -> int:
    y, m = map(int, s.split("-"))
    return y * 12 + m


def validate(series: dict, source: str, estimated_note: str | None = None) -> dict:
    months = sorted(series)
    now = time.localtime()
    behind = now.tm_year * 12 + now.tm_mon - _ym(months[-1])
    span = _ym(months[-1]) - _ym(months[0]) + 1
    coverage = len(months) / span
    gaps = [_ym(b) - _ym(a) - 1 for a, b in zip(months, months[1:])]
    max_gap = max(gaps) if gaps else 0
    bad = [m for m in months if not series[m] or series[m] <= 0]
    jumps = [b for a, b in zip(months, months[1:])
             if _ym(b) - _ym(a) == 1 and series[a] > 0 and abs(series[b] / series[a] - 1) > JUMP_LIMIT]
    checks = [
        ["Prices are positive numbers", not bad, "all valid" if not bad else f"{len(bad)} invalid value(s)"],
        ["Enough real readings", len(months) >= MIN_REAL_MONTHS, f"{len(months)} real months (minimum {MIN_REAL_MONTHS})"],
        ["Recent", behind <= FRESH_MAX_MONTHS, f"latest real reading is {behind} month(s) old"],
        ["Few gaps", max_gap <= 6, f"longest gap {max_gap} month(s); {coverage * 100:.0f}% of months are real"],
        ["No suspicious jumps", not jumps, "none above 50% in a month" if not jumps else f"{len(jumps)} jump(s) above 50%, e.g. {jumps[0]}"],
    ]
    if bad or len(months) < MIN_REAL_MONTHS or behind > FRESH_MAX_MONTHS:
        grade = "D"
    elif coverage >= 0.70 and behind <= 3 and not jumps:
        grade = "A"
    elif coverage >= 0.50:
        grade = "B"
    else:
        grade = "C"
    if estimated_note and grade in "AB":
        grade = "C"
    return dict(grade=grade, label=LABELS[grade], coverage=round(coverage, 2), first=months[0], last=months[-1],
                behind=behind, real_months=len(months), max_gap=max_gap, checks=checks,
                estimated=bool(estimated_note), note=estimated_note or "", source=source)