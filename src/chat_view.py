"""
chat_view.py - the SIMPLE view of NaijaFoodWatch: a short-message food price assistant for ordinary Nigerians.
(The technical view, for researchers and agencies, is separate.)

It chats in short messages, shows a small chart, and can answer questions about the app itself.
Optional: set the ANTHROPIC_API_KEY environment variable and unmatched questions go to Claude, which is
told to use only the facts below. Without a key, a built-in rule brain answers.

How it behaves when NBS is late (three tiers, never a bare refusal):
  1. Newest official price is recent enough  -> today's estimate + a 3-month outlook (as before).
  2. Official price is older, but today is still inside the tested horizon -> today's estimate + a SHORTER
     outlook, only as far as the back-test has validated. Nothing is extrapolated beyond that.
  3. Official price is older than the tested horizon -> no estimate; show the last official price, how old
     it is, and tell the person to check their market.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from data.official_prices import get_official_prices
except Exception:      # openpyxl missing, or no reports found: fall back to the prices passed in
    get_official_prices = None


@st.cache_data(show_spinner=False)
def _official():
    return get_official_prices() if get_official_prices else {}


def _nc():
    """The back-tested model, or None if it cannot be built (too little consecutive official data, etc.).
    A None here does NOT stop the app: foods then fall back to 'last official price only'."""
    try:
        return _nowcast()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Checking which foods I can estimate...")
def _estimable():
    """Foods the chatbot can answer about right now. With the tiered fallback every food that has an official
    price is answerable (at worst: last official price only), so this list is only empty if no price files load."""
    try:
        prices = _official()
        if not prices:
            return [], "No official NBS price files could be read (check data/nbs_reports, and that openpyxl is installed)."
        nc = _nc()
        ok = []
        for k, v in prices.items():
            try:
                get_facts(k, v, nc)
                ok.append(k)
            except Exception:
                pass
        if ok:
            return ok, None
        return [], "The price files loaded, but none of the foods could be read. Check data/nbs_reports."
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner="Checking the model against past prices...")
def _nowcast():
    from src.nowcast import build
    return build(_official())


PLACEHOLDER = "Pick a food..."
SHELF_DAYS = [("frozen", 60), ("tomato", 10), ("plantain", 7), ("fresh", 3), ("beef", 3), ("chicken", 14), ("potato", 60), ("onion", 90),
              ("yam", 120), ("garri", 120), ("beans", 240), ("rice", 300), ("maize", 180), ("oil", 180)]   # rough assumptions, needs expert review
STORAGE_COST = {"good": 0.01, "fair": 0.02, "poor": 0.04}   # rough assumption: share of price per month
STORAGE_NAME = {"good": "dry, sealed", "fair": "basic bags", "poor": "open or damp"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
GENERIC = {"local", "white", "brown", "yellow", "fresh", "frozen", "oil", "imported", "ripe", "unripe", "sweet", "irish",
           "wings", "feet", "boneless", "tuber", "bottle", "bulb"}   # help pick between similar foods, but cannot identify one alone

# ---- Plain-language answers used by the rule brain and given to Claude as facts. -------------------------------
# NOTE: your pasted chat_view.py used KB and SYSTEM but never defined them. These are defaults - replace the wording
# with your own if you already have it.
KB = {
    "estimate": "My numbers are estimates built from official NBS prices. Where NBS hasn't published yet, I project forward with a method I tested on past months, and I show a range, not a single price.",
    "chart": "The solid line is official NBS prices, the dotted line is my middle guess, and the shaded area is the range I could be off by.",
    "source": "Prices come from the National Bureau of Statistics (NBS) Selected Food Price Watch reports. Anything after the latest report is my model's estimate, not NBS data.",
    "range": "The range is where I expect the price to land about 8 times in 10, based on how far off I was on past months. It gets wider the further ahead I look.",
    "technical": "The Technical view in the sidebar has the full model comparison and diagnostics for researchers and agencies.",
    "inflation": "Food inflation is how much faster food prices are rising compared with a year ago. A high number means your naira buys less food than it did last year.",
    "oil": "Fuel affects food prices because food has to be transported. When oil gets dearer, moving food usually costs more.",
    "naira": "A weaker naira makes imported goods and farm inputs like fertiliser and machinery cost more, which tends to push food prices up.",
    "storage": "Dry, sealed storage keeps stock safe the longest. Open or damp storage risks spoilage and loss, which can cost more than a higher price earns you.",
    "area": "NBS prices are national averages. Prices in your own market or state can be higher or lower, so check locally before you act.",
    "app": "I'm NaijaFoodPrice. Tell me a food and I'll give you its latest official price, an estimate for today, where it may be heading, and whether to buy or sell.",
}
SYSTEM = ("You are NaijaFoodPrice, a short-message food price assistant for ordinary Nigerians. Use ONLY the facts provided. "
          "Never invent prices or dates. Answer in 2-4 short sentences of plain English. If the facts don't cover the question, "
          "say so and suggest checking the local market. Never promise that a price will rise or fall.")


# ---------- helpers ----------
def _ym(s):
    y, m = map(int, s.split("-"))
    return y * 12 + m - 1


def _label(s):
    y, m = map(int, s.split("-"))
    return f"{MONTHS[m - 1]} {y}"


def _unit(key):
    if "bottle" in key.lower():
        return "bottle"
    m = re.search(r"\(\s*\d*\s*(kg|l)\s*\)", key, re.I)
    return "litre" if m and m.group(1).lower() == "l" else "kilogram"


def _short(u):
    return {"litre": "litres", "bottle": "bottles"}.get(u, "kg")


def _base(key):
    return re.sub(r"\s*\(.*?\)", "", key).strip()


def _naira(x):
    return f"N{x:,.0f}"


def _stem(w):
    return w[:-2] if w.endswith("es") else w[:-1] if w.endswith("s") else w


def match_food(text, prices):
    words = {_stem(w) for w in re.findall(r"[a-z]+", text.lower())}
    if words & {"gari", "cassava"}:
        words.add("garri")
    best, score = None, 0.99          # a food needs at least one distinctive word (rice, garri...), not just "oil" or "white"
    for key in prices:
        ws = [w for w in re.findall(r"[a-z]+", _base(key).lower()) if len(w) > 2]
        s = sum((0.5 if w in GENERIC else 1) for w in ws if _stem(w) in words)
        if s > score:
            best, score = key, s
    return best


# ---------- numbers ----------
class StaleData(Exception):
    pass


def get_facts(key, monthly, nc):
    """Everything the chat needs about one food. Never refuses: see the three tiers in the module docstring.
    f['mode'] is 'estimate' (tiers 1-2) or 'verified' (tier 3: last official price only)."""
    months = sorted(monthly)
    last_m, last = months[-1], float(monthly[months[-1]])
    now = datetime.now()
    gap = max(0, now.year * 12 + now.month - 1 - _ym(last_m))
    base = _base(key)
    recent = [float(monthly[m]) for m in months if _ym(last_m) - _ym(m) < 12]
    f = dict(key=key, last_m=last_m, last=last, gap=gap, mode="verified", unit=_unit(key), now=now,
             shelf=next((d for k, d in SHELF_DAYS if k in key.lower()), 90),
             prev=(months[-2], float(monthly[months[-2]])) if len(months) > 1 else None,
             recent_real=sum(1 for m in months if _ym(last_m) - _ym(m) < 24), recent_span=min(24, _ym(last_m) - _ym(months[0]) + 1),
             hist=[(m, float(monthly[m])) for m in months[-18:]],
             # defaults for 'verified' mode: the last official price, nothing projected
             now_est=last, now_lo=last, now_hi=last, fut=last, lo=last, hi=last, pct3=0.0, direction=None, pos=None,
             target=last_m, n=gap, outlook=False, shortened=False,
             fc_idx=[], fc_mean=[], fc_lo=[], fc_hi=[], err_now=None, err_target=None)

    # Tier 3: no model, or the newest price is older than the horizon the back-test validated.
    if nc is None or base not in nc["foods"] or gap > nc["H"]:
        return f

    e = nc["foods"][base]
    n_ = min(gap + 3, nc["H"])                 # months from the last official price; never beyond the tested horizon
    now_c, now_lo, now_hi = (last, last, last) if gap == 0 else (e["c"][gap - 1], e["lo"][gap - 1], e["hi"][gap - 1])
    fut, lo, hi = e["c"][n_ - 1], e["lo"][n_ - 1], e["hi"][n_ - 1]
    pct3 = fut / now_c - 1
    direction = None
    # A direction call was only validated for a full 3-month look-ahead from a fresh price, so it is never made otherwise.
    if gap <= 1 and n_ == gap + 3 and nc["tau"] is not None and abs(np.log(fut / now_c)) / nc["sd"][base] >= nc["tau"]:
        direction = "up" if pct3 > 0 else "down"
    pos = (last - sum(recent) / len(recent)) / (sum(recent) / len(recent)) if len(recent) >= 3 and gap <= 1 else None
    idx = [str(pd.Period(last_m) + h) for h in range(1, n_ + 1)]
    f.update(mode="estimate", now_est=now_c, now_lo=now_lo, now_hi=now_hi, target=idx[-1], fut=fut, lo=lo, hi=hi, pct3=pct3,
             direction=direction, pos=pos, n=n_, outlook=n_ > gap, shortened=n_ < gap + 3,
             fc_idx=idx, fc_mean=e["c"][:n_], fc_lo=e["lo"][:n_], fc_hi=e["hi"][:n_],
             err_now=nc["mae"].get(max(gap, 1)), err_target=nc["mae"].get(n_))
    return f


def reason_lines(f, ss):
    out, fm, oil, exr = [], ss.cv_food_monthly, ss.cv_oil, ss.cv_exr
    if f["prev"] and abs(f["last"] / f["prev"][1] - 1) >= 0.01:
        out.append(f"{_base(f['key'])} itself moved {f['last'] / f['prev'][1] - 1:+.0%} in {_label(f['last_m'])}.")
    if fm:
        k = sorted(fm)
        out.append(f"Food prices nationwide are {'rising' if fm[k[-1]] > fm[k[0]] else 'easing'} ({fm[k[-1]]}% food inflation in {_label(k[-1])}).")
    o5, o6 = oil.get(2025), oil.get(2026)
    if o5 and o6 and abs(o6 / o5 - 1) > 0.03:
        out.append(f"Fuel {'costs more' if o6 > o5 else 'is cheaper'} (oil ${o6:.0f} a barrel vs ${o5:.0f} last year), so moving food {'costs more' if o6 > o5 else 'costs less'}.")
    e5, e6 = exr.get(2025), exr.get(2026)
    if e5 and e6 and abs(e6 / e5 - 1) > 0.03:
        out.append(f"The naira is {'weaker' if e6 > e5 else 'stronger'} ({_naira(e6)} to the dollar vs {_naira(e5)}), which {'pushes costs up' if e6 > e5 else 'helps a little'}.")
    return out[:3]


def _festive_note(f):
    """Nov/Dec/Jan caution. The estimates come from past NBS prices and add no festive bump, so say so plainly."""
    months = {f["now"].month, int(f["target"].split("-")[1])}
    if months & {11, 12, 1}:
        return ("Heads up: prices in Nigerian markets often climb around Christmas and New Year. My estimate comes from past NBS prices and "
                "does not add a festive bump, so the real price in your market may be higher.")
    return None


def decide(role, f, storage):
    if f["mode"] == "verified":
        if role == "sell" and f["shelf"] < 30:
            return "SELL SOON", f"It spoils in about {f['shelf']} days, so don't keep it long, whatever the price."
        return "CHECK YOUR MARKET FIRST", (f"My newest official price is {f['gap']} months old, too old for me to advise you properly. "
                                           "Check today's price in your market before you decide.")
    carry, tgt, d = STORAGE_COST[storage] * 3, _label(f["target"]), f["direction"]
    if role == "sell":
        if f["shelf"] < 30:
            return "SELL NOW", f"It spoils in about {f['shelf']} days, so don't keep it."
        if storage == "poor":
            return "SELL NOW", "Open or damp storage can ruin stock, and I can't promise a higher price. Sell now."
        if d == "up" and f["pct3"] > carry:
            return "STORE", f"Keep it and sell around {tgt}. Prices look set to rise about {f['pct3']:.0%}, more than storing costs you (about {carry:.0%})."
        if f["pos"] is not None and f["pos"] >= 0.10:
            return "SELL NOW", f"The price is {f['pos']:.0%} above usual. A good time to sell."
        return "SELL NOW", f"I can't tell if the price will go up or down, and storing costs about {carry:.0%} over 3 months. Selling now is the safer choice."
    if d == "up" and f["pct3"] >= 0.05 and f["shelf"] >= 30:
        return "BUY NOW", f"Prices look set to rise about {f['pct3']:.0%}. Buy now, and stock up if you can store it safely."
    if d == "down":
        return "WAIT", f"Prices look set to fall about {abs(f['pct3']):.0%}. Buy only what you need today."
    if f["pos"] is not None and f["pos"] <= -0.07:
        return "BUY NOW", f"The price is {abs(f['pos']):.0%} below usual. A good time to buy."
    return "BUY NOW", "I can't tell where the price is heading, so buy what you need now and don't stock up on a guess."


def make_chart(f):
    d = lambda ms: [pd.Timestamp(m + "-01") for m in ms]
    hx = d([m for m, _ in f["hist"]])
    hy = [v for _, v in f["hist"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines+markers", line=dict(color="#0a3d62", width=3), name="Official NBS prices"))
    if f["fc_idx"]:
        fx = d(f["fc_idx"])
        fig.add_trace(go.Scatter(x=[hx[-1]] + fx + fx[::-1] + [hx[-1]], y=[hy[-1]] + f["fc_hi"] + f["fc_lo"][::-1] + [hy[-1]], fill="toself",
                                 fillcolor="rgba(18,140,126,0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[hx[-1]] + fx, y=[hy[-1]] + f["fc_mean"], mode="lines", line=dict(color="#128C7E", width=3, dash="dot"), name="Middle guess"))
    fig.update_layout(height=270, margin=dict(t=10, b=10, l=10, r=10), template="plotly_white", yaxis_title="Naira",
                      legend=dict(orientation="h", y=-0.2), dragmode=False)
    return fig


def _all_facts():
    nc = _nc()
    ss = st.session_state
    for k in ss.cv_prices:
        if k not in ss.cv_facts:
            try:
                ss.cv_facts[k] = get_facts(k, ss.cv_prices[k], nc)
            except Exception:
                ss.cv_facts[k] = None
    return {k: v for k, v in ss.cv_facts.items() if v}


def _fact_line(k, f):
    if f["mode"] == "verified":
        return (f"- {_base(k)} per {f['unit']}: latest official {_naira(f['last'])} ({_label(f['last_m'])}, {f['gap']} months old); "
                f"no reliable estimate available")
    return (f"- {_base(k)} per {f['unit']}: latest {_naira(f['last'])} ({_label(f['last_m'])}), about {_naira(f['now_est'])} today (estimate), "
            f"expected {f['pct3']:+.0%} by {_label(f['target'])} to {_naira(f['fut'])}; real prices for {f['recent_real']} of the last {f['recent_span']} months")


def ask_claude(question):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        ss = st.session_state
        facts = "\n".join(_fact_line(k, f) for k, f in _all_facts().items())
        chat = "\n".join(f"{m['role']}: {m['text']}" for m in ss.cv_msgs[-6:] if m["text"])
        r = anthropic.Anthropic(api_key=key).messages.create(
            model=os.getenv("NAIJAFOOD_MODEL", "claude-haiku-5-5"), max_tokens=300,
            system=SYSTEM + "\nAPP FACTS: " + " ".join(KB.values()) + "\nFOODS:\n" + facts,
            messages=[{"role": "user", "content": f"Recent chat:\n{chat}\n\nQuestion: {question}"}])
        return r.content[0].text
    except Exception:
        return None


# ---------- chat plumbing ----------
def _say(text, fig=None):
    st.session_state.cv_msgs.append({"role": "assistant", "text": text, "fig": fig})


def _user(text):
    st.session_state.cv_msgs.append({"role": "user", "text": text, "fig": None})


def _need_food():
    _say("Which food? Type its name or pick one from the list below.")


def handle_food(key):
    ss = st.session_state
    f = ss.cv_facts.get(key)
    if f is None:
        try:
            f = ss.cv_facts[key] = get_facts(key, ss.cv_prices[key], _nc())
        except Exception as e:
            return _say(f"Sorry, I can't estimate {_base(key)} right now. (Detail for the developer: {type(e).__name__}: {e})")
    ss.cv_food, ss.cv_stage, ss.cv_role = key, "role", None
    u, name = f["unit"], _base(key)

    if f["mode"] == "verified":
        _say(f"**{name}**: the latest official NBS price I have is **{_naira(f['last'])} per {u}**, for {_label(f['last_m'])}. "
             f"That is {f['gap']} months old and NBS hasn't published newer reports yet, so I can't give you a reliable estimate for today. "
             f"Please check your market's price.")
        _say("Here are the official prices so far.", make_chart(f))
    else:
        if f["gap"] >= 1:
            _say(f"**{name}**: the latest official NBS price I have is **{_naira(f['last'])} per {u}**, for {_label(f['last_m'])}. "
                 f"Newer NBS prices aren't in yet, so today's price is an estimate: most likely between **{_naira(f['now_lo'])} and {_naira(f['now_hi'])}**.")
        else:
            _say(f"**{name}** is **{_naira(f['last'])} per {u}** ({_label(f['last_m'])}, official NBS).")
        if not f["outlook"]:
            _say("I can't look any further ahead than this with any tested accuracy, so I won't guess at the coming months.")
        elif f["direction"]:
            word = "rise" if f["direction"] == "up" else "fall"
            _say(f"By **{_label(f['target'])}** I expect it to **{word} about {abs(f['pct3']):.0%}**, to around **{_naira(f['fut'])}** (between {_naira(f['lo'])} and {_naira(f['hi'])}).")
        else:
            short = (f" I'm only looking as far as {_label(f['target'])} because that's the furthest my method has been tested."
                     if f["shortened"] else "")
            _say(f"By **{_label(f['target'])}** I can't tell if it will go up or down. My best guess is **{_naira(f['fut'])}**, "
                 f"but it could well be anywhere from **{_naira(f['lo'])} to {_naira(f['hi'])}**.{short}")
        _say("Here's the picture. The solid line is official prices, the dotted line is my middle guess, and the shaded area is how far off I could be.", make_chart(f))
        why = reason_lines(f, ss)
        if why:
            _say("Things that can move it:\n" + "\n".join(f"- {w}" for w in why))
    festive = _festive_note(f)
    if festive:
        _say(festive)
    _say("Do you want to **buy**, or do you have **stock to sell**?")


def handle_role(role, said=None):
    ss = st.session_state
    if not ss.cv_food:
        return _need_food()
    _user(said or ("I want to buy" if role == "buy" else "I have stock to sell"))
    ss.cv_role = role
    f = ss.cv_facts[ss.cv_food]
    a, why = decide(role, f, ss.cv_storage)
    _say(f"My advice: **{a}**. {why}")
    _say(f"How many {_short(f['unit'])} do you {'have' if role == 'sell' else 'want to buy'}?")
    ss.cv_stage = "qty"


def handle_qty(n):
    ss = st.session_state
    f = ss.cv_facts[ss.cv_food]
    u, lo, mid, hi = _short(f["unit"]), n * f["now_lo"], n * f["now_est"], n * f["now_hi"]
    if f["mode"] == "verified":
        _say(f"At the last official price ({_label(f['last_m'])}), {n:,.0f} {u} cost about **{_naira(mid)}**. "
             f"Today's price is probably different, so check your market.")
    elif ss.cv_role == "sell":
        cost = mid * STORAGE_COST[ss.cv_storage] * 3
        _say(f"{n:,.0f} {u} is worth about **{_naira(mid)}** today (between {_naira(lo)} and {_naira(hi)}). Storing it for 3 months would cost you roughly "
             f"**{_naira(cost)}**, and I can't promise the price will rise enough to cover that.")
    else:
        later = (f" By {_label(f['target'])} it could cost between **{_naira(n * f['lo'])} and {_naira(n * f['hi'])}**."
                 if f["outlook"] else "")
        _say(f"{n:,.0f} {u} costs about **{_naira(mid)}** today (between {_naira(lo)} and {_naira(hi)}).{later}")
    if ss.cv_role == "sell":
        _say("Where do you keep your stock?")
        ss.cv_stage = "storage"
    else:
        _say("Anything else? Ask me, or pick another food below.")
        ss.cv_stage = "done"


def handle_storage(k, said=None):
    ss = st.session_state
    _user(said or STORAGE_NAME[k].capitalize())
    ss.cv_storage = k
    a, why = decide("sell", ss.cv_facts[ss.cv_food], k)
    _say(f"With {STORAGE_NAME[k]} storage, my advice is **{a}**. {why}")
    _say("Anything else? Ask me, or pick another food below.")
    ss.cv_stage = "done"


def _compare(kind):
    fs = _all_facts()
    if len(fs) < 2:
        return _say("I don't have enough foods with recent prices to compare yet.")
    if kind == "price":
        kg = sorted((f for f in fs.values() if f["unit"] == "kilogram"), key=lambda f: f["now_est"])
        if len(kg) < 2:
            return _say("I don't have enough foods priced per kg to compare.")
        _say(f"Among foods priced per kg, **{_base(kg[0]['key'])}** is the cheapest (about {_naira(kg[0]['now_est'])}) and "
             f"**{_base(kg[-1]['key'])}** is the dearest (about {_naira(kg[-1]['now_est'])}).")
    else:
        ch = sorted(((f["last"] / f["prev"][1] - 1, f) for f in fs.values() if f["prev"]), key=lambda x: x[0])
        up, dn = ch[-1], ch[0]
        _say(f"I can't reliably say which foods will rise most from here. Looking only at official prices, between {_label(up[1]['prev'][0])} and "
             f"{_label(up[1]['last_m'])}, **{_base(up[1]['key'])}** rose the most ({up[0]:+.0%}). "
             + (f"**{_base(dn[1]['key'])}** fell the most ({dn[0]:+.0%})." if dn[0] < -0.005 else "No food fell."))


def _confidence():
    ss = st.session_state
    if not ss.cv_food:
        return _say(KB["estimate"] + " Pick a food and I'll tell you how sure I am about it.")
    f = ss.cv_facts[ss.cv_food]
    have = f"I have official NBS prices for {f['recent_real']} of the last {f['recent_span']} months (up to {_label(f['last_m'])})"
    if f["mode"] == "verified":
        return _say(f"{have}, but nothing newer, and the last one is {f['gap']} months old. That's too old for me to estimate today's price reliably, "
                    f"so I'm showing the last official price only. Please check your market.")
    if f["gap"] >= 1 and f["err_now"]:
        tail = f", and {f['n']} months after that by about {f['err_target']:.0%}" if f["err_target"] and f["n"] > f["gap"] else ""
        _say(f"{have}, but none after that. I tested my method on past months: when the newest price was {f['gap']} months old, I was off by about "
             f"{f['err_now']:.0%} on average{tail}. So treat my numbers as a range, and check your market before you act.")
    else:
        _say(f"{have}. When I tested my method on past months, my estimates were off by about {f['err_target']:.0%} on average {f['n']} months ahead. "
             f"Prices can still surprise us, so check your market.")


def handle_text(text):
    ss, t = st.session_state, text.lower().strip()
    _user(text)
    if ss.cv_stage == "storage":
        for k, pat in (("good", r"seal|dry|good|silo"), ("fair", r"bag|fair|basic"), ("poor", r"open|damp|wet|poor|outside")):
            if re.search(pat, t):
                ss.cv_msgs.pop()
                return handle_storage(k, text)
    food = match_food(text, {k: None for k in ss.cv_names.values()})
    if food and food != ss.cv_food:
        return handle_food(food)
    num = re.search(r"\d[\d,]*\.?\d*", text)
    if num and ss.cv_food and ss.cv_role and ss.cv_stage in ("qty", "done"):
        return handle_qty(float(num.group().replace(",", "")))
    if re.search(r"\b(buy|buying|purchase)\b", t) and not re.search(r"\bsell", t):
        ss.cv_msgs.pop()
        return handle_role("buy", text)
    if re.search(r"\b(sell|selling|my stock|i have)\b", t):
        ss.cv_msgs.pop()
        return handle_role("sell", text)
    f = ss.cv_facts.get(ss.cv_food) if ss.cv_food else None
    if re.search(r"how sure|accura|trust|reliab|confiden|estimate|interpolat|filled|guess|real price", t):
        _say(KB["estimate"]) if re.search(r"estimate|interpolat|filled|guess|real price", t) else None
        return _confidence()
    if re.search(r"\b(old|outdated|stale|up to date|updated?|latest)\b|how recent|why.*(may|months? ago|old)", t) and f:
        return _say(f"The newest NBS report I have is for {_label(f['last_m'])}. NBS normally publishes about a month after each month ends, but I don't have anything newer yet, so I estimate today's price and give a range."
                    if f["mode"] == "estimate" else
                    f"The newest NBS report I have is for {_label(f['last_m'])}, which is {f['gap']} months old. Nothing newer has been published yet, and that's too old for me to estimate from.")
    if re.search(r"\bwhy\b|reason|because|explain|what.*push", t):
        if not f:
            return _need_food()
        w = reason_lines(f, ss)
        return _say("Mainly this:\n" + "\n".join(f"- {x}" for x in w) if w else "Mostly the recent trend in this food's own prices.")
    if re.search(r"cheap|expensive|dear|highest|lowest", t):
        return _compare("price")
    if re.search(r"(which|what).*(food|one).*(rise|rising|fall|falling|up|down|increase)|rise most|fall most|going up|going down", t):
        return _compare("move")
    if re.search(r"chart|graph|line|dotted|shade|picture|star|illustrat", t):
        _say(KB["chart"])
        return _say("Here it is again.", make_chart(f)) if f else None
    for pat, k in ((r"where.*(data|price|number)|source|nbs|from where|who gives", "source"), (r"range|margin|between", "range"),
                   (r"technical|researcher|economist|professor|agenc|model|sarima", "technical"), (r"inflation", "inflation"),
                   (r"\boil\b|brent|fuel|petrol|diesel", "oil"), (r"naira|dollar|exchange|forex", "naira"),
                   (r"stor(e|age)|spoil|damp|sealed", "storage"), (r"state|area|market|lagos|kano|abuja|port harcourt|near me", "area"),
                   (r"who are you|what can you|what are you|\bhelp\b|how does this work", "app")):
        if re.search(pat, t):
            return _say(KB[k])
    if re.search(r"\b(hi|hello|hey|good (morning|afternoon|evening)|how far)\b", t):
        return _say("Hello! Which food do you want to check today?")
    if re.search(r"thank|thanks|great|nice|good job", t):
        return _say("You're welcome! Want to check another food?")
    if re.search(r"another|other food|new food|start over|restart", t):
        return _need_food()
    reply = ask_claude(text)
    _say(reply or "I'm not sure I got that. You can ask about a food's price, why it's moving, how sure I am, or what the chart means.")


def _pick():
    ss = st.session_state
    v = ss.cv_pick
    if v != PLACEHOLDER:
        _user(v)
        handle_food(ss.cv_names[v])
        ss.cv_pick = PLACEHOLDER


def _restart():
    keep = ("cv_prices", "cv_fc", "cv_food_monthly", "cv_exr", "cv_oil", "cv_names")
    for k in [k for k in st.session_state if k.startswith("cv_") and k not in keep]:
        del st.session_state[k]


def _data_age_banner(prices):
    """Tell the person, once, how old the newest official price is and what the app is doing about it."""
    if not prices:
        return
    newest = max(max(v) for v in prices.values())
    now = datetime.now()
    gap = now.year * 12 + now.month - 1 - _ym(newest)
    if gap < 2:
        return
    nc = _nc()
    if nc is not None and gap <= nc["H"]:
        st.info(f"The newest official NBS prices are for {_label(newest)} ({gap} months old). NBS hasn't published newer reports yet, "
                f"so today's prices are model estimates, and I only look ahead as far as my method has been tested.")
    else:
        st.warning(f"The newest official NBS prices are for {_label(newest)} ({gap} months old), too old to estimate from. "
                   f"I'm showing the last official prices only. Add newer NBS reports to data/nbs_reports to get estimates again.")


def render_chat_view(prices, forecast_fn, food_monthly, exr, oil):
    ss = st.session_state
    prices = _official() or prices
    ss.cv_prices, ss.cv_fc, ss.cv_food_monthly, ss.cv_exr, ss.cv_oil = prices, forecast_fn, food_monthly, exr, oil
    est, ss.cv_err = _estimable()
    ss.cv_names = {f"{_base(k)} (per {_unit(k)})": k for k in est}
    if "cv_msgs" not in ss:
        ss.cv_msgs = [{"role": "assistant", "fig": None, "text": "Welcome to **NaijaFoodPrice**! What's your favourite food? Type it below, or pick one from the list."}]
        ss.cv_stage, ss.cv_food, ss.cv_role, ss.cv_storage, ss.cv_facts, ss.cv_pick = "food", None, None, "fair", {}, PLACEHOLDER

    typed = st.chat_input("Type a food, or ask me anything...")
    if typed:
        handle_text(typed)

    st.markdown("### NaijaFoodPrice")
    if ss.cv_err:
        st.error(f"I can't answer right now. Detail: {ss.cv_err}")
    else:
        _data_age_banner(prices)
    for i, m in enumerate(ss.cv_msgs):
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("fig") is not None:
                st.plotly_chart(m["fig"], use_container_width=True, config={"displayModeBar": False}, key=f"cv_fig_{i}")

    if ss.cv_stage == "role":
        c1, c2 = st.columns(2)
        c1.button("I want to buy", use_container_width=True, on_click=handle_role, args=("buy",), key="cv_b_buy")
        c2.button("I have stock to sell", use_container_width=True, on_click=handle_role, args=("sell",), key="cv_b_sell")
    elif ss.cv_stage == "storage":
        for col, k in zip(st.columns(3), ("good", "fair", "poor")):
            col.button(STORAGE_NAME[k].capitalize(), use_container_width=True, on_click=handle_storage, args=(k,), key=f"cv_s_{k}")
    elif ss.cv_stage == "done":
        for col, q in zip(st.columns(3), ("Why?", "How sure are you?", "What does the chart mean?")):
            col.button(q, use_container_width=True, on_click=handle_text, args=(q,), key=f"cv_q_{q}")

    st.selectbox("Pick a food from the list", [PLACEHOLDER] + list(ss.cv_names), key="cv_pick", on_change=_pick)
    st.button("Start over", on_click=_restart, key="cv_restart")