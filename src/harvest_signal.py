"""
harvest_signal.py
=================
Early-warning guidance for NaijaFoodWatch Pro: translates price forecasts into
simple visual advice on when to SELL, STORE or BUY (supports SDG 1 and SDG 2).
Every signal is explainable: factors, confidence and assumptions are shown.
"""

from __future__ import annotations

import re

import plotly.graph_objects as go

SHELF_LIFE_DAYS = [("tomato", 10), ("pepper", 14), ("onion", 90), ("yam", 120), ("garri", 120),
                   ("cassava", 120), ("gari", 120), ("sorghum", 240), ("millet", 240), ("beans", 240), ("rice", 300), ("maize", 180), ("oil", 180)]
DEFAULT_SHELF_DAYS = 90
STORAGE_COST_PER_MONTH = {"good": 0.01, "fair": 0.02, "poor": 0.04}
STORAGE_LABEL = {"good": "dry, sealed", "fair": "basic bags", "poor": "open or damp"}
WEATHER_LABEL = {"dry": "dry and stable", "mixed": "mixed", "wet": "heavy rain / flood risk"}
H = 3
COLORS = {"SELL": ("#FDF0E7", "#C2410C", "#7C2D12"), "STORE": ("#EAF2FB", "#1D6FB8", "#12417A"),
          "BUY": ("#EAF7EE", "#15803D", "#14532D"), "WAIT": ("#FFF8E6", "#B7791F", "#7A5900")}
_TRIGGERS = r"\b(sell|selling|store|storing|storage|hold|holding|buy|buying|purchase|stock up|harvest|harvested|when to)\b"


def is_guidance_query(text: str) -> bool:
    t = text.lower()
    return bool(re.search(_TRIGGERS, t)) or "should i" in t


def shelf_life_days(name: str) -> int:
    n = name.lower()
    return next((d for k, d in SHELF_LIFE_DAYS if k in n), DEFAULT_SHELF_DAYS)


def _pct(x):
    return f"{abs(x) * 100:.0f}%"


def _clip(x):
    return max(-1.0, min(1.0, x))


def analyse(crop, monthly, forecast_fn, role, storage, weather):
    """Core signal logic shared by the visual view and the chat."""
    fc = forecast_fn(monthly, periods=H)
    hist, fut = fc.history_values, fc.forecast_mean
    last = hist[-1]
    avg12 = sum(hist[-12:]) / len(hist[-12:])
    pos = (last - avg12) / avg12
    past = hist[-4] if len(hist) >= 4 else hist[0]
    trend = (last - past) / past
    outlook = (fut[H - 1] - last) / last
    lo, hi = fc.forecast_lower[H - 1], fc.forecast_upper[H - 1]
    shelf = shelf_life_days(crop)
    carry = STORAGE_COST_PER_MONTH[storage] * H
    pts = (3 if shelf < 30 else 0) + {"poor": 2, "fair": 1, "good": 0}[storage] \
        + {"wet": 2, "mixed": 1, "dry": 0}[weather] + (1 if shelf < 150 else 0)
    risk = "high" if pts >= 5 else "medium" if pts >= 3 else "low"

    if role == "grow":
        left, right = "Sell now", "Store / hold"
        if shelf < 30 or risk == "high":
            action, why = "SELL", "Spoilage or flood risk is high, so waiting could cost you the crop. Sell now."
        elif outlook > carry:
            action, why = "STORE", f"Prices are forecast to rise about {_pct(outlook)} in {H} months, more than the storage cost (about {_pct(carry)})."
        elif pos >= 0.10:
            action, why = "SELL", f"Today's price is {_pct(pos)} above its 12-month average and no further rise is forecast. Good time to cash in."
        elif pos <= -0.05 and risk == "low":
            action, why = "STORE", "Price is below its 12-month average and your storage is safe. Hold and wait for recovery."
        else:
            action, why = "SELL", "Waiting is unlikely to pay after storage costs. Sell most of it now."
    else:
        left, right = "Wait", "Buy now"
        if outlook >= max(carry, 0.05) and shelf >= 30:
            action, why = "BUY", f"Prices are forecast to rise about {_pct(outlook)}. Stock up now if you can store it safely."
        elif outlook <= -0.03:
            action, why = "WAIT", f"Prices are forecast to fall about {_pct(outlook)}. Buy only what you need today."
        elif pos <= -0.07:
            action, why = "BUY", f"Price is {_pct(pos)} below its 12-month average. A good time to buy."
        elif pos >= 0.10:
            action, why = "WAIT", "Price is well above normal. Buy only essentials and look for substitutes."
        else:
            action, why = "BUY", "Price is near normal. Buy your usual amount."

    factors = [("Price vs 12-month avg", _clip(-pos / 0.2)), ("Last 3 months' trend", _clip(trend / 0.15)),
               ("Forecast vs storage cost" if role == "grow" else "Forecast change",
                _clip((outlook - (carry if role == "grow" else 0)) / 0.1)),
               ("Spoilage risk", {"low": 0.5, "medium": 0.0, "high": -1.0}[risk])]
    d = fc.diagnostics
    obs, interp = d.get("observed_months", 0), d.get("interpolated_months_shown_in_chart", 0)
    share = obs / (obs + interp) if (obs + interp) else 0
    conf = "Low" if share < 0.30 or (hi - lo) / last > 0.30 else "Moderate"

    warns = []
    if trend <= -0.08: warns.append(("md", f"PRICE DROP: down {_pct(trend)} over the last 3 months."))
    if trend >= 0.08: warns.append(("md", f"PRICE RISE: up {_pct(trend)} over the last 3 months."))
    if outlook <= -0.05: warns.append(("hi", f"FORECAST FALL: about -{_pct(outlook)} expected over {H} months."))
    if weather == "wet": warns.append(("hi", "FLOOD / DAMP: keep stock off the floor and dry."))
    if shelf < 30: warns.append(("hi", f"PERISHABLE: keeps only about {shelf} days. Do not plan to store it."))
    if storage == "poor": warns.append(("md", "STORAGE: open or damp storage can destroy much of a grain or bean harvest. Sealed bags help."))
    if not warns: warns.append(("lo", "ALL CLEAR: no major risks detected right now."))
    return dict(fc=fc, last=last, avg12=avg12, pos=pos, trend=trend, outlook=outlook, lo=lo, hi=hi,
                fut=fut[H - 1], shelf=shelf, carry=carry, risk=risk, action=action, why=why,
                left=left, right=right, factors=factors, conf=conf, share=share, warns=warns)


def _factor_fig(a):
    f = a["factors"]
    fig = go.Figure(go.Bar(x=[v for _, v in f], y=[k for k, _ in f], orientation="h",
                           marker_color=["#128C7E" if v >= 0 else "#E24B4A" for _, v in f]))
    fig.update_layout(height=230, margin=dict(t=10, b=40, l=10, r=10), template="plotly_white",
                      xaxis=dict(range=[-1, 1], title=f"<- pushes toward {a['left']}      pushes toward {a['right']} ->"),
                      yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def _parse_context(text):
    t = text.lower()
    buy = re.search(r"\b(buy|buying|purchase|stock up)\b", t)
    grow = re.search(r"\b(sell|selling|store|storing|hold|holding|harvest|harvested)\b", t)
    storage = "poor" if re.search(r"damp|open|leak|outdoor|poor storage", t) else \
        "good" if re.search(r"sealed|hermetic|silo|warehouse|good storage", t) else None
    weather = "wet" if re.search(r"flood|heavy rain|raining|rainy|\bwet\b", t) else \
        "dry" if re.search(r"dry season|harmattan|dry weather", t) else None
    return ("buy" if buy and not grow else "grow"), storage, weather


def reply_guidance(user_text, prices, forecast_fn, match_fn, age_fn):
    """Chat reply: (text, figure)."""
    crop = match_fn(user_text)
    if crop is None:
        return ("I can advise on whether to **sell, store or buy** -- which food are you asking about? "
                f"For example: {', '.join(list(prices)[:8])}.\n\nYou can add your situation, e.g. "
                "\"should I sell my maize, my storage is damp\" or \"when to buy rice\".", None)
    role, s_in, w_in = _parse_context(user_text)
    storage, weather = s_in or "fair", w_in or "mixed"
    a = analyse(crop, prices[crop], forecast_fn, role, storage, weather)
    age_msg, stale = age_fn(max(prices[crop].keys()))
    lines = [f"**{a['action']} -- {crop}**", a["why"], "", "**Why I'm saying this**",
             f"- Price now: N{a['last']:,.0f} vs 12-month average N{a['avg12']:,.0f} ({a['pos'] * 100:+.0f}%)",
             f"- Last 3 months: {a['trend'] * 100:+.0f}%",
             f"- Next {H} months: about N{a['fut']:,.0f} (range N{a['lo']:,.0f} to N{a['hi']:,.0f})",
             f"- Spoilage risk: {a['risk']} ({a['shelf']} day shelf life, {STORAGE_LABEL[storage]} storage, {WEATHER_LABEL[weather]} weather)",
             "", "**Early warnings**"] + [f"- {t}" for _, t in a["warns"]] + [
             "", f"**How sure am I?** {a['conf']}. Only about {a['share'] * 100:.0f}% of the months behind this price are genuine NBS readings.",
             "", f"**What I assumed:** {STORAGE_LABEL[storage]} storage, {WEATHER_LABEL[weather]} weather"
             + ("" if (s_in or w_in) else " (you didn't say)") + ". Tell me, e.g. \"my storage is damp\", and I'll redo it."]
    if stale:
        lines += ["", f"Note: {age_msg}"]
    lines += ["", "This is guidance, not a guarantee."]
    return "\n".join(lines), _factor_fig(a)


_CSS = """<style>
.hs-card{border-radius:16px;padding:20px 24px;margin:10px 0 14px;border-left:14px solid}
.hs-verb{font-size:2.8rem;font-weight:800;line-height:1.05}
.hs-why{font-size:1.1rem;margin-top:8px}
.hs-chip{display:inline-block;font-size:.75rem;padding:2px 10px;border-radius:12px;background:#ffffff99;margin-top:10px}
.hs-risk{display:flex;gap:6px;margin:4px 0 12px}
.hs-risk div{flex:1;text-align:center;padding:8px;border-radius:8px;background:#eceff1;color:#78909c;font-size:.85rem}
.hs-mini{border-radius:10px;padding:10px 12px;border-left:6px solid;font-size:.85rem;line-height:1.35}
</style>"""


def render_simple_view(st, go, banner, render_chart, prices, forecast_fn, food_inflation, age_fn):
    """Visual early-warning dashboard for the simple view."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("## Should you sell, store or buy today?")
    st.markdown('<span class="source-badge">SDG 1: No Poverty</span><span class="source-badge">SDG 2: Zero Hunger</span>',
                unsafe_allow_html=True)
    st.caption("Early-warning guidance that turns price data into one clear signal, to protect your income and your family's food.")

    c1, c2, c3 = st.columns([1.2, 1, 1])
    role = "grow" if c1.radio("I am...", ["Selling my harvest", "Buying food"], key="hs_role").startswith("Selling") else "buy"
    crop = c2.selectbox("Food / crop", list(prices), key="hs_crop")
    qty = c3.number_input("How many kg?", min_value=1, value=100, step=10, key="hs_qty")
    with st.expander("My storage and weather (optional)"):
        s1, s2 = st.columns(2)
        storage = s1.selectbox("My storage", list(STORAGE_LABEL), index=1, format_func=STORAGE_LABEL.get, key="hs_st")
        weather = s2.selectbox("Weather outlook", list(WEATHER_LABEL), index=1, format_func=WEATHER_LABEL.get, key="hs_wx")

    a = analyse(crop, prices[crop], forecast_fn, role, storage, weather)
    bg, edge, ink = COLORS[a["action"]]
    st.markdown(f'<div class="hs-card" style="background:{bg};border-color:{edge};color:{ink}">'
                f'<div class="hs-verb" style="color:{edge}">{a["action"]}</div><div class="hs-why">{a["why"]}</div>'
                f'<span class="hs-chip">Confidence: {a["conf"]}</span></div>', unsafe_allow_html=True)

    g, ch = st.columns([1, 2])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=a["pos"] * 100, number=dict(suffix="%", valueformat="+.0f"),
        title=dict(text="Price today vs normal"),
        gauge=dict(axis=dict(range=[-30, 30]), bar=dict(color="#0a3d62"),
                   steps=[dict(range=[-30, -7], color="#BFE5C8"), dict(range=[-7, 10], color="#F5E6B0"),
                          dict(range=[10, 30], color="#F6C9B0")])))
    gauge.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    g.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False}, key="hs_gauge")
    g.caption("Green = cheap, yellow = normal, orange = expensive")

    fc = a["fc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.history_index[-12:], y=fc.history_values[-12:], mode="lines",
                             line=dict(color="#0a3d62", width=3), name="Past prices"))
    fig.add_trace(go.Scatter(x=fc.forecast_index + fc.forecast_index[::-1], y=fc.forecast_upper + fc.forecast_lower[::-1],
                             fill="toself", fillcolor="rgba(18,140,126,0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[fc.history_index[-1]] + fc.forecast_index, y=[fc.history_values[-1]] + fc.forecast_mean,
                             mode="lines+markers", line=dict(color="#128C7E", width=3, dash="dash"), name="Expected"))
    fig.add_hline(y=a["avg12"], line_dash="dot", line_color="#888888", annotation_text="normal price")
    fig.update_layout(height=260, margin=dict(t=30, b=10, l=10, r=10), template="plotly_white", yaxis_title="Naira",
                      title=dict(text="Where prices are heading", font=dict(size=14)), legend=dict(orientation="h", y=-0.2))
    with ch:
        render_chart(fig, key="hs_outlook")

    lv = ["low", "medium", "high"]
    cols = {"low": "#15803D", "medium": "#B7791F", "high": "#C2410C"}
    st.markdown("**Spoilage and loss risk**")
    st.markdown('<div class="hs-risk">' + "".join(
        f'<div style="background:{cols[k]};color:#fff;font-weight:700">{k.title()}</div>' if k == a["risk"]
        else f'<div>{k.title()}</div>' for k in lv) + "</div>", unsafe_allow_html=True)

    st.markdown("#### What this means for your money")
    now, later = qty * a["last"], qty * a["fut"]
    m1, m2, m3 = st.columns(3)
    if role == "grow":
        net = later - now * a["carry"]
        m1.metric(f"Sell {qty} kg today", f"N{now:,.0f}")
        m2.metric(f"Store ~{H} months (expected)", f"N{net:,.0f}", f"{net - now:+,.0f} vs today")
        m3.metric("Possible loss if it spoils", f"up to N{now:,.0f}", delta_color="off")
    else:
        m1.metric(f"Buy {qty} kg today", f"N{now:,.0f}")
        m2.metric(f"Same food in ~{H} months", f"N{later:,.0f}", f"{later - now:+,.0f} vs today", delta_color="inverse")
        m3.metric("National food inflation", f"{food_inflation[max(food_inflation)]}%", "yearly", delta_color="off")
    st.caption("Assumes NBS prices are per kg. Expected values are estimates, not promises.")

    st.markdown("#### Early warnings")
    for lvl, text in a["warns"]:
        banner("warning" if lvl in ("hi", "md") else "success", text)

    with st.expander("Why this advice? (the reasoning, explained)"):
        render_chart(_factor_fig(a), key="hs_xai")
        st.write(f"Price now N{a['last']:,.0f} vs normal N{a['avg12']:,.0f} ({a['pos'] * 100:+.0f}%). "
                 f"Next {H} months: about N{a['fut']:,.0f} (range N{a['lo']:,.0f} to N{a['hi']:,.0f}). "
                 f"Storage assumed to cost about {_pct(a['carry'])} over {H} months. Shelf life about {a['shelf']} days. "
                 "Green bars push toward the right-hand action, red toward the left.")

    if st.toggle("Show all foods at a glance", key="hs_watch"):
        key = ("watch", role, storage, weather)
        if st.session_state.get("hs_watch_key") != key:
            st.session_state["hs_watch_rows"] = [(n, analyse(n, m, forecast_fn, role, storage, weather)) for n, m in prices.items()]
            st.session_state["hs_watch_key"] = key
        rows = st.session_state["hs_watch_rows"]
        for i in range(0, len(rows), 3):
            for col, (n, r) in zip(st.columns(3), rows[i:i + 3]):
                b, e, k = COLORS[r["action"]]
                col.markdown(f'<div class="hs-mini" style="background:{b};border-color:{e};color:{k}"><b>{n}</b><br>'
                             f'<b style="color:{e}">{r["action"]}</b> &middot; N{r["last"]:,.0f} &middot; {r["outlook"] * 100:+.0f}% in {H} mo</div>',
                             unsafe_allow_html=True)

    age_msg, stale = age_fn(max(prices[crop].keys()))
    banner("warning" if stale else "info",
           f"{age_msg} Only about {a['share'] * 100:.0f}% of the months behind this price are genuine NBS readings; the rest are interpolated. "
           "Use this as a planning guide and check local market prices.")
    st.divider()
    st.markdown("#### Prefer to type? Ask the assistant")