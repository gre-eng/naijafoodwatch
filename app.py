"""
app.py
======
NaijaFoodWatch Pro -- Nigeria Food Price Forecasting Engine
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))

from data.historical_data import (
    CPI_INFLATION as CPI_INFLATION_FALLBACK,
    FOOD_INFLATION,
    FOOD_INFLATION_MONTHLY_2026,
    EXCHANGE_RATE as EXCHANGE_RATE_FALLBACK,
    OIL_PRICE_BRENT as OIL_PRICE_BRENT_FALLBACK,
    FAO_FOOD_PRICE_INDEX,
    NBS_COMMODITY_PRICES,
    STRUCTURAL_EVENTS,
    RICE_PRICE_ESTIMATE_NOTICE,
    DATA_GAP_NOTICE,
)
from data.live_data import (
    get_cpi_inflation_live,
    get_exchange_rate_live,
    get_oil_price_live,
    freshness_note,
    data_age_note,
)
from src.forecasting_engine import (
    holt_winters_forecast,
    sarima_forecast,
    ols_macro_forecast,
    model_comparison_table,
    commodity_monthly_forecast,
)
from src.chat_view import render_chat_view
from src.harvest_signal import reply_guidance, is_guidance_query, render_simple_view

# Nigerian general-election years. These are fixed by the calendar in
# advance, so flagging them does not depend on how high inflation turned
# out to be (unlike picking "shock years" after the fact, which would make
# the OLS event dummy partly fit the answer by construction). The next
# general election is due in 2027.
ELECTION_YEARS: set[int] = {1999, 2003, 2007, 2011, 2015, 2019, 2023}
NEXT_ELECTION_YEAR = 2027


def banner(kind: str, text: str, container=None) -> None:
    """Plain colored message box with no icon (no checkmark, no emoji).
    `container` lets this render in the sidebar (st.sidebar) or main body (default)."""
    target = container if container is not None else st
    target.markdown(f'<div class="banner banner-{kind}">{text}</div>', unsafe_allow_html=True)


def render_chart(fig: go.Figure, use_container_width: bool = True, key: str | None = None) -> None:
    fig.update_layout(dragmode=False)
    st.plotly_chart(
        fig,
        use_container_width=use_container_width,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "staticPlot": False,
        },
        key=key,
    )


CPI_INFLATION, cpi_is_live = get_cpi_inflation_live(CPI_INFLATION_FALLBACK)
EXCHANGE_RATE, exr_is_live = get_exchange_rate_live(EXCHANGE_RATE_FALLBACK)
OIL_PRICE_BRENT, oil_is_live = get_oil_price_live(OIL_PRICE_BRENT_FALLBACK)

st.set_page_config(
    page_title="NaijaFoodWatch Pro | Nigeria Food Price Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .source-badge {
        display:inline-block; font-size:0.72rem; padding:2px 9px;
        border-radius:12px; background:#E7F0FE; color:#0C447C;
        border:1px solid #378ADD33; margin-right:6px; margin-bottom:4px;
    }
    .metric-note { font-size:0.75rem; color:#6b7280; }
    .disclaimer-box {
        background:#f8f9fb; border:1px solid #e5e7eb; border-radius:10px;
        padding:14px 16px; font-size:0.82rem; line-height:1.55; color:#374151;
    }
    .banner {
        border-radius:10px; padding:14px 16px; font-size:0.92rem;
        line-height:1.55; margin-bottom:10px;
    }
    .banner-success { background:#EAF7EE; border:1px solid #B7E4C2; color:#1B5E30; }
    .banner-info    { background:#EEF4FC; border:1px solid #BFD7F5; color:#1B3A66; }
    .banner-warning { background:#FFF8E6; border:1px solid #F5DE9E; color:#7A5900; }
    .wa-header {
        background:#075E54; color:#fff; padding:14px 18px;
        border-radius:10px 10px 0 0; display:flex; align-items:center;
        gap:12px; margin-bottom:0;
    }
    .wa-avatar {
        width:40px; height:40px; border-radius:50%; background:#128C7E;
        display:flex; align-items:center; justify-content:center;
        font-weight:600; font-size:1.1rem; color:#fff; flex-shrink:0;
    }
    .wa-title { font-weight:600; font-size:1.02rem; line-height:1.25; }
    .wa-subtitle { font-size:0.78rem; color:#d9f2ee; }
    .wa-chatpane {
        background:#ECE5DD; padding:18px 16px 8px 16px; border-radius:0 0 10px 10px;
        min-height:120px;
    }
    .wa-bubble-bot {
        background:#ffffff; color:#111827; padding:10px 13px; border-radius:8px;
        border-top-left-radius:2px; max-width:78%; margin:4px 0 10px 0;
        box-shadow:0 1px 1px rgba(0,0,0,0.08); font-size:0.92rem; line-height:1.5;
    }
    .wa-bubble-user {
        background:#DCF8C6; color:#111827; padding:10px 13px; border-radius:8px;
        border-top-right-radius:2px; max-width:78%; margin:4px 0 10px auto;
        box-shadow:0 1px 1px rgba(0,0,0,0.08); font-size:0.92rem; line-height:1.5;
        text-align:left;
    }
    .wa-time {
        font-size:0.68rem; color:#8b98a3; margin-top:3px; text-align:right;
    }
    .wa-row-bot { display:flex; justify-content:flex-start; }
    .wa-row-user { display:flex; justify-content:flex-end; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NaijaFoodWatch Pro")
st.caption(
    "Multi-source, multi-variable food price forecasting engine for Nigeria - "
    "1997-2026 - Built on Holt-Winters, SARIMA, and structural OLS regression"
)

st.markdown(
    '<span class="source-badge">World Bank / FRED CPI</span>'
    '<span class="source-badge">NBS Selected Food Price Watch</span>'
    '<span class="source-badge">CBN Exchange Rates</span>'
    '<span class="source-badge">US BLS/EIA Brent Crude</span>'
    '<span class="source-badge">FAO Food Price Index</span>',
    unsafe_allow_html=True,
)

_all_live = cpi_is_live and exr_is_live and oil_is_live
_any_live = cpi_is_live or exr_is_live or oil_is_live
_top_banner_text = (
    f"CPI inflation: {freshness_note(cpi_is_live, CPI_INFLATION)}. "
    f"Exchange rate: {freshness_note(exr_is_live, EXCHANGE_RATE)}. "
    f"Brent crude: {freshness_note(oil_is_live, OIL_PRICE_BRENT)}."
)
banner("success" if _all_live else ("info" if _any_live else "warning"), _top_banner_text)

if FOOD_INFLATION_MONTHLY_2026:
    _latest_food_month = max(FOOD_INFLATION_MONTHLY_2026.keys())
    _food_age_msg, _food_is_stale = data_age_note(_latest_food_month)
    if _food_is_stale:
        banner("warning", f"Food inflation data: {_food_age_msg}")

st.divider()

MODE_ORDINARY = "Simple view"
MODE_ADVANCED = "Technical view"

view_mode = st.sidebar.radio("View mode", [MODE_ORDINARY, MODE_ADVANCED])
is_ordinary = view_mode == MODE_ORDINARY
is_advanced = view_mode == MODE_ADVANCED

st.sidebar.caption(
    {
        MODE_ORDINARY: (
            "A simple chat assistant for everyday budgeting -- ask it "
            "about food prices in plain language and it replies like "
            "a message, with the numbers behind the answer disclosed -- "
            "including explained sell / store / buy guidance."
        ),
        MODE_ADVANCED: (
            "Full model comparison, confidence intervals, and "
            "statistical diagnostics (AIC, BIC, R-squared) for "
            "analysts and researchers."
        ),
    }[view_mode]
)
st.sidebar.divider()

future_shock_years: set[int] = set()

if is_advanced:
    st.sidebar.header("Forecast configuration")
    series_choice = st.sidebar.selectbox(
        "Series to forecast",
        ["Food inflation (YoY %)", "Headline CPI inflation (YoY %)"],
    )
    horizon = st.sidebar.slider("Forecast horizon (years)", 1, 5, 3)
    models_selected = st.sidebar.multiselect(
        "Models to compare",
        ["Holt-Winters", "SARIMA", "OLS (macro drivers)"],
        default=["Holt-Winters", "SARIMA", "OLS (macro drivers)"],
    )
    st.sidebar.divider()
    st.sidebar.subheader("OLS scenario assumptions")
    st.sidebar.caption(
        "Forward exchange rate / oil price path used by the OLS model. "
        "Defaults hold the last observed value constant (no-further-shock baseline)."
    )
    custom_exr = st.sidebar.number_input(
        "Assumed NGN/USD (held constant across horizon)",
        min_value=100.0, max_value=5000.0, value=float(EXCHANGE_RATE[2026]), step=10.0,
    )
    custom_oil = st.sidebar.number_input(
        "Assumed Brent crude $/bbl (held constant across horizon)",
        min_value=10.0, max_value=300.0, value=float(OIL_PRICE_BRENT[2026]), step=1.0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Election / policy-year scenario")
    st.sidebar.caption(
        "The OLS model learns from past general-election years (1999-2023) "
        "via an event-year dummy. It cannot know a future year is an "
        "election or major-policy year unless you flag it here."
    )
    _last_data_year = max(FOOD_INFLATION.keys())
    _horizon_years = [_last_data_year + i for i in range(1, horizon + 1)]
    flag_future_event = st.sidebar.checkbox(
        "Flag an election / major-policy year in the forecast horizon",
        value=False,
    )
    if flag_future_event:
        _default_idx = _horizon_years.index(NEXT_ELECTION_YEAR) if NEXT_ELECTION_YEAR in _horizon_years else 0
        _chosen_event_year = st.sidebar.selectbox(
            "Which forecast year?", _horizon_years, index=_default_idx,
        )
        future_shock_years = {_chosen_event_year}

    st.sidebar.divider()
    st.sidebar.caption(
        "Data integrity: every figure in this app traces to a named, dated "
        "source. See the Sources & Methodology tab for full citations."
    )
    banner(
        "success",
        f"Active settings: forecasting {horizon} year(s) ahead using "
        f"{len(models_selected)} model(s). Switch to the Forecast "
        f"tab to see the result update.",
        container=st.sidebar,
    )
    series_data = FOOD_INFLATION if series_choice.startswith("Food") else CPI_INFLATION
    series_label = "Food inflation YoY (%)" if series_choice.startswith("Food") else "Headline CPI inflation YoY (%)"
    future_exr = [custom_exr] * horizon
    future_oil = [custom_oil] * horizon
else:
    horizon = 3
    custom_exr = float(EXCHANGE_RATE[2026])
    custom_oil = float(OIL_PRICE_BRENT[2026])
    st.sidebar.caption(
        "Just type a question in the chat -- for example \"food prices\", "
        "\"rice\", or \"why are prices rising\"."
    )

if is_ordinary:
    render_chat_view(
        NBS_COMMODITY_PRICES, commodity_monthly_forecast,
        FOOD_INFLATION_MONTHLY_2026, EXCHANGE_RATE, OIL_PRICE_BRENT,
    )

else:
    (
        tab_overview,
        tab_forecast,
        tab_models,
        tab_commodity,
        tab_commodity_forecast,
        tab_drivers,
        tab_sources,
    ) = st.tabs(
        [
            "Overview",
            "Forecast",
            "Model comparison",
            "Commodity prices",
            "Commodity forecast",
            "Macro drivers",
            "Sources & Methodology",
        ]
    )

    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)

        if FOOD_INFLATION_MONTHLY_2026:
            _latest_fi_month = max(FOOD_INFLATION_MONTHLY_2026.keys())
            _latest_fi_val = FOOD_INFLATION_MONTHLY_2026[_latest_fi_month]
            _jan_2026 = FOOD_INFLATION_MONTHLY_2026.get("2026-01")
            _fi_label = f"Food inflation ({_latest_fi_month})"
            _fi_delta = f"{_latest_fi_val - _jan_2026:+.1f}pp since Jan 2026" if _jan_2026 is not None else None
            col1.metric(_fi_label, f"{_latest_fi_val}%", _fi_delta)
        else:
            col1.metric("Food inflation", f"{FOOD_INFLATION[max(FOOD_INFLATION.keys())]}%")

        col2.metric("NGN/USD (2026)", f"N{EXCHANGE_RATE[2026]:,.0f}", "vs N21.9 in 1997")
        col3.metric("Brent crude (2026)", f"${OIL_PRICE_BRENT[2026]:.0f}/bbl")

        _avg_food_infl = np.mean(list(FOOD_INFLATION.values()))
        col4.metric(
            f"Historical avg food infl. ({min(FOOD_INFLATION)}-{max(FOOD_INFLATION)})",
            f"{_avg_food_infl:.2f}%",
        )

        st.markdown("#### Long-run food inflation, Nigeria (1997-2026)")
        df_hist = pd.DataFrame({
            "Year": list(FOOD_INFLATION.keys()),
            "Food inflation (%)": list(FOOD_INFLATION.values()),
            "Headline CPI (%)": [CPI_INFLATION[y] for y in FOOD_INFLATION.keys()],
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["Year"], y=df_hist["Food inflation (%)"],
            mode="lines+markers", name="Food inflation YoY %",
            line=dict(color="#E24B4A", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=df_hist["Year"], y=df_hist["Headline CPI (%)"],
            mode="lines", name="Headline CPI YoY %",
            line=dict(color="#378ADD", width=1.5, dash="dot"),
        ))
        for _ey in sorted(ELECTION_YEARS):
            if _ey in FOOD_INFLATION:
                fig.add_vline(x=_ey, line_width=1, line_dash="dot", line_color="#BBBBBB")
        fig.update_layout(
            height=400, margin=dict(t=20, b=20, l=10, r=10),
            yaxis_title="% YoY", xaxis_title="Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white",
            hovermode="x unified",
        )
        render_chart(fig)
        st.caption(
            "Source: NBS via Trading Economics (food inflation, 1996-2026); "
            "World Bank/FRED FPCPITOTLZGNGA (headline CPI, 1997-2024) + NBS (2025-2026). "
            "Faint vertical lines mark general-election years."
        )

        st.markdown("#### Structural breaks and major shocks")
        for ev in STRUCTURAL_EVENTS:
            with st.expander(f"{ev['year']} -- {ev['title']}"):
                st.write(ev["text"])
                st.caption(f"Source: {ev['source']}")

    with tab_forecast:
        years_hist = sorted(series_data.keys())
        vals_hist = [series_data[y] for y in years_hist]
        colors = {"Holt-Winters": "#E24B4A", "SARIMA": "#BA7517", "OLS (macro drivers)": "#1D9E75"}
        results_for_table = {}
        model_errors = []

        if "Holt-Winters" in models_selected:
            try:
                results_for_table["Holt-Winters"] = holt_winters_forecast(series_data, periods=horizon)
            except Exception as e:
                model_errors.append(f"Holt-Winters could not be fit: {e}")

        if "SARIMA" in models_selected:
            try:
                results_for_table["SARIMA"] = sarima_forecast(series_data, periods=horizon)
            except Exception as e:
                model_errors.append(f"SARIMA could not be fit: {e}")

        if "OLS (macro drivers)" in models_selected and series_choice.startswith("Food"):
            try:
                results_for_table["OLS (macro drivers)"] = ols_macro_forecast(
                    FOOD_INFLATION, EXCHANGE_RATE, OIL_PRICE_BRENT,
                    periods=horizon, future_exr=future_exr, future_oil=future_oil,
                    shock_years=ELECTION_YEARS, future_shock_years=future_shock_years,
                )
            except Exception as e:
                model_errors.append(f"OLS could not be fit: {e}")
        elif "OLS (macro drivers)" in models_selected:
            banner("info", "OLS macro-driver model is only available for the Food inflation series.")

        st.markdown("### What this means for Nigeria")
        if results_for_table:
            last_actual = vals_hist[-1]
            end_year = years_hist[-1] + horizon
            ensemble_end = round(
                np.mean([res.forecast_mean[-1] for res in results_for_table.values()]), 1
            )
            ensemble_lo = round(min(res.forecast_lower[-1] for res in results_for_table.values()), 1)
            ensemble_hi = round(max(res.forecast_upper[-1] for res in results_for_table.values()), 1)
            direction = "rise further to" if ensemble_end > last_actual else "ease to" if ensemble_end < last_actual else "stay near"

            fx_note = (
                f"This assumes the naira stays around N{custom_exr:,.0f} to the dollar "
                f"and Brent crude oil stays around ${custom_oil:,.0f} a barrel -- "
                "adjust these in the sidebar to test other scenarios."
            )
            event_note = ""
            if future_shock_years and "OLS (macro drivers)" in results_for_table:
                event_note = (
                    f" The OLS model treats {sorted(future_shock_years)[0]} as an "
                    "election / major-policy year (sidebar scenario); the other "
                    "two models do not use this flag."
                )
            impact_text = (
                f"Combining {len(results_for_table)} forecasting model(s), {series_label.lower()} "
                f"is projected to {direction} approximately {ensemble_end}% by {end_year} "
                f"(plausible range: {ensemble_lo}%-{ensemble_hi}%), compared to {last_actual}% "
                f"in {years_hist[-1]}. {fx_note}{event_note} Full diagnostics (AIC, BIC, "
                "R-squared, coefficients) are available in the Model comparison tab. If the "
                "low and high ends of the range are far apart, the underlying models "
                "disagree -- treat the estimate as a planning guide, not a precise "
                "prediction. These are annual figures and do not show within-year "
                "festive-season swings."
            )
            banner("success", impact_text)
            st.caption(
                "This statement is computed directly from the model outputs below, "
                "not a separate claim -- change the sidebar settings and it updates."
            )
        else:
            banner(
                "warning",
                "No model is currently selected, or all selected models failed to "
                "fit. Select at least one model under Models to compare in the "
                "sidebar to generate a forecast.",
            )

        for err in model_errors:
            banner("warning", err)

        st.divider()
        st.markdown(f"#### {series_label} -- supporting chart, forecast to {2026 + horizon}")
        banner(
            "info",
            "How to read this chart: the solid dark-blue line is real "
            "historical data (1997-2026). At 2026, each selected model "
            "branches off into its own dashed line -- that dashed segment "
            "is the prediction for years ahead. The shaded area around each "
            "dashed line is the 80% confidence band (the model's honest "
            "uncertainty range, not a guess presented as fact).",
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=years_hist, y=vals_hist, mode="lines+markers",
            name="Historical (actual data)", line=dict(color="#0a3d62", width=3),
            marker=dict(size=5),
        ))
        fig2.add_vline(
            x=years_hist[-1], line_width=1.5, line_dash="dot", line_color="#888888",
        )
        fig2.add_annotation(
            x=years_hist[-1], y=1.05, yref="paper", showarrow=False,
            text="History  |  Forecast", font=dict(size=11, color="#888888"),
        )

        if "Holt-Winters" in results_for_table:
            hw = results_for_table["Holt-Winters"]
            fig2.add_trace(go.Scatter(
                x=[years_hist[-1]] + hw.forecast_index, y=[vals_hist[-1]] + hw.forecast_mean,
                mode="lines+markers", name="Holt-Winters forecast",
                line=dict(color=colors["Holt-Winters"], width=2, dash="dash"),
            ))
            fig2.add_trace(go.Scatter(
                x=hw.forecast_index + hw.forecast_index[::-1],
                y=hw.forecast_upper + hw.forecast_lower[::-1],
                fill="toself", fillcolor="rgba(226,75,74,0.12)",
                line=dict(color="rgba(255,255,255,0)"), name="HW 80% CI", showlegend=False,
            ))

        if "SARIMA" in results_for_table:
            sar = results_for_table["SARIMA"]
            fig2.add_trace(go.Scatter(
                x=[years_hist[-1]] + sar.forecast_index, y=[vals_hist[-1]] + sar.forecast_mean,
                mode="lines+markers", name="SARIMA forecast",
                line=dict(color=colors["SARIMA"], width=2, dash="dash"),
            ))
            fig2.add_trace(go.Scatter(
                x=sar.forecast_index + sar.forecast_index[::-1],
                y=sar.forecast_upper + sar.forecast_lower[::-1],
                fill="toself", fillcolor="rgba(186,117,23,0.12)",
                line=dict(color="rgba(255,255,255,0)"), name="SARIMA 80% CI", showlegend=False,
            ))

        if "OLS (macro drivers)" in results_for_table:
            ols = results_for_table["OLS (macro drivers)"]
            fig2.add_trace(go.Scatter(
                x=[years_hist[-1]] + ols.forecast_index, y=[vals_hist[-1]] + ols.forecast_mean,
                mode="lines+markers", name="OLS forecast",
                line=dict(color=colors["OLS (macro drivers)"], width=2, dash="dash"),
            ))
            fig2.add_trace(go.Scatter(
                x=ols.forecast_index + ols.forecast_index[::-1],
                y=ols.forecast_upper + ols.forecast_lower[::-1],
                fill="toself", fillcolor="rgba(29,158,117,0.12)",
                line=dict(color="rgba(255,255,255,0)"), name="OLS 80% CI", showlegend=False,
            ))

        fig2.update_layout(
            height=460, margin=dict(t=20, b=20, l=10, r=10),
            yaxis_title="% YoY", xaxis_title="Year", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        for trace in fig2.data:
            if "CI" in (trace.name or ""):
                trace.hoverinfo = "skip"
                trace.hovertemplate = None
        render_chart(fig2)

        if results_for_table:
            st.markdown("#### Forecast values")
            table_rows = []
            for i in range(horizon):
                row = {"Year": list(results_for_table.values())[0].forecast_index[i]}
                for name, res in results_for_table.items():
                    row[f"{name} -- central"] = res.forecast_mean[i]
                    row[f"{name} -- 80% CI"] = f"[{res.forecast_lower[i]}, {res.forecast_upper[i]}]"
                table_rows.append(row)
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="disclaimer-box">Forecast values are model outputs '
            'with explicit uncertainty bands -- they are statistical '
            'projections conditioned on historical patterns and the stated '
            'scenario assumptions, not guarantees. Always read the confidence '
            'interval, not just the central estimate.</div>',
            unsafe_allow_html=True,
        )

    with tab_models:
        st.markdown("#### Side-by-side model diagnostics")
        st.caption(
            "Comparing model fit statistics helps decide which forecast to trust. "
            "Lower AIC/BIC indicates better fit relative to model complexity; "
            "higher R-squared indicates more explained variance (OLS only)."
        )

        diag_cols = st.columns(3)
        if "Holt-Winters" in results_for_table:
            with diag_cols[0]:
                st.markdown("Holt-Winters")
                for k, v in results_for_table["Holt-Winters"].diagnostics.items():
                    st.write(f"{k}: `{v}`")
        if "SARIMA" in results_for_table:
            with diag_cols[1]:
                st.markdown("SARIMA(1,1,1)")
                for k, v in results_for_table["SARIMA"].diagnostics.items():
                    st.write(f"{k}: `{v}`")
        if "OLS (macro drivers)" in results_for_table:
            with diag_cols[2]:
                st.markdown("OLS macro-driver model")
                for k, v in results_for_table["OLS (macro drivers)"].diagnostics.items():
                    st.write(f"{k}: `{v}`")

        st.divider()
        st.markdown("#### Ensemble forecast (average of all fitted models)")
        if series_choice.startswith("Food"):
            comp_df = model_comparison_table(
                FOOD_INFLATION, EXCHANGE_RATE, OIL_PRICE_BRENT, periods=horizon,
                shock_years=ELECTION_YEARS, future_shock_years=future_shock_years,
            )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.caption(
                "Ensemble averaging is a standard technique to reduce the risk of any "
                "single model's specification error dominating the forecast. Real "
                "institutions (e.g. FAO GIEWS, IMF WEO) routinely report ensemble or "
                "model-range forecasts rather than a single point estimate."
            )
        else:
            banner("info", "Ensemble comparison table is available for the Food inflation series.")

    with tab_commodity:
        st.markdown("#### NBS Selected Food Price Watch -- commodity prices (2010-2026)")
        commodity_choice = st.selectbox("Select commodity", list(NBS_COMMODITY_PRICES.keys()))
        series = NBS_COMMODITY_PRICES[commodity_choice]
        dates = sorted(series.keys())
        prices = [series[d] for d in dates]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines+markers",
            line=dict(color="#0a3d62", width=2),
            name=commodity_choice,
        ))
        fig3.update_layout(
            height=380, margin=dict(t=20, b=20, l=10, r=10),
            yaxis_title="N", xaxis_title="Month", template="plotly_white",
            hovermode="x unified",
        )
        render_chart(fig3)

        last_price = prices[-1]
        first_price = prices[0]
        total_change = (last_price / first_price - 1) * 100
        st.metric(
            f"{commodity_choice}: {dates[0]} -> {dates[-1]}",
            f"N{last_price:,.2f}",
            f"{total_change:+.1f}% cumulative",
        )
        st.caption("Source: NBS Selected Food Price Watch monthly bulletins, nigerianstat.gov.ng")

    with tab_commodity_forecast:
        st.markdown("#### Price forecast by commodity (monthly)")
        _latest_commodity_month = max(
            max(monthly.keys()) for monthly in NBS_COMMODITY_PRICES.values()
        )
        _commodity_age_msg, _commodity_is_stale = data_age_note(_latest_commodity_month)
        banner("warning" if _commodity_is_stale else "info", f"{DATA_GAP_NOTICE} {_commodity_age_msg}")

        good = st.selectbox("Commodity", list(NBS_COMMODITY_PRICES.keys()))
        months_ahead = st.slider("Forecast horizon (months)", 1, 24, 6, key="adv_horizon")
        festive_uplift = st.slider(
            "Assumed festive uplift for Nov / Dec / Jan (%) -- scenario assumption",
            0, 30, 0, key="adv_festive_uplift",
            help=(
                "Off at 0. Prices usually rise around Christmas/New Year, but the "
                "NBS history here is too sparse to estimate by how much for any "
                "commodity. Set a % here to see a what-if; it is applied as your "
                "stated assumption, not as something the data showed."
            ),
        )
        result = commodity_monthly_forecast(
            NBS_COMMODITY_PRICES[good], periods=months_ahead,
            festive_uplift_pct=float(festive_uplift),
        )

        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=result.history_index, y=result.history_values,
                                    mode="lines", name="Historical (interpolated monthly)",
                                    line=dict(color="#0a3d62", width=2)))
        fig_a.add_trace(go.Scatter(
            x=[result.history_index[-1]] + result.forecast_index,
            y=[result.history_values[-1]] + result.forecast_mean,
            mode="lines+markers", name="SARIMA forecast",
            line=dict(color="#E24B4A", width=2, dash="dash")))
        fig_a.add_trace(go.Scatter(
            x=result.forecast_index + result.forecast_index[::-1],
            y=result.forecast_upper + result.forecast_lower[::-1],
            fill="toself", fillcolor="rgba(226,75,74,0.12)",
            line=dict(color="rgba(255,255,255,0)"), name="80% CI", showlegend=False))
        fig_a.update_layout(height=420, margin=dict(t=20, b=20, l=10, r=10),
                             yaxis_title="N per unit", xaxis_title="Month",
                             template="plotly_white", hovermode="x unified")
        render_chart(fig_a)

        df_fc = pd.DataFrame({
            "Month": result.forecast_index,
            "Central estimate (N)": result.forecast_mean,
            "80% CI": [f"[{lo}, {hi}]" for lo, hi in zip(result.forecast_lower, result.forecast_upper)],
        })
        st.dataframe(df_fc, use_container_width=True, hide_index=True)

        st.markdown("#### Model diagnostics")
        for k, v in result.diagnostics.items():
            st.write(f"{k}: `{v}`")
        st.caption(
            "NBS publishes these prices irregularly (some months are missing "
            "from the bulletins). The model itself is fit only on genuine NBS "
            "readings, with gaps treated as missing data rather than filled "
            "in -- see 'model_fit_basis' above. The chart's history line uses "
            "linear interpolation across gaps purely for visual continuity; "
            "'interpolated_months_shown_in_chart' above discloses how many of "
            "the plotted points are interpolated rather than independently "
            "observed, so the forecast is never mistaken for more raw data "
            "than actually exists. 'festive_adjustment' says whether any "
            "Nov/Dec/Jan adjustment was applied and whether it was estimated "
            "from data or is your scenario assumption."
        )

    with tab_drivers:
        st.markdown("#### Macro drivers of Nigeria food prices")
        driver_choice = st.radio(
            "Driver series", ["Exchange rate (NGN/USD)", "Brent crude oil ($/bbl)", "FAO Food Price Index"],
            horizontal=True,
        )

        if driver_choice.startswith("Exchange"):
            years_d = sorted(EXCHANGE_RATE.keys())
            vals_d = [EXCHANGE_RATE[y] for y in years_d]
            ytitle, color = "N per USD", "#378ADD"
            src = "Source: Central Bank of Nigeria (CBN) Exchange Rate Policy report + NAFEM figures"
        elif driver_choice.startswith("Brent"):
            years_d = sorted(OIL_PRICE_BRENT.keys())
            vals_d = [OIL_PRICE_BRENT[y] for y in years_d]
            ytitle, color = "USD/barrel", "#BA7517"
            src = "Source: US Bureau of Labor Statistics / EIA historical Brent series"
        else:
            years_d = sorted(FAO_FOOD_PRICE_INDEX.keys())
            vals_d = [FAO_FOOD_PRICE_INDEX[y] for y in years_d]
            ytitle, color = "Index (2014-16=100)", "#1D9E75"
            src = "Source: FAO Food Price Index, Food and Agriculture Organization of the UN"

        fig4 = go.Figure()
        fillcolor_map = {
            "#378ADD": "rgba(55,138,221,0.12)",
            "#BA7517": "rgba(186,117,23,0.12)",
            "#1D9E75": "rgba(29,158,117,0.12)",
        }
        fig4.add_trace(go.Scatter(
            x=years_d, y=vals_d, mode="lines+markers",
            line=dict(color=color, width=2.2), fill="tozeroy",
            fillcolor=fillcolor_map.get(color, "rgba(55,138,221,0.12)"),
        ))
        fig4.update_layout(
            height=380, margin=dict(t=20, b=20, l=10, r=10),
            yaxis_title=ytitle, xaxis_title="Year", template="plotly_white",
            hovermode="x unified",
        )
        render_chart(fig4)
        st.caption(src)

        st.markdown("##### Correlation with Nigeria food inflation (1997-2026)")
        corr_df = pd.DataFrame({
            "year": sorted(FOOD_INFLATION.keys()),
            "food_infl": [FOOD_INFLATION[y] for y in sorted(FOOD_INFLATION.keys())],
            "exr": [EXCHANGE_RATE[y] for y in sorted(FOOD_INFLATION.keys())],
            "oil": [OIL_PRICE_BRENT[y] for y in sorted(FOOD_INFLATION.keys())],
        })
        corr_exr = corr_df["food_infl"].corr(corr_df["exr"])
        corr_oil = corr_df["food_infl"].corr(corr_df["oil"])
        c1, c2 = st.columns(2)
        c1.metric("Correlation: food inflation vs. exchange rate level", f"{corr_exr:.2f}")
        c2.metric("Correlation: food inflation vs. oil price level", f"{corr_oil:.2f}")
        st.caption(
            "Computed directly from the historical series in this app (Pearson "
            "correlation, levels, 1997-2026, n=30). Note correlation of levels "
            "with two non-stationary trending series can overstate the "
            "relationship; the OLS model in the Forecast tab uses first "
            "differences for the exchange rate to address this."
        )

    with tab_sources:
        st.markdown("#### Data sources")
        st.markdown(
            """
| Series | Source | Coverage | Update mode |
|---|---|---|---|
| Headline CPI inflation | World Bank Indicators API (`FP.CPI.TOTL.ZG`) | 1997-latest published year | Live -- auto-fetched every run, 12h cache |
| Exchange rate (NGN/USD) | World Bank Indicators API (`PA.NUS.FCRF`) | 1997-latest published year | Live -- auto-fetched every run, 12h cache |
| Brent crude oil price | FRED / US EIA (`DCOILBRENTEU`), daily | 1997-latest published day | Live -- auto-fetched every run, 12h cache |
| Food inflation YoY | NBS CPI reports, via Trading Economics; cross-checked with CEIC | 1997-2026 | Static fallback (no free public API exists for this series) -- staleness auto-detected on every load |
| FAO Food Price Index | FAO, Food and Agriculture Organization of the UN | 2000-2026 | Static fallback |
| Commodity-level prices (N/kg) | NBS Selected Food Price Watch (monthly bulletins) | 2010-2026 | Static fallback -- staleness auto-detected on every load |
            """
        )

        st.markdown("#### How the auto-update works")
        st.markdown(
            """
This app does not require you to open the code and edit numbers
when new data is published. Three series -- headline CPI inflation,
the exchange rate, and Brent crude oil price -- are fetched live
every time the app runs: CPI and the exchange rate from the World
Bank Indicators API (`api.worldbank.org`, free, no key required), and
Brent crude from FRED's public CSV endpoint (`fred.stlouisfed.org`,
also free, no key required). Both are cached for 12 hours so the app
doesn't over-call either API. The banner at the top of the page tells
you, on every visit, whether you're looking at a live-fetched figure
or a fallback static one -- and because oil is live, a market shock
that moves the oil price (a war, an OPEC decision, a shipping
disruption) flows into the OLS forecast automatically, with nobody
touching the code.

Why aren't all series live? Food inflation, the FAO index, and NBS
commodity-level prices currently have no free public REST API -- NBS
publishes CPI/food-inflation reports and the Selected Food Price
Watch as PDF/web bulletins, not a machine-readable feed, and the same
is true for the FAO index at no cost. This is a genuine, structural
limit, not a bug: there is no legitimate free API to point at. The
two real options if you want these live are (a) a paid data vendor
that has already licensed or scraped NBS's numbers, or (b) building a
web scraper against NBS's own site -- which this app deliberately
does NOT do, because NBS can restructure their page at any time and
silently break a scraper, and scraping may run against their terms
of use. Instead, those series use the verified static dataset in
`data/historical_data.py`, refreshed by a human periodically when NBS
releases a new bulletin -- and the app automatically checks today's
real date against the newest data point on every load, so it always
tells you truthfully how many months out of date that figure is,
rather than silently presenting an old price as current (see the
warning banners on the Commodity forecast tab, and above, if that
data has gone stale).

If NBS later opens a public API, or you find a paid/free data vendor
that mirrors it, `data/live_data.py` is built so you only need to add
one more `fetch_*` function there -- the rest of the app's plumbing
(caching, fallback, freshness banner, staleness check) already exists
and will pick it up automatically. No market event -- war, pandemic,
election, anything else -- can be detected by this app on its own;
it can only ever react to a number that some API actually publishes.
            """
        )

        banner("warning", RICE_PRICE_ESTIMATE_NOTICE)
        _latest_commodity_month_src = max(
            max(monthly.keys()) for monthly in NBS_COMMODITY_PRICES.values()
        )
        _commodity_age_msg_src, _commodity_is_stale_src = data_age_note(_latest_commodity_month_src)
        banner(
            "warning" if _commodity_is_stale_src else "info",
            f"{DATA_GAP_NOTICE} {_commodity_age_msg_src}",
        )

        st.markdown("#### Forecasting methodology")
        st.markdown(
            """
Holt-Winters Exponential Smoothing (damped trend). Triple exponential
smoothing fit via `statsmodels.tsa.holtwinters`. Appropriate for
short-to-medium horizon forecasts where recent trend matters more
than distant history. Confidence intervals derived from in-sample
residual standard deviation scaled by forecast horizon (root-h rule).

SARIMA. Classical Box-Jenkins ARIMA, fit via
`statsmodels.tsa.statespace.sarimax.SARIMAX`. Default order (1,1,1):
one autoregressive term, one order of differencing for stationarity,
one moving-average term. This is a parsimonious, defensible
specification for a ~30-observation annual series; over-parameterised
models would overfit given the sample size.

OLS multivariate regression (macro drivers). Food inflation regressed
on: (i) exchange rate depreciation (% YoY, winsorized at the
5th/95th percentile to limit the influence of one-off devaluation
shocks such as 1999 and 2023), (ii) Brent crude oil price level,
(iii) one-period lagged food inflation (AR1 / momentum term), and
(iv) an election-year dummy flagging Nigeria's general-election years
(1999, 2003, 2007, 2011, 2015, 2019, 2023). Election years are fixed
by the calendar in advance, so the dummy is not chosen after seeing
which years had high inflation. The Technical view sidebar lets you
flag an upcoming election or major-policy year in the forecast
horizon (the model cannot know this on its own). Check the dummy's
p-value in the Model comparison tab: with seven flagged years it is
often not statistically distinguishable from zero, and if so the
data do not show a reliable election-year effect on top of FX, oil
and momentum. This mirrors the structural decomposition central banks
use to separate FX pass-through, energy cost pass-through, and
inflation persistence.

Ensemble. The simple average of the three annual model forecasts is
also reported. Ensembling is standard practice in operational
forecasting (e.g. IMF World Economic Outlook ranges, FAO GIEWS)
because it reduces exposure to any single model's specification error.

Monthly commodity forecasts. NBS Selected Food Price Watch reports
commodity prices irregularly (some months are not published). The
model is fit with SARIMA on the genuine readings only, with gaps
treated as missing data; interpolated points are used only to draw
the history line. A 12-month seasonal SARIMA term is included only
when at least 36 genuine monthly readings exist, since a shorter
series cannot support a reliable seasonal estimate. Every commodity
forecast discloses how many months were genuinely observed versus
interpolated.

Festive-period (Nov/Dec/Jan) prices. Prices in Nigeria commonly rise
around Christmas and New Year, but this app can only show that effect
in two disclosed ways. (1) Estimated: each genuine Nov/Dec/Jan NBS
reading is compared with the straight line between the genuine
readings on either side of it (so a rising price trend is not
mistaken for a seasonal effect), and a multiplier is applied only if
at least three such readings exist for that month and all point the
same direction. With the current NBS history (about 24-25 genuine
readings per commodity, only two Decembers each) this is not met for
any commodity, so no estimated festive effect is applied. (2) Assumed:
the Commodity forecast tab has a festive-uplift slider (default 0%)
that applies a percentage you choose to Nov/Dec/Jan forecast months,
clearly labelled as your scenario assumption, not a finding from the
data. The annual models cannot represent any within-year pattern by
construction.

Known limitations (disclosed, not hidden):
- Annual frequency limits the food-inflation dataset to ~30
  observations; this is a genuine sample-size constraint for any
  econometric model and is why R-squared for the OLS model is in the
  0.2-0.4 range rather than higher -- that is the realistic ceiling
  for this kind of macro series, not a modelling failure.
- Pre-2010 commodity-level prices are not available from NBS at
  monthly granularity; only the food inflation rate and CPI exist at
  annual frequency for that period.
- Election-year and policy effects are estimated from a handful of
  years; they cannot capture a new government's specific policies
  (e.g. a subsidy removal) unless that policy lands in a flagged year,
  and the model cannot predict future policy.
- Festive effects are not estimated from data for any commodity until
  NBS history is dense enough (see above); this becomes automatically
  re-testable as more monthly bulletins are added to
  `data/historical_data.py`.
- No forecast in this app -- annual or monthly -- claims perfect
  accuracy. All forecasts carry an explicit confidence interval and
  are conditional on the stated assumptions; changing those
  assumptions (sidebar, or the commodity data itself as new NBS
  bulletins are added) will change the forecast.
            """
        )

        st.markdown("#### Data integrity statement")
        st.markdown(
            '<div class="disclaimer-box">Every numeric value in this '
            'application is sourced from a named, citable institution '
            '(World Bank, NBS, CBN, FAO, US BLS/EIA) or is explicitly '
            'labelled as an estimate, interpolation or user scenario '
            'assumption with its derivation method disclosed. No figure '
            'has been fabricated. Forecast values are clearly distinguished '
            'from historical values in every chart and table, and are '
            'always shown with a confidence interval rather than as a bare '
            'point estimate.</div>',
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "NaijaFoodWatch Pro - Built for academic and policy-analysis purposes - "
    "Not financial or trading advice - Source code: see GitHub repository"
)