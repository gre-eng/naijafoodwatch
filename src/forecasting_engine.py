"""
forecasting_engine.py
======================
International-standard time series forecasting methods applied to
Nigeria food price / food inflation data.

CHANGELOG (this revision)
--------------------------
Two gaps were raised: food prices visibly jump in festive periods
(Christmas/New Year) and around political/policy events (elections, a new
president, new policies), but the forecasts did not reflect either.

1. Festive window (commodity_monthly_forecast)
   Two separate mechanisms, deliberately kept apart:

   a) ESTIMATED festive multiplier (data-driven, usually withheld).
      For each genuine Nov/Dec/Jan NBS reading, price is compared with the
      straight line between the genuine readings on either side of it, so a
      rising price TREND is not mistaken for a SEASONAL effect (comparing
      December with that year's average, as a naive approach would, makes
      every December look "high" in an inflationary year because December
      is the last month of the year). A multiplier is applied only when
      MIN_FESTIVE_OBSERVATIONS genuine readings exist for that month AND
      they all point the same direction. With NBS data this sparse that
      will often not be met -- in that case nothing is applied and the
      diagnostics say why.

   b) ASSUMED festive uplift (scenario input, always disclosed).
      `festive_uplift_pct` lets the user state, as an explicit scenario
      assumption, that Nov/Dec/Jan prices run X% above the model path --
      the same way the app already lets users set an assumed exchange rate
      or oil price. It is labelled as an assumption, never as an estimate.
      Default is FESTIVE_UPLIFT_ASSUMPTION_PCT (0.0 = off).

2. Election / policy-event dummy (ols_macro_forecast)
   Optional `shock_years` regressor (history) and `future_shock_years`
   (forecast horizon). The caller decides which years to flag. Years fixed
   in advance by the calendar (general elections) are preferable to years
   picked afterwards because inflation was high -- the latter makes the
   dummy partly fit the outcome by construction and inflates its
   coefficient. With no shock years supplied, behaviour is identical to
   the previous version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dataclasses import dataclass


@dataclass
class ForecastResult:
    method: str
    history_index: list
    history_values: list
    forecast_index: list
    forecast_mean: list
    forecast_lower: list
    forecast_upper: list
    diagnostics: dict


def _to_series(data: dict[int, float]) -> pd.Series:
    s = pd.Series(data).sort_index()
    s.index = pd.PeriodIndex(s.index, freq="Y").to_timestamp()
    return s


def holt_winters_forecast(
    data: dict[int, float],
    periods: int = 3,
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
) -> ForecastResult:
    series = _to_series(data)
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=True,
    ).fit(optimized=True)

    resid_std = float(np.std(model.resid))

    forecast_mean = model.forecast(periods)
    z = 1.2816
    lower = forecast_mean - z * resid_std * np.sqrt(np.arange(1, periods + 1))
    upper = forecast_mean + z * resid_std * np.sqrt(np.arange(1, periods + 1))

    last_year = series.index[-1].year
    forecast_years = [last_year + i for i in range(1, periods + 1)]

    return ForecastResult(
        method="Holt-Winters Exponential Smoothing (damped trend)",
        history_index=[d.year for d in series.index],
        history_values=series.values.round(2).tolist(),
        forecast_index=forecast_years,
        forecast_mean=forecast_mean.round(2).tolist(),
        forecast_lower=lower.round(2).tolist(),
        forecast_upper=upper.round(2).tolist(),
        diagnostics={
            "AIC": round(float(model.aic), 2),
            "SSE": round(float(model.sse), 2),
            "smoothing_level (alpha)": round(float(model.params["smoothing_level"]), 4),
            "smoothing_trend (beta)": round(float(model.params["smoothing_trend"]), 4),
        },
    )


def sarima_forecast(
    data: dict[int, float],
    periods: int = 3,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (0, 0, 0, 0),
) -> ForecastResult:
    series = _to_series(data)

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    pred = model.get_forecast(steps=periods)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.20)

    last_year = series.index[-1].year
    forecast_years = [last_year + i for i in range(1, periods + 1)]

    return ForecastResult(
        method=f"SARIMA{order}",
        history_index=[d.year for d in series.index],
        history_values=series.values.round(2).tolist(),
        forecast_index=forecast_years,
        forecast_mean=mean.round(2).tolist(),
        forecast_lower=ci.iloc[:, 0].round(2).tolist(),
        forecast_upper=ci.iloc[:, 1].round(2).tolist(),
        diagnostics={
            "AIC": round(float(model.aic), 2),
            "BIC": round(float(model.bic), 2),
            "Log-likelihood": round(float(model.llf), 2),
        },
    )


def ols_macro_forecast(
    food_inflation: dict[int, float],
    exchange_rate: dict[int, float],
    oil_price: dict[int, float],
    periods: int = 3,
    future_exr: list[float] | None = None,
    future_oil: list[float] | None = None,
    shock_years: set[int] | None = None,
    future_shock_years: set[int] | None = None,
) -> ForecastResult:
    """
    OLS: food inflation ~ FX pass-through + oil + AR1 (+ optional event dummy).

    shock_years: historical calendar years to flag (e.g. general-election
        years). Prefer years fixed by the calendar over years chosen
        because inflation was high -- the latter is selection on the
        outcome and inflates the dummy's coefficient.
    future_shock_years: forecast-horizon years the caller flags (e.g. a
        known upcoming election year). The model cannot know this itself.
    If fewer than 2 flagged years fall inside the fitted sample, the dummy
    is left out and the model is exactly the original 3-regressor model.
    """
    years = sorted(food_inflation.keys())
    df = pd.DataFrame({
        "year": years,
        "food_infl": [food_inflation[y] for y in years],
        "exr": [exchange_rate[y] for y in years],
        "oil": [oil_price[y] for y in years],
    })
    df["exr_chg_pct"] = df["exr"].pct_change() * 100
    lo, hi = df["exr_chg_pct"].quantile([0.05, 0.95])
    df["exr_chg_pct"] = df["exr_chg_pct"].clip(lower=lo, upper=hi)
    df["food_infl_lag1"] = df["food_infl"].shift(1)

    shock_years = set(shock_years or set())
    df["shock_year"] = df["year"].isin(shock_years).astype(int)
    df_model = df.dropna()

    n_flagged = int(df_model["shock_year"].sum())
    use_shock_dummy = n_flagged >= 2

    feature_cols = ["exr_chg_pct", "oil", "food_infl_lag1"]
    if use_shock_dummy:
        feature_cols.append("shock_year")

    X = sm.add_constant(df_model[feature_cols])
    y = df_model["food_infl"]
    model = sm.OLS(y, X).fit()

    last_year = years[-1]
    last_exr = exchange_rate[last_year]
    last_oil = oil_price[last_year]
    last_food_infl = food_inflation[last_year]

    if future_exr is None:
        future_exr = [last_exr] * periods
    if future_oil is None:
        future_oil = [last_oil] * periods
    future_shock_years = set(future_shock_years or set())

    forecast_mean, forecast_lower, forecast_upper = [], [], []
    prev_exr = last_exr
    prev_food_infl = last_food_infl

    se = float(np.sqrt(model.mse_resid))
    z = 1.2816
    forecast_years = [last_year + i for i in range(1, periods + 1)]

    for i, fyear in enumerate(forecast_years):
        exr_chg = (future_exr[i] - prev_exr) / prev_exr * 100
        row = {
            "const": [1.0],
            "exr_chg_pct": [exr_chg],
            "oil": [future_oil[i]],
            "food_infl_lag1": [prev_food_infl],
        }
        if use_shock_dummy:
            row["shock_year"] = [1.0 if fyear in future_shock_years else 0.0]
        pred = float(model.predict(pd.DataFrame(row))[0])
        forecast_mean.append(round(pred, 2))
        spread = z * se * np.sqrt(i + 1)
        forecast_lower.append(round(pred - spread, 2))
        forecast_upper.append(round(pred + spread, 2))
        prev_exr = future_exr[i]
        prev_food_infl = pred

    diagnostics = {
        "R-squared": round(float(model.rsquared), 4),
        "Adj. R-squared": round(float(model.rsquared_adj), 4),
        "F-statistic p-value": round(float(model.f_pvalue), 5),
        "coef_exr_pass_through": round(float(model.params["exr_chg_pct"]), 4),
        "coef_oil": round(float(model.params["oil"]), 4),
        "coef_AR1": round(float(model.params["food_infl_lag1"]), 4),
        "scenario_assumption": (
            "Held exchange rate and oil price constant at last observed value"
            if future_exr == [last_exr] * periods
            else "Custom forward scenario supplied"
        ),
    }
    if use_shock_dummy:
        diagnostics["coef_event_year"] = round(float(model.params["shock_year"]), 4)
        diagnostics["event_year_p_value"] = round(float(model.pvalues["shock_year"]), 4)
        diagnostics["event_dummy_basis"] = (
            f"{n_flagged} flagged historical year(s) in the fitted sample. "
            "coef_event_year is the estimated extra food inflation (percentage "
            "points) in those years, holding FX/oil/AR1 constant. Check its "
            "p-value before trusting it -- with this few flagged years it is "
            "often not statistically distinguishable from zero."
        )
        diagnostics["event_years_assumed_in_forecast"] = (
            sorted(future_shock_years) if future_shock_years
            else "None flagged for the forecast horizon"
        )
    else:
        diagnostics["event_dummy_basis"] = (
            f"Not used -- only {n_flagged} flagged year(s) in the fitted "
            "sample (need 2+)."
        )

    return ForecastResult(
        method=(
            "OLS multivariate regression (FX pass-through + oil + AR1"
            + (" + event-year dummy)" if use_shock_dummy else ")")
        ),
        history_index=df_model["year"].tolist(),
        history_values=df_model["food_infl"].round(2).tolist(),
        forecast_index=forecast_years,
        forecast_mean=forecast_mean,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        diagnostics=diagnostics,
    )


def model_comparison_table(
    food_inflation: dict[int, float],
    exchange_rate: dict[int, float],
    oil_price: dict[int, float],
    periods: int = 3,
    shock_years: set[int] | None = None,
    future_shock_years: set[int] | None = None,
) -> pd.DataFrame:
    hw = holt_winters_forecast(food_inflation, periods=periods)
    sarima = sarima_forecast(food_inflation, periods=periods)
    ols = ols_macro_forecast(
        food_inflation, exchange_rate, oil_price, periods=periods,
        shock_years=shock_years, future_shock_years=future_shock_years,
    )

    rows = []
    for year_i in range(periods):
        rows.append({
            "Year": hw.forecast_index[year_i],
            "Holt-Winters": hw.forecast_mean[year_i],
            "SARIMA(1,1,1)": sarima.forecast_mean[year_i],
            "OLS (macro drivers)": ols.forecast_mean[year_i],
            "Ensemble average": round(
                np.mean([
                    hw.forecast_mean[year_i],
                    sarima.forecast_mean[year_i],
                    ols.forecast_mean[year_i],
                ]), 2
            ),
        })
    return pd.DataFrame(rows)


def _prepare_monthly_series_raw(monthly_data: dict[str, float]) -> tuple[pd.Series, int]:
    """
    Builds a continuous monthly index spanning the first to last reported
    NBS date, WITHOUT filling gaps -- missing months are left as NaN.

    This is deliberate. Linearly interpolating gaps and then fitting a
    model on the result manufactures artificial autocorrelation: the model
    ends up "detecting" patterns (including false seasonal cycles) in
    straight lines it was handed, not in anything NBS actually observed.
    SARIMAX's Kalman filter natively marginalizes over missing
    observations, which is the statistically correct way to handle a
    sparse, irregularly-reported series -- the resulting uncertainty
    bands widen honestly where real data is thin, instead of looking
    artificially tight.
    """
    dated = {pd.Period(k, freq="M").to_timestamp(): v for k, v in monthly_data.items()}
    s = pd.Series(dated).sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s_full = s.reindex(full_index)
    n_observed = int(s_full.notna().sum())
    return s_full, n_observed


def _prepare_monthly_series_display(monthly_data: dict[str, float]) -> pd.Series:
    """Interpolated version used ONLY for drawing a continuous history
    line in charts -- never used to fit a model."""
    dated = {pd.Period(k, freq="M").to_timestamp(): v for k, v in monthly_data.items()}
    s = pd.Series(dated).sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full_index).interpolate(method="linear").ffill().bfill()


# Genuine readings required before a 12-month seasonal SARIMA term is fit.
MIN_OBSERVATIONS_FOR_SEASONAL = 36

# Genuine, detrendable readings required PER FESTIVE MONTH (Nov, Dec, Jan)
# before an estimated multiplier is applied, and they must all agree on
# direction. Two readings that happen to agree is not evidence of a
# seasonal pattern, so the bar is three.
MIN_FESTIVE_OBSERVATIONS = 3

# A festive reading is only detrended against its neighbours if they are
# no more than this many months apart; a wider gap says nothing about
# what the "normal" price at that moment should have been.
MAX_FLANK_GAP_MONTHS = 12

# Assumed festive uplift (%) applied to Nov/Dec/Jan forecast months when the
# caller does not pass one. 0.0 = off. This is a SCENARIO ASSUMPTION, not an
# estimate -- set it only if you are prepared to label it that way.
FESTIVE_UPLIFT_ASSUMPTION_PCT = 0.0


def _festive_multiplier(series_raw: pd.Series) -> tuple[dict[int, float], dict[int, int], int]:
    """
    Trend-robust estimate of Nov/Dec/Jan price multipliers from GENUINE
    readings only.

    Each festive reading is divided by the value on the straight line
    between the genuine readings immediately before and after it. A ratio
    above 1 means the price was above its own local trend in that month.

    Returns (multipliers, usable_counts, n_decembers):
      multipliers  {month: ratio} only for months with >= MIN_FESTIVE_OBSERVATIONS
                   usable readings that all lie on the same side of 1.0
      usable_counts {month: number of usable readings} for disclosure
      n_decembers  genuine December readings in the series
    """
    obs = series_raw.dropna()
    n_decembers = int((obs.index.month == 12).sum())
    ratios: dict[int, list[float]] = {11: [], 12: [], 1: []}

    idx = list(obs.index)
    for k, t in enumerate(idx):
        if t.month not in ratios or k == 0 or k == len(idx) - 1:
            continue
        t_prev, t_next = idx[k - 1], idx[k + 1]
        gap_months = (t_next.year - t_prev.year) * 12 + (t_next.month - t_prev.month)
        if gap_months > MAX_FLANK_GAP_MONTHS:
            continue
        frac = ((t.year - t_prev.year) * 12 + (t.month - t_prev.month)) / gap_months
        expected = obs.iloc[k - 1] + (obs.iloc[k + 1] - obs.iloc[k - 1]) * frac
        if expected > 0:
            ratios[t.month].append(float(obs.iloc[k] / expected))

    multipliers: dict[int, float] = {}
    counts = {m: len(v) for m, v in ratios.items()}
    for m, vals in ratios.items():
        if len(vals) < MIN_FESTIVE_OBSERVATIONS:
            continue
        if all(v > 1 for v in vals) or all(v < 1 for v in vals):
            multipliers[m] = round(float(np.mean(vals)), 4)
    return multipliers, counts, n_decembers


def commodity_monthly_forecast(
    monthly_data: dict[str, float],
    periods: int = 12,
    festive_uplift_pct: float | None = None,
) -> ForecastResult:
    """
    festive_uplift_pct: assumed % uplift for Nov/Dec/Jan forecast months,
        supplied as an explicit scenario. None -> FESTIVE_UPLIFT_ASSUMPTION_PCT.
        If > 0 it takes precedence over any data-estimated multiplier.
    """
    if festive_uplift_pct is None:
        festive_uplift_pct = FESTIVE_UPLIFT_ASSUMPTION_PCT

    series_raw, n_observed = _prepare_monthly_series_raw(monthly_data)
    series_display = _prepare_monthly_series_display(monthly_data)
    n_total = len(series_raw)
    use_seasonal = n_observed >= MIN_OBSERVATIONS_FOR_SEASONAL

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12) if use_seasonal else (0, 0, 0, 0)

    # enforce_stationarity/invertibility=True: with a sparse missing-data
    # series an unconstrained fit can wander to explosive roots and produce
    # nonsensical forecasts.
    model = SARIMAX(
        series_raw, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=True, enforce_invertibility=True,
    ).fit(disp=False)

    pred = model.get_forecast(steps=periods)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.20)

    est_mult, est_counts, n_decembers = _festive_multiplier(series_raw)

    if festive_uplift_pct and festive_uplift_pct > 0:
        factor_by_month = {m: 1 + festive_uplift_pct / 100.0 for m in (11, 12, 1)}
        festive_source = "assumed"
    elif est_mult:
        factor_by_month = dict(est_mult)
        festive_source = "estimated"
    else:
        factor_by_month = {}
        festive_source = "none"

    mean_adj = mean.copy()
    ci_adj = ci.copy()
    for i, dt in enumerate(mean.index):
        f = factor_by_month.get(dt.month)
        if f is None:
            continue
        mean_adj.iloc[i] = mean.iloc[i] * f
        ci_adj.iloc[i] = ci.iloc[i] * f

    if festive_source == "assumed":
        festive_text = (
            f"SCENARIO ASSUMPTION (not estimated from data): Nov/Dec/Jan "
            f"forecast months raised by {festive_uplift_pct:g}% above the "
            "model path."
        )
    elif festive_source == "estimated":
        festive_text = (
            f"Estimated from NBS data (detrended vs neighbouring readings): "
            f"{factor_by_month}. Usable readings per month: {est_counts}."
        )
    else:
        festive_text = (
            "None applied. Estimated festive effect withheld: usable readings "
            f"per month {est_counts} (need {MIN_FESTIVE_OBSERVATIONS}+ per "
            "month, all pointing the same way). You can still set an explicit "
            "festive uplift scenario."
        )

    forecast_index = [d.strftime("%Y-%m") for d in mean.index]
    history_index = [d.strftime("%Y-%m") for d in series_display.index]

    return ForecastResult(
        method=(
            f"Monthly SARIMA{order}"
            + (f"x{seasonal_order}" if use_seasonal else "")
            + (f" + festive adjustment ({festive_source})" if festive_source != "none" else "")
            + " (fit on genuine NBS readings only; gaps handled as missing data, not interpolated)"
        ),
        history_index=history_index,
        history_values=series_display.round(2).tolist(),
        forecast_index=forecast_index,
        forecast_mean=mean_adj.round(2).tolist(),
        forecast_lower=ci_adj.iloc[:, 0].round(2).tolist(),
        forecast_upper=ci_adj.iloc[:, 1].round(2).tolist(),
        diagnostics={
            "AIC": round(float(model.aic), 2),
            "BIC": round(float(model.bic), 2),
            "observed_months": n_observed,
            "interpolated_months_shown_in_chart": n_total - n_observed,
            "seasonal_component": (
                "12-month seasonal SARIMA term included "
                f"({n_observed} genuine readings available)"
                if use_seasonal else
                f"No seasonal SARIMA term -- only {n_observed} genuine NBS "
                f"readings available (need {MIN_OBSERVATIONS_FOR_SEASONAL}+ to "
                "reliably estimate a full yearly cycle)"
            ),
            "model_fit_basis": (
                f"Fit directly on {n_observed} genuine NBS monthly readings "
                "with gaps treated as missing data (Kalman filter), not on "
                "interpolated substitutes -- interpolated points are used "
                "only to draw a continuous history line in charts."
            ),
            "festive_adjustment": festive_text,
            "genuine_december_readings": n_decembers,
        },
    )