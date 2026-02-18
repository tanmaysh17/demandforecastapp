"""Page 8 - Launch Curve Modeler: model demand uptake for new product launches."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Launch Curve | Demand Forecasting Studio", layout="wide")

from app.ui.shared import setup_page  # noqa: E402

setup_page()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from datetime import date  # noqa: E402

st.header("8) Launch Curve Modeler")
st.caption(
    "Model demand uptake for a new product launch — no historical data required. "
    "Define your market size, pick a curve shape, tune the parameters, and compare up to three scenarios."
)

# ─────────────────────────────────────────────────────────────────────────────
# Curve math
# ─────────────────────────────────────────────────────────────────────────────

def _logistic(t: np.ndarray, k: float, t0: float) -> np.ndarray:
    """Standard logistic. Returns values in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-k * (t - t0)))


def _gompertz(t: np.ndarray, b: float, c: float) -> np.ndarray:
    """Gompertz curve. Returns values in (0, 1)."""
    return np.exp(-b * np.exp(-c * t))


def _bass_cumulative(t: np.ndarray, p: float, q: float) -> np.ndarray:
    """Bass diffusion cumulative fraction in [0, 1]."""
    pq = p + q
    exp_term = np.exp(-pq * t)
    if q == 0:
        return 1.0 - exp_term
    return (1.0 - exp_term) / (1.0 + (q / p) * exp_term)


def _power(t: np.ndarray, alpha: float, T: float) -> np.ndarray:
    """Power ramp. alpha<1 = fast ramp; alpha>1 = slow build."""
    return np.clip(t / T, 0.0, 1.0) ** alpha


def _exp_saturation(t: np.ndarray, k: float) -> np.ndarray:
    """Exponential saturation. Returns values in (0, 1)."""
    return 1.0 - np.exp(-k * t)


def _normalize_to_peak(raw: np.ndarray, peak: float, horizon_weeks: int) -> np.ndarray:
    """
    Scale raw curve so it starts at 0 and equals peak at horizon_weeks.
    Beyond the horizon the curve is held at peak (steady state).
    """
    y = raw - raw[0]                          # start at 0
    anchor = y[horizon_weeks] if horizon_weeks < len(y) else y[-1]
    if anchor <= 0:
        anchor = y.max() if y.max() > 0 else 1.0
    y = np.clip(y / anchor * peak, 0.0, None)
    y[horizon_weeks:] = peak                  # flat steady-state tail
    return y


def build_curve(
    weeks: np.ndarray,
    peak: float,
    horizon_weeks: int,
    curve_type: str,
    params: dict,
) -> np.ndarray:
    t = weeks.astype(float)
    T = float(horizon_weeks)

    if curve_type == "S-Curve (Logistic)":
        k = params["steepness"] / T
        t0 = T * params["inflection_pct"] / 100.0
        raw = _logistic(t, k, t0)

    elif curve_type == "S-Curve (Gompertz)":
        c = params["growth_rate"] / T
        raw = _gompertz(t, params["displacement"], c)

    elif curve_type == "Bass Diffusion":
        # p and q are annual rates; scale to weekly
        p_w = params["p"] / 52.0
        q_w = params["q"] / 52.0
        raw = _bass_cumulative(t, p_w, q_w)

    elif curve_type == "Rapid Ramp (R-Curve)":
        k = params["ramp_speed"] / T
        raw = _exp_saturation(t, k)

    elif curve_type == "Slow Build":
        raw = _power(t, params["alpha"], T)

    else:  # Linear
        raw = _power(t, 1.0, T)

    return _normalize_to_peak(raw, peak, horizon_weeks)


# ─────────────────────────────────────────────────────────────────────────────
# Curve catalogue
# ─────────────────────────────────────────────────────────────────────────────
CURVES = {
    "S-Curve (Logistic)": (
        "Slow start → rapid growth at inflection point → plateau. "
        "The classic adoption pattern for mainstream consumer products."
    ),
    "S-Curve (Gompertz)": (
        "Asymmetric S-curve — faster early growth than Logistic, longer slow tail to plateau. "
        "Common for tech products and pharmaceuticals."
    ),
    "Bass Diffusion": (
        "Marketing-science standard. Separates innovators (p) who adopt independently "
        "from imitators (q) driven by word-of-mouth. The higher q relative to p, the more viral the launch."
    ),
    "Rapid Ramp (R-Curve)": (
        "Fast initial uptake that decelerates to a plateau — exponential saturation shape. "
        "Typical when strong launch marketing drives immediate awareness."
    ),
    "Slow Build": (
        "Gradual start that accelerates before levelling off (convex power curve). "
        "Typical for B2B, specialist, or distribution-constrained products."
    ),
    "Linear": (
        "Steady, constant weekly growth to peak. "
        "A simple baseline assumption when there is no evidence of a particular adoption pattern."
    ),
}

SCENARIO_DEFAULTS = [
    {"name": "Base Case",   "peak_pct": 100, "curve": "S-Curve (Logistic)",  "color": "#0ea5e9"},
    {"name": "Optimistic",  "peak_pct": 140, "curve": "Rapid Ramp (R-Curve)", "color": "#10b981"},
    {"name": "Pessimistic", "peak_pct": 65,  "curve": "Slow Build",           "color": "#f59e0b"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Market Setup
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Market Setup")
ms1, ms2, ms3 = st.columns(3)

with ms1:
    peak_demand = st.number_input(
        "Peak weekly demand (units)",
        min_value=1, value=10_000, step=500,
        help="The steady-state weekly demand once the product is fully adopted at horizon.",
    )

with ms2:
    horizon_years = st.selectbox(
        "Uptake horizon",
        options=[1, 2, 3, 4, 5],
        index=1,
        format_func=lambda x: f"{x} year{'s' if x > 1 else ''}",
        help="Time from launch to reaching peak / steady-state demand.",
    )
    horizon_weeks = int(horizon_years) * 52

with ms3:
    launch_date = st.date_input("Launch date", value=date.today())

# Total window = uptake horizon + 1 year of steady state
total_weeks = horizon_weeks + 52
weeks_arr = np.arange(total_weeks, dtype=float)
dates_idx = pd.date_range(pd.Timestamp(launch_date), periods=total_weeks, freq="W-MON")

# ─────────────────────────────────────────────────────────────────────────────
# Scenario Builder
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Scenarios")
st.caption(
    "Each tab is one scenario. Adjust peak demand and curve shape independently. "
    "All active scenarios are overlaid on the charts below."
)

built_scenarios: dict[str, dict] = {}
tabs = st.tabs([s["name"] for s in SCENARIO_DEFAULTS])

for i, (tab, sdef) in enumerate(zip(tabs, SCENARIO_DEFAULTS)):
    with tab:
        enabled = st.checkbox("Include this scenario", value=True, key=f"en_{i}")
        if not enabled:
            continue

        left, right = st.columns([1, 2])

        with left:
            sc_peak_pct = st.slider(
                "Peak demand (% of base)",
                20, 300, sdef["peak_pct"], step=5,
                key=f"peak_pct_{i}",
                help="100% = the peak demand set above.",
            )
            sc_peak = peak_demand * sc_peak_pct / 100.0

            curve_type = st.selectbox(
                "Curve shape",
                options=list(CURVES.keys()),
                index=list(CURVES.keys()).index(sdef["curve"]),
                key=f"curve_{i}",
            )
            st.caption(CURVES[curve_type])

        with right:
            st.markdown("**Shape Parameters**")

            if curve_type == "S-Curve (Logistic)":
                inflection_pct = st.slider(
                    "Inflection point (% of horizon)", 10, 90, 40, key=f"infl_{i}",
                    help="The week at which adoption is growing fastest, as % of the uptake horizon. "
                         "40% means the curve inflects 40% of the way through the ramp.",
                )
                steepness = st.slider(
                    "Steepness", 2, 25, 10, key=f"steep_{i}",
                    help="Controls how sharp the S transition is. Higher = more abrupt switchover.",
                )
                params = {"inflection_pct": inflection_pct, "steepness": steepness}

            elif curve_type == "S-Curve (Gompertz)":
                displacement = st.slider(
                    "Displacement (b)", 1.0, 10.0, 4.0, step=0.5, key=f"b_{i}",
                    help="Higher = longer lag before takeoff begins.",
                )
                growth_rate = st.slider(
                    "Growth rate (c)", 2, 20, 8, key=f"c_{i}",
                    help="Higher = faster rise from takeoff to plateau.",
                )
                params = {"displacement": displacement, "growth_rate": growth_rate}

            elif curve_type == "Bass Diffusion":
                p_innov = st.slider(
                    "Innovation coefficient (p)", 0.001, 0.10, 0.03, step=0.001,
                    format="%.3f", key=f"p_{i}",
                    help="Annual rate of independent adoption (innovators). Typical consumer goods: 0.01–0.05.",
                )
                q_imit = st.slider(
                    "Imitation coefficient (q)", 0.01, 0.60, 0.20, step=0.01,
                    key=f"q_{i}",
                    help="Strength of word-of-mouth / social contagion. Typical: 0.1–0.4. "
                         "High q = viral launch. Low q = slow organic spread.",
                )
                st.caption(
                    f"q/p ratio = **{q_imit/p_innov:.1f}x** — "
                    + ("word-of-mouth dominant (viral)" if q_imit / p_innov > 5 else
                       "mixed innovator/imitator" if q_imit / p_innov > 2 else
                       "innovator-led adoption")
                )
                params = {"p": p_innov, "q": q_imit}

            elif curve_type == "Rapid Ramp (R-Curve)":
                ramp_speed = st.slider(
                    "Ramp speed", 1, 20, 8, key=f"ramp_{i}",
                    help="Controls how quickly demand saturates. Higher = faster approach to plateau.",
                )
                params = {"ramp_speed": ramp_speed}

            elif curve_type == "Slow Build":
                alpha = st.slider(
                    "Convexity (α)", 1.2, 5.0, 2.5, step=0.1, key=f"alpha_{i}",
                    help="Higher = more gradual start with a sharper acceleration near the horizon. "
                         "α=2 is a quadratic ramp; α=3 is cubic.",
                )
                params = {"alpha": alpha}

            else:  # Linear
                params = {}
                st.info("Linear ramp has no additional parameters — demand increases by a constant amount each week.")

        y = build_curve(weeks_arr, sc_peak, horizon_weeks, curve_type, params)
        built_scenarios[sdef["name"]] = {
            "y": y,
            "peak": sc_peak,
            "curve_type": curve_type,
            "color": sdef["color"],
        }

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
if not built_scenarios:
    st.info("Enable at least one scenario above.")
    st.stop()

horizon_date = dates_idx[horizon_weeks]

# ── Weekly demand ─────────────────────────────────────────────────────────────
st.subheader("Weekly Demand Curve")
st.caption(
    "Weekly units from launch date. The dashed vertical line marks the end of the uptake horizon — "
    "beyond it, demand is held at the peak (steady-state). "
    "Scroll right to see the full period."
)
fig_w = go.Figure()
for sname, sc in built_scenarios.items():
    fig_w.add_trace(go.Scatter(
        x=dates_idx, y=sc["y"],
        mode="lines", name=sname,
        line=dict(color=sc["color"], width=2.5),
    ))
fig_w.add_vline(
    x=horizon_date.isoformat(), line_dash="dash", line_color="#94a3b8",
    annotation_text="Uptake horizon", annotation_position="top right",
)
fig_w.update_layout(
    xaxis_title="Date", yaxis_title="Weekly Demand (units)",
    template="plotly_white", yaxis=dict(rangemode="tozero"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=20),
)
st.plotly_chart(fig_w, use_container_width=True)

# ── % of peak ─────────────────────────────────────────────────────────────────
st.subheader("Adoption Rate (% of Peak Demand)")
st.caption(
    "Normalised view showing how quickly each scenario ramps relative to its own peak. "
    "Useful for comparing curve shapes independently of volume assumptions."
)
fig_pct = go.Figure()
for sname, sc in built_scenarios.items():
    pct_y = sc["y"] / sc["peak"] * 100 if sc["peak"] > 0 else sc["y"] * 0
    fig_pct.add_trace(go.Scatter(
        x=dates_idx, y=pct_y,
        mode="lines", name=sname,
        line=dict(color=sc["color"], width=2.5),
    ))
fig_pct.add_vline(x=horizon_date.isoformat(), line_dash="dash", line_color="#94a3b8")
fig_pct.update_layout(
    xaxis_title="Date", yaxis_title="% of Peak Demand",
    template="plotly_white",
    yaxis=dict(range=[0, 110], ticksuffix="%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=20),
)
st.plotly_chart(fig_pct, use_container_width=True)

# ── Cumulative ────────────────────────────────────────────────────────────────
st.subheader("Cumulative Demand")
st.caption(
    "Total units shipped from launch to each date. "
    "Use this to size inventory builds, production ramp capacity, and cumulative revenue."
)
fig_cum = go.Figure()
for sname, sc in built_scenarios.items():
    fig_cum.add_trace(go.Scatter(
        x=dates_idx, y=np.cumsum(sc["y"]),
        mode="lines", name=sname,
        line=dict(color=sc["color"], width=2.5),
    ))
fig_cum.update_layout(
    xaxis_title="Date", yaxis_title="Cumulative Units",
    template="plotly_white", yaxis=dict(rangemode="tozero"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_cum, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Milestone table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Adoption Milestones")
st.caption(
    "Weeks from launch date to reach each threshold of peak demand. "
    "Use the 50% milestone to compare how quickly scenarios reach mass-market adoption."
)
MILESTONES = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
rows = []
for sname, sc in built_scenarios.items():
    y = sc["y"]
    row = {
        "Scenario": sname,
        "Curve": sc["curve_type"],
        "Peak (units/wk)": f"{sc['peak']:,.0f}",
    }
    for pct in MILESTONES:
        threshold = sc["peak"] * pct
        hits = np.where(y >= threshold)[0]
        row[f"{int(pct * 100)}%"] = f"Wk {hits[0]}" if len(hits) > 0 else "—"
    rows.append(row)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Export")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    export_scenario = st.selectbox(
        "Scenario to export",
        options=list(built_scenarios.keys()),
        key="launch_export",
    )

sc_exp = built_scenarios[export_scenario]
export_df = pd.DataFrame({
    "date": dates_idx.strftime("%Y-%m-%d"),
    "week": np.arange(1, total_weeks + 1),
    "weekly_demand": np.round(sc_exp["y"], 1),
    "cumulative_demand": np.round(np.cumsum(sc_exp["y"]), 1),
    "pct_of_peak": np.round(sc_exp["y"] / sc_exp["peak"] * 100, 2),
})

with exp_col1:
    st.download_button(
        "Download CSV",
        data=export_df.to_csv(index=False).encode(),
        file_name=f"launch_curve_{export_scenario.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

with exp_col2:
    st.markdown("&nbsp;")  # vertical align
    if st.button("Send to Export page as forecast", type="primary"):
        forecast_df = pd.DataFrame({
            "date": dates_idx,
            "forecast": np.round(sc_exp["y"], 1),
            "lower_80": np.round(sc_exp["y"] * 0.80, 1),
            "upper_80": np.round(sc_exp["y"] * 1.20, 1),
            "lower_95": np.round(sc_exp["y"] * 0.70, 1),
            "upper_95": np.round(sc_exp["y"] * 1.30, 1),
            "model_id": "launch_curve",
        })
        st.session_state.forecasts["launch_curve"] = forecast_df
        st.session_state.selected_model = "launch_curve"
        st.success(
            f"'{export_scenario}' launch curve added as a forecast — "
            "go to the Export page to download or adjust it."
        )
