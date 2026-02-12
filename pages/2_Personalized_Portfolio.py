# ===============================================================
# PAGE 02 — Personalized Portfolio Recommendation (FINAL EDITION)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_predictions():
    path = os.path.join(DIRS["predictions"], "latest_predictions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.rename(columns={
        "1D": "pred_1D", "5D": "pred_5D", "1M": "pred_1M",
        "3M": "pred_3M", "6M": "pred_6M", "1Y": "pred_1Y",
        "3Y": "pred_3Y", "5Y": "pred_5Y"
    })
    return df


@st.cache_data
def load_risks():
    path = os.path.join(DIRS["risk"], "risk_scores.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_macro_scores():
    path = os.path.join(DIRS["macro"], "macro_scores.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ---------------------------------------------------------------
# HORIZON GROUPS
# ---------------------------------------------------------------
HORIZON_GROUPS = {
    "1D":  {"selected": ["pred_1D"], "adjacent": ["pred_5D"], "others": ["pred_1M","pred_3M","pred_6M","pred_1Y","pred_3Y","pred_5Y"]},
    "5D":  {"selected": ["pred_5D"], "adjacent": ["pred_1D","pred_1M"], "others": ["pred_3M","pred_6M","pred_1Y","pred_3Y","pred_5Y"]},
    "1M":  {"selected": ["pred_1M"], "adjacent": ["pred_5D","pred_3M"], "others": ["pred_1D","pred_6M","pred_1Y","pred_3Y","pred_5Y"]},
    "3M":  {"selected": ["pred_3M"], "adjacent": ["pred_1M","pred_6M"], "others": ["pred_1D","pred_5D","pred_1Y","pred_3Y","pred_5Y"]},
    "6M":  {"selected": ["pred_6M"], "adjacent": ["pred_3M","pred_1Y"], "others": ["pred_1D","pred_5D","pred_1M","pred_3Y","pred_5Y"]},
    "1Y":  {"selected": ["pred_1Y"], "adjacent": ["pred_6M","pred_3Y"], "others": ["pred_1D","pred_5D","pred_1M","pred_3M","pred_5"]},
    "3Y":  {"selected": ["pred_3Y"], "adjacent": ["pred_1Y","pred_5Y"], "others": ["pred_1D","pred_5D","pred_1M","pred_3M","pred_6M"]},
    "5Y":  {"selected": ["pred_5Y"], "adjacent": ["pred_3Y"], "others": ["pred_1D","pred_5D","pred_1M","pred_3M","pred_6M","pred_1Y"]},
}


# ---------------------------------------------------------------
# DYNAMIC FORECAST SCORE (Selected = 55%, Adjacent = 25%, Others = 20%)
# ---------------------------------------------------------------
def compute_dynamic_forecast(row, horizon):

    groups = HORIZON_GROUPS[horizon]

    def avg(cols):
        vals = [row[c] for c in cols if c in row and not pd.isna(row[c])]
        return np.mean(vals) if vals else 0

    sel = avg(groups["selected"])
    adj = avg(groups["adjacent"])
    oth = avg(groups["others"])

    score = (0.55 * sel) + (0.25 * adj) + (0.20 * oth)
    return score


# ---------------------------------------------------------------
# RISK-BASED WEIGHTS
# ---------------------------------------------------------------
def get_weight_params(risk_type):
    if risk_type == "Conservative":
        return 0.20, 0.60, 0.15, 0.05
    elif risk_type == "Moderate":
        return 0.35, 0.35, 0.20, 0.10
    else:
        return 0.60, 0.20, 0.15, 0.05

def goal_multiplier(row, goal):
    """
    Adjusts asset weight based on user goal.
    """
    risk = row["final_risk_score"]
    forecast = row["forecast_score"]

    if goal == "Wealth Growth":
        return 1.0 + (forecast * 0.5)

    if goal == "Aggressive Growth":
        return 1.0 + (forecast * 0.7) + (risk / 200)

    if goal == "Stable Income":
        return 1.2 - (risk / 150)

    if goal == "Capital Preservation":
        return 1.3 - (risk / 100)

    return 1.0

# ---------------------------------------------------------------
# BUILD PORTFOLIO LOGIC
# ---------------------------------------------------------------
def build_portfolio(preds, risks, macro, user_profile):

    α, β, γ, δ = get_weight_params(user_profile["risk_level"])
    macro_strength = float(macro["macro_composite"].iloc[0])

    df = preds.merge(risks, on="symbol", how="left")

    # FORECAST SCORE BASED ON SELECTED HORIZON
    df["forecast_score"] = df.apply(
        lambda r: compute_dynamic_forecast(r, user_profile["horizon"]),
        axis=1
    )

    # Remove negative forecasts
    df = df[df["forecast_score"] > 0]

    # RISK FILTER
    if user_profile["risk_level"] == "Conservative":
        df = df[df["risk_label"].isin(["LOW","MEDIUM"])]
    elif user_profile["risk_level"] == "Moderate":
        df = df[(df["risk_label"] != "HIGH") |
                (df["forecast_score"] > df["forecast_score"].mean())]

    if df.empty:
        return pd.DataFrame(), "No assets match your filters."

    df["risk_norm"] = df["final_risk_score"] / 100
    df["macro_norm"] = macro_strength / 100
    df["user_factor"] = user_profile["income"] / 150000  

    # MAIN FORMULA
    df["goal_bias"] = df.apply(
    lambda r: goal_multiplier(r, user_profile["goal"]),
    axis=1
    )

    df["raw_weight"] = (
        α * df["forecast_score"] +
        β * (1 - df["risk_norm"]) +
        γ * df["macro_norm"] +
        δ * df["user_factor"]
    ) * df["goal_bias"]


    # SELECT TOP ASSETS
    top_n = 6 if user_profile["risk_level"] == "Conservative" else \
            8 if user_profile["risk_level"] == "Moderate" else 12

    df = df.nlargest(top_n, "raw_weight")

    # NORMALIZE
    df["weight"] = df["raw_weight"] / df["raw_weight"].sum()

    final_df = df[["symbol", "risk_label", "weight"]]

    explanation = f"""
### Why These Weights?
Using **Dynamic Multi-Horizon Forecasting**, the model gives **55% weight** 
to your selected horizon (**{user_profile['horizon']}**),
25% to adjacent trends, and 20% to remaining outlooks.

This is combined with:
- **α Forecast Influence ({α})**
- **β Risk Adjustment ({β})**
- **γ Macro Sensitivity ({γ})**
- **δ User Profile Factor ({δ})**

This ensures a stable, personalized, multi-agent portfolio aligned with your risk and goals.

### How Your Investment Goal Influenced This Portfolio

In addition to risk tolerance and forecast horizons, your selected goal
(**{user_profile['goal']}**) directly influenced asset allocation.

#### Goal-Based Allocation Logic:
- **Wealth Growth** → Higher allocation to assets with strong predicted returns
- **Aggressive Growth** → Strong forecast assets allowed higher risk exposure
- **Stable Income** → Portfolio tilted toward lower-risk, stable assets
- **Capital Preservation** → Capital protection prioritized over returns

Mathematically, a **goal-specific bias multiplier** was applied to each asset’s
weight after forecast and risk evaluation, ensuring your objective directly
shapes the final portfolio.

This guarantees that:
> Same assets + same risk ≠ same portfolio if goals differ
"""
    return final_df, explanation


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("Personalized Portfolio Recommendation")

left, right = st.columns([1, 2])

# ------------------- LEFT INPUTS -------------------
with left:
    st.subheader("Provide your details")

    age = st.number_input("Age", 18, 80, 30)
    income = st.number_input("Annual Income ($)", value=50000)
    amount = st.number_input("Investment Amount ($)", 100, 1000000, 5000)

    horizon = st.radio(
        "Select Horizon",
        ["1D","5D","1M","3M","6M","1Y","3Y","5Y"],
        horizontal=True
    )

    risk_type = st.selectbox("Risk Preference", ["Conservative", "Moderate", "Aggressive"])
    goal = st.selectbox("Primary Goal", ["Wealth Growth","Aggressive Growth","Stable Income","Capital Preservation"])

    generate = st.button("Generate Portfolio")


# ------------------- RIGHT OUTPUT -------------------
with right:

    if not generate:
        st.info("Fill your details and click Generate Portfolio.")
        st.stop()

    preds = load_predictions()
    risks = load_risks()
    macro = load_macro_scores()

    if preds.empty or risks.empty:
        st.error("Run all agents first (market, risk, predictions).")
        st.stop()

    user_profile = {
        "risk_level": risk_type,
        "age": age,
        "goal": goal,
        "income": income,
        "horizon": horizon
    }

    df, explanation = build_portfolio(preds, risks, macro, user_profile)

    if df.empty:
        st.warning("No assets match the current filters.")
        st.stop()

    df["investment"] = df["weight"] * amount

    st.subheader("Final Portfolio Allocation")
    st.dataframe(df, use_container_width=True)

    fig = px.pie(df, names="symbol", values="weight", title="Portfolio Weight Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Reasoning")
    st.info(explanation)

    st.success("Portfolio successfully generated!")
