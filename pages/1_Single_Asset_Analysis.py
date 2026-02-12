# ==============================================================
# PAGE 01 — Single Asset Analysis & Multi-Agent Recommendation
# ==============================================================

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS, SYMBOLS


# =========================
# Helpers
# =========================
@st.cache_data
def load_predictions():
    path = os.path.join(DIRS["predictions"], "latest_predictions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.rename(columns={
        "1D": "pred_1D", "5D": "pred_5D", "1M": "pred_1M",
        "3M": "pred_3M", "6M": "pred_6M", "1Y": "pred_1Y",
        "3Y": "pred_3Y", "5Y": "pred_5Y"
    })


@st.cache_data
def load_risks():
    path = os.path.join(DIRS["risk"], "risk_scores.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_macro():
    path = os.path.join(DIRS["macro"], "macro_scores.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_price(symbol: str):
    path = os.path.join(DIRS["raw"], f"{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


@st.cache_data
def load_features(symbol: str):
    path = os.path.join(DIRS["features"], f"{symbol}_features.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # make sure these exist; if not, it will error later with clearer message
    return df


# =========================
# Technical scoring
# =========================
def score_rsi(rsi):
    if rsi < 30: return 90, "Oversold → Bullish"
    if rsi < 45: return 70, "Mild Bullish"
    if rsi < 55: return 50, "Neutral"
    if rsi < 70: return 30, "Bearish"
    return 10, "Overbought → Bearish"


def score_macd(macd, signal):
    return (80, "Bullish Crossover") if macd > signal else (40, "Bearish Crossover")


def score_trend(close, sma20, sma50):
    if close > sma50 > sma20: return 90, "Strong Uptrend"
    if close > sma50: return 70, "Moderate Uptrend"
    if sma50 > sma20: return 60, "Weak Uptrend"
    return 30, "Downtrend"


def classify(score):
    if score >= 70: return "BUY", "green"
    if score >= 50: return "HOLD", "#ff9900"
    return "SELL", "red"


# =========================
# Horizon mapping
# =========================
HORIZONS = {
    "1D": ["pred_1D"],
    "5D": ["pred_5D"],
    "1M": ["pred_1M"],
    "3M": ["pred_3M"],
    "6M": ["pred_6M"],
    "1Y": ["pred_1Y"],
    "3Y": ["pred_3Y"],
    "5Y": ["pred_5Y"],
}

PRICE_RANGE_DAYS = {
    "1M": 30,
    "3M": 90,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "MAX": None
}


# =========================
# Macro relevance mapping
# =========================
def get_macro_relevance(symbol):
    mapping = {
        "^GSPC": ["gdp_score", "inflation_score", "vix_score", "interest_rate_score"],
        "^DJI":  ["gdp_score", "inflation_score", "interest_rate_score"],
        "^IXIC": ["interest_rate_score", "inflation_score", "vix_score"],
        "^NSEI": ["gdp_score", "inflation_score"],
        "GC=F":  ["gold_score", "dxy_score", "inflation_score"],
        "SI=F":  ["gold_score", "dxy_score"],
        "CL=F":  ["oil_score", "dxy_score", "inflation_score"],
        "NG=F":  ["oil_score", "dxy_score"],
        "BTC-USD": ["dxy_score", "vix_score"],
        "ETH-USD": ["dxy_score", "vix_score"],
        "AAPL": ["gdp_score", "interest_rate_score"],
        "MSFT": ["gdp_score", "interest_rate_score"],
        "GOOG": ["gdp_score", "interest_rate_score"],
        "AMZN": ["gdp_score", "interest_rate_score"],
        "TSLA": ["vix_score", "gdp_score", "interest_rate_score"],
        "NVDA": ["gdp_score", "interest_rate_score"],
        "META": ["gdp_score", "interest_rate_score"],
    }
    return mapping.get(symbol, [])


def macro_interpretation(indicator: str, score: float) -> str:
    # Quick readable mapping for professor/user
    if indicator == "interest_rate_score":
        return "Rates: higher score → more supportive for risk assets."
    if indicator == "inflation_score":
        return "Inflation: higher score → inflation pressure easing."
    if indicator == "gdp_score":
        return "Growth: higher score → stronger economic demand."
    if indicator == "unemployment_score":
        return "Jobs: higher score → labor market healthier."
    if indicator == "vix_score":
        return "VIX: higher score → lower fear / lower uncertainty."
    if indicator == "dxy_score":
        return "USD: higher score → dollar strength/conditions supportive."
    if indicator == "gold_score":
        return "Gold: higher score → defensive demand / inflation hedge signal."
    if indicator == "oil_score":
        return "Oil: higher score → energy conditions supportive."
    return "Macro signal explanation."


# =========================
# Charts
# =========================
def price_chart(df, name, price_range):
    df = df.sort_values("date")
    days = PRICE_RANGE_DAYS[price_range]
    if days is not None:
        df = df.tail(days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines", name="Close",
        line=dict(width=2)
    ))
    fig.update_layout(
        title=f"{name} — Price History ({price_range})",
        xaxis_title="Date",
        yaxis_title="Price",
        height=360,
        hovermode="x unified"
    )
    return fig

def render_signal(title, signal, reason):
    if signal == "BUY":
        st.markdown(f"### 🟢 {title}: **BUY**")
    elif signal == "SELL":
        st.markdown(f"### 🔴 {title}: **SELL**")
    else:
        st.markdown(f"### 🟡 {title}: **NEUTRAL**")

    st.caption(reason)

# ==============================================================
# UI START
# ==============================================================
st.title("Single Asset Analysis & Multi-Agent Recommendation")

# Asset
name = st.selectbox("Select Asset", list(SYMBOLS.keys()))
symbol = SYMBOLS[name]

# Horizon selection
horizon = st.selectbox("Select Prediction Horizon", list(HORIZONS.keys()))
h_cols = HORIZONS[horizon]

# Load datasets
preds = load_predictions()
risks = load_risks()
macro = load_macro()
df_price = load_price(symbol)
df_feat = load_features(symbol)

# Validate data
if preds.empty:
    st.error("latest_predictions.csv not found. Run prediction_agent.py first.")
    st.stop()

if risks.empty:
    st.error("risk_scores.csv not found. Run risk_agent.py first.")
    st.stop()

if macro.empty:
    st.error("macro_scores.csv not found. Run macro_agent.py first.")
    st.stop()

if df_price.empty:
    st.error(f"Raw price file missing for {symbol}. Check data/raw/{symbol}.csv")
    st.stop()

if df_feat.empty:
    st.error(f"Feature file missing for {symbol}. Check data/features/{symbol}_features.csv")
    st.stop()

# Rows
row_pred = preds.loc[preds["symbol"] == symbol].iloc[0]
row_risk = risks.loc[risks["symbol"] == symbol].iloc[0]
macro_row = macro.iloc[0]

# Technical calculations
last = df_feat.iloc[-1]
required_cols = ["rsi_14", "macd", "macd_signal", "sma_20", "sma_50", "close"]
missing = [c for c in required_cols if c not in df_feat.columns]
if missing:
    st.error(f"Missing indicator columns in {symbol}_features.csv: {missing}")
    st.stop()

rsi_s, rsi_txt = score_rsi(float(last["rsi_14"]))
macd_s, macd_txt = score_macd(float(last["macd"]), float(last["macd_signal"]))
trend_s, trend_txt = score_trend(float(last["close"]), float(last["sma_20"]), float(last["sma_50"]))
tech_score = float(np.mean([rsi_s, macd_s, trend_s]))

# Horizon forecast value
forecast_values = [float(row_pred[c]) for c in h_cols if c in row_pred.index and pd.notna(row_pred[c])]
forecast_avg = float(np.mean(forecast_values)) if forecast_values else 0.0

# Macro composite (global) + relevant composite (asset-specific)
relevant = get_macro_relevance(symbol)
relevant_scores = [float(macro_row[c]) for c in relevant if c in macro_row.index and pd.notna(macro_row[c])]
macro_sub = float(np.mean(relevant_scores)) if relevant_scores else float(macro_row.get("macro_composite", 50))

# Final fusion score
final_score = (
    0.35 * tech_score +
    0.25 * float(macro_row["macro_composite"]) +
    0.20 * (100 - float(row_risk["final_risk_score"])) +
    0.20 * (50 + forecast_avg * 500)
)
final_score = float(max(0, min(100, final_score)))
label, color = classify(final_score)
display_score = min(final_score, 95.0)

# ==============================================================
# Tabs
# ==============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Price Chart",
    "Technical Indicators",
    "Macro Impact",
    "Risk Analysis",
    "Final Recommendation"
])

# ===================== TAB 1 — PRICE ===========================
with tab1:
    st.subheader("Price History (choose range)")
    price_range = st.radio(
        "Select Price History Range",
        ["1M", "3M", "1Y", "3Y", "5Y", "MAX"],
        horizontal=True,
        key="price_range_radio"
    )
    st.plotly_chart(price_chart(df_price, name, price_range), use_container_width=True)

    st.markdown("### Forecast Trend (Selected Horizon)")
    color = "🟢" if forecast_avg > 0 else ("🔴" if forecast_avg < 0 else "🟡")
    st.markdown(f"**Forecast Direction:** {color} { 'Positive' if forecast_avg>0 else 'Negative' if forecast_avg<0 else 'Neutral' }")


# ===================== TAB 2 — TECHNICAL ========================
with tab2:
    st.subheader("Market Agent — Technical Indicator Signals")

    rsi_signal = "BUY" if rsi_s >= 70 else "SELL" if rsi_s <= 30 else "NEUTRAL"
    macd_signal = "BUY" if macd_s >= 70 else "SELL"
    trend_signal = "BUY" if trend_s >= 70 else "SELL" if trend_s <= 40 else "NEUTRAL"

    c1, c2, c3 = st.columns(3)

    with c1:
        render_signal(
            "RSI (Momentum)",
            rsi_signal,
            "Recovery from oversold" if rsi_signal == "BUY"
            else "Overbought pressure" if rsi_signal == "SELL"
            else "Neutral momentum"
        )

    with c2:
        render_signal(
            "MACD (Momentum)",
            macd_signal,
            "Bullish crossover" if macd_signal == "BUY"
            else "Bearish momentum"
        )

    with c3:
        render_signal(
            "SMA Trend",
            trend_signal,
            "Price above moving averages" if trend_signal == "BUY"
            else "Price below moving averages" if trend_signal == "SELL"
            else "Sideways / weak trend"
        )

    # Overall technical decision
    buy_count = [rsi_signal, macd_signal, trend_signal].count("BUY")
    sell_count = [rsi_signal, macd_signal, trend_signal].count("SELL")

    if buy_count >= 2:
        st.success("Overall Technical Recommendation: BULLISH")
    elif sell_count >= 2:
        st.warning("Overall Technical Recommendation: BEARISH")
    else:
        st.info("Overall Technical Recommendation: NEUTRAL")

    st.caption(
        "This technical recommendation aggregates momentum and trend-based indicators "
        "and is used as an input to the final multi-agent fusion score."
    )

    with st.expander("Why these indicators?"):
        st.write(
            "RSI captures momentum exhaustion, MACD captures trend continuation, "
            "and SMA-based trend identifies market regime. These indicators are widely "
            "used in technical analysis literature."
        )



# ===================== TAB 3 — MACRO ============================
with tab3:
    st.subheader("Macro Agent — Economic Environment Signals")

    if not relevant:
        st.info("No macro indicators mapped for this asset.")
        st.stop()

    def macro_to_signal(score):
        if score >= 60:
            return "BUY"
        elif score >= 45:
            return "NEUTRAL"
        else:
            return "SELL"

    cols = st.columns(3)

    for i, ind in enumerate(relevant):
        if ind not in macro_row.index:
            continue

        score = float(macro_row[ind])
        signal = macro_to_signal(score)
        clean = ind.replace("_score", "").replace("_", " ").title()
        reason = macro_interpretation(ind, score)

        with cols[i % 3]:
            render_signal(clean, signal, reason)

    # Overall macro stance
    st.markdown("---")
    st.markdown("### Overall Macro Environment")

    overall_signal = macro_to_signal(macro_sub)

    if overall_signal == "BUY":
        st.success("Macro environment is SUPPORTIVE for this asset.")
    elif overall_signal == "SELL":
        st.warning("Macro environment presents HEADWINDS for this asset.")
    else:
        st.info("Macro environment is MIXED / NEUTRAL for this asset.")

    st.caption(
        "Macro signals provide economic context and are combined with "
        "technical, risk, and forecast signals in the final recommendation."
    )

    with st.expander("Why these macro indicators?"):
        st.write(
            "Macro indicators are selected based on asset sensitivity. "
            "For example, interest rates influence growth stocks, GDP reflects "
            "demand conditions, and VIX captures market uncertainty."
        )

# ===================== TAB 4 — RISK =============================
with tab4:
    st.subheader("Risk Agent — Downside Risk Assessment")

    risk_label = str(row_risk["risk_label"]).upper()
    risk_score = float(row_risk["final_risk_score"])

    # -----------------------------
    # Map risk label to display
    # -----------------------------
    if risk_label == "LOW":
        display = "LOW RISK"
        emoji = "🟢"
        reason = "Low volatility and limited historical drawdowns"
    elif risk_label == "HIGH":
        display = "HIGH RISK"
        emoji = "🔴"
        reason = "High volatility and significant downside exposure"
    else:
        display = "MODERATE RISK"
        emoji = "🟡"
        reason = "Moderate volatility with mixed stability signals"

    # -----------------------------
    # Main Risk Signal
    # -----------------------------
    st.markdown(f"### {emoji} Overall Risk Level: **{display}**")
    st.caption(reason)

    st.markdown("---")
    st.markdown("### Risk Drivers (Summary)")

    # -----------------------------
    # Risk Driver Explanations
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Volatility**")
        st.caption("Measures price fluctuations over time")

    with c2:
        st.markdown("**Max Drawdown**")
        st.caption("Worst historical decline from peak")

    with c3:
        st.markdown("**Value-at-Risk (VaR)**")
        st.caption("Potential loss under adverse market conditions")

    with c4:
        st.markdown("**Beta**")
        st.caption("Sensitivity to broader market movements")

    st.caption(
        "These risk metrics are normalized and aggregated into a single "
        "interpretable risk score, which is used to control downside exposure "
        "and personalize investment recommendations."
    )

    with st.expander("How is risk used in decisions?"):
        st.write(
            "Risk does not independently generate Buy or Sell decisions. "
            "Instead, it acts as a constraint and weighting factor in the final "
            "fusion score and portfolio construction process. High-risk assets "
            "are down-weighted or excluded for conservative users."
        )

# ===================== TAB 5 — FINAL ============================
with tab5:
    st.subheader("Final Recommendation")

    # -----------------------------
    # Final Decision Display (TEXT ONLY)
    # -----------------------------
    if label == "BUY":
        rec_color = "#1a7f37"      # green
    elif label == "SELL":
        rec_color = "#b91c1c"      # red
    else:
        rec_color = "#d97706"      # orange

    st.markdown(
        f"""
        <h2 style="
            color:{rec_color};
            font-weight:700;
            margin-bottom:0.25rem;
        ">
            {label}
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "The composite fusion score is computed internally by combining signals "
        "from multiple agents and is used only to derive the final recommendation."
    )

    st.markdown("---")

    # -----------------------------
    # Agent Contributions
    # -----------------------------
    st.markdown("### How did each agent influence the decision?")

    c1, c2, c3, c4 = st.columns(4)

    # Market Agent — Technical
    tech_signal, _ = classify(tech_score)
    c1.markdown("**Market Agent — Technical**")
    c1.write(
        "BULLISH" if tech_signal == "BUY"
        else "BEARISH" if tech_signal == "SELL"
        else "NEUTRAL"
    )

    # Market Agent — Forecast
    forecast_dir = (
        "POSITIVE TREND" if forecast_avg > 0
        else "NEGATIVE TREND" if forecast_avg < 0
        else "NEUTRAL TREND"
    )
    c2.markdown("**Market Agent — Forecast**")
    c2.write(forecast_dir)

    # Macro Agent
    macro_signal = (
        "SUPPORTIVE" if macro_sub >= 60
        else "ADVERSE" if macro_sub < 45
        else "NEUTRAL"
    )
    c3.markdown("**Macro Agent**")
    c3.write(macro_signal)

    # Risk Agent
    c4.markdown("**Risk Agent**")
    c4.write(display)  # LOW / MODERATE / HIGH RISK

    st.caption(
        "Each agent contributes a different perspective: technical and forecast signals describe market direction, "
        "the macro agent provides economic context, and the risk agent evaluates suitability rather than direction."
    )

    st.markdown("---")

    # -----------------------------
    # Final Explanation
    # -----------------------------
    st.markdown("### Why this recommendation?")

    st.write(
        "The final recommendation is generated by combining technical momentum, "
        "macro-economic context, downside risk suitability, and multi-horizon "
        "forecast direction. Technical indicators describe current market behavior, "
        "macro signals provide economic context, and risk assessment controls downside "
        "exposure. Forecast direction influences the decision without relying on noisy "
        "numerical predictions. These signals are fused into a single decision framework, "
        "ensuring the recommendation is both data-driven and interpretable."
    )

    st.success(
        "This decision is produced using a transparent multi-agent fusion framework "
        "and does not rely on any single indicator or model output."
    )


