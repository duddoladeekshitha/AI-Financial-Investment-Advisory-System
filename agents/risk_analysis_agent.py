# ===========================================================
# RISK ANALYSIS AGENT — Computes asset-level risk metrics
# ===========================================================

import os
import sys
import json
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS, SYMBOLS


# -----------------------------------------------------------
# Load processed feature file
# -----------------------------------------------------------
def load_features(symbol):
    path = os.path.join(DIRS["features"], f"{symbol}_features.csv")
    if not os.path.exists(path):
        print(f"[WARN] No features found for {symbol}")
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


# -----------------------------------------------------------
# Compute Max Drawdown
# -----------------------------------------------------------
def max_drawdown(series):
    cumulative = series / series.iloc[0]
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min() * -1  # positive number


# -----------------------------------------------------------
# Compute Value at Risk (95% VaR)
# -----------------------------------------------------------
def value_at_risk(returns):
    return abs(np.percentile(returns.dropna(), 5))


# -----------------------------------------------------------
# Compute Beta vs S&P 500 (^GSPC)
# -----------------------------------------------------------
def compute_beta(asset_returns, market_returns):
    if len(asset_returns) != len(market_returns):
        min_len = min(len(asset_returns), len(market_returns))
        asset_returns = asset_returns.tail(min_len)
        market_returns = market_returns.tail(min_len)

    cov = np.cov(asset_returns, market_returns)[0][1]
    var = np.var(market_returns)

    if var == 0:
        return 0
    return cov / var


# -----------------------------------------------------------
# Convert risk metric into standardized 0–100 score
# -----------------------------------------------------------
def normalize_score(value, max_expected, invert=False):
    """
    Example: higher volatility = higher risk score
    invert=True means high value → low score
    """
    score = (value / max_expected) * 100
    score = max(0, min(score, 100))
    if invert:
        score = 100 - score
    return round(score, 2)


# -----------------------------------------------------------
# Risk Category
# -----------------------------------------------------------
def risk_label(score):
    if score < 35:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    else:
        return "HIGH"


# -----------------------------------------------------------
# Generate explanation for advisor engine
# -----------------------------------------------------------
def generate_risk_explanation(symbol, vol, mdd, var, beta, score):
    lines = []

    # Volatility comments
    if vol > 0.025:
        lines.append("This asset shows high price volatility, making it sensitive to market swings.")
    else:
        lines.append("The asset has relatively low volatility.")

    # Max Drawdown comments
    if mdd > 0.35:
        lines.append("Historical drawdowns have been large, indicating significant risk during downturns.")
    elif mdd < 0.15:
        lines.append("Drawdowns have been shallow, suggesting strong downside protection.")

    # VaR comments
    if var > 0.03:
        lines.append("The 95% Value-at-Risk is high, meaning the asset can experience large short-term losses.")
    else:
        lines.append("The Value-at-Risk is relatively stable.")

    # Beta comments
    if beta > 1.2:
        lines.append("It is more sensitive than the overall market (high beta).")
    elif beta < 0.8:
        lines.append("It moves slower than the overall market (low beta).")

    # Final risk summary
    if score < 35:
        lines.append("Overall risk level: LOW — suitable for conservative portfolios.")
    elif score < 70:
        lines.append("Overall risk level: MEDIUM — suitable for moderate-risk investors.")
    else:
        lines.append("Overall risk level: HIGH — suitable only for aggressive investors.")

    return lines


# -----------------------------------------------------------
# MAIN RISK AGENT
# -----------------------------------------------------------
def run_risk_agent():

    print("\n========== RISK AGENT STARTED ==========\n")

    # Load benchmark (^GSPC)
    df_spx = load_features("^GSPC")
    spx_returns = df_spx["return_1d"]

    all_results = []

    for name, symbol in SYMBOLS.items():

        print(f"[INFO] Processing risk → {symbol}")

        df = load_features(symbol)
        if df is None:
            continue

        # Compute metrics
        close = df["close"]
        returns = df["return_1d"]

        volatility = returns.std()
        mdd = max_drawdown(close)
        var = value_at_risk(returns)
        beta = compute_beta(returns, spx_returns)

        # Convert to standardized 0–100 risk scores
        vol_score = normalize_score(volatility, 0.04)       # stocks ~2–4% daily volatility
        mdd_score = normalize_score(mdd, 0.60)              # max drawdown up to 60%
        var_score = normalize_score(var, 0.05)              # VaR up to ~5%
        beta_score = normalize_score(abs(beta), 2.0)        # beta range ~0 to 2

        # Composite
        final_score = round(
            0.35 * vol_score +
            0.30 * mdd_score +
            0.20 * var_score +
            0.15 * beta_score, 2
        )

        label = risk_label(final_score)
        explanation = generate_risk_explanation(symbol, volatility, mdd, var, beta, final_score)

        # Save result
        all_results.append([
            symbol, volatility, mdd, var, beta,
            vol_score, mdd_score, var_score, beta_score,
            final_score, label, explanation
        ])

    # Create DF
    df_out = pd.DataFrame(
        all_results,
        columns=[
            "symbol", "volatility", "max_drawdown", "VaR_95", "beta",
            "vol_score", "mdd_score", "var_score", "beta_score",
            "final_risk_score", "risk_label", "risk_explanation"
        ]
    )

    # Save CSV
    out_csv = os.path.join(DIRS["risk"], "risk_scores.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Risk scores saved → {out_csv}")

    # Save JSON for Advisor Engine
    out_json = os.path.join(DIRS["risk"], "risk_scores.json")
    with open(out_json, "w") as f:
        json.dump(df_out.to_dict(orient="records"), f, indent=4)

    print(f"[OK] JSON saved → {out_json}")
    print("\n========== RISK AGENT COMPLETE ==========\n")


if __name__ == "__main__":
    run_risk_agent()
