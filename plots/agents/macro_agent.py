# ===========================================================
# MACRO ECONOMIC AGENT
# ===========================================================

import os
import sys
import json
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS


# -----------------------------------------------------------
# Load processed macro file
# -----------------------------------------------------------
def load_macro(name):
    path = os.path.join(DIRS["macro"], "processed", f"{name}_daily.csv")
    df = pd.read_csv(path)

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Identify numeric column
    value_col = [c for c in df.columns if c != "date"][0]

    # Remove rows where value is not numeric
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["date", value_col])

    # Reset index
    df = df.reset_index(drop=True)

    return df



# -----------------------------------------------------------
# Compute trend score (0–100)
# -----------------------------------------------------------
def trend_score(series, window=180):
    recent = series.tail(window)
    change = recent.iloc[-1] - recent.iloc[0]
    pct = (change / abs(recent.iloc[0])) * 100 if recent.iloc[0] != 0 else 0

    # Scale change to 0–100 range
    score = max(0, min(100, 50 + pct))
    return round(score, 2)


# -----------------------------------------------------------
# Macro Regime Classifier
# -----------------------------------------------------------
def classify_macro_regime(cpi, unrate, gdp, dff, vix, dxy, gold, oil):
    regime = []

    # Inflation regime
    if cpi > 55:
        regime.append("High Inflation")
    elif cpi < 45:
        regime.append("Low Inflation")
    else:
        regime.append("Stable Inflation")

    # Employment
    if unrate > 55:
        regime.append("Weak Labor Market")
    elif unrate < 45:
        regime.append("Strong Labor Market")
    else:
        regime.append("Moderate Labor Market")

    # GDP
    if gdp < 45:
        regime.append("Recession Risk")
    elif gdp > 55:
        regime.append("Economic Expansion")
    else:
        regime.append("Neutral Growth")

    # Interest Rate Regime (DFF)
    if dff > 55:
        regime.append("Restrictive Monetary Policy")
    elif dff < 45:
        regime.append("Accommodative Monetary Policy")
    else:
        regime.append("Neutral Policy")

    # VIX Volatility
    if vix > 55:
        regime.append("High Market Volatility")
    else:
        regime.append("Low/Normal Volatility")

    # Dollar Strength
    if dxy > 55:
        regime.append("Strong Dollar Environment")
    else:
        regime.append("Weak Dollar Environment")

    # Commodities
    if gold > 55:
        regime.append("Risk-Off Sentiment (Gold Rising)")
    if oil > 55:
        regime.append("Inflationary Pressure (Oil Rising)")

    return regime


# -----------------------------------------------------------
# Generate Natural Language Summary
# -----------------------------------------------------------
def generate_summary(regime_list):
    summary = []

    for item in regime_list:
        if item == "High Inflation":
            summary.append("Inflation is rising — this usually pressures equities and favors commodities like Gold and Oil.")
        elif item == "Low Inflation":
            summary.append("Inflation is cooling — supportive for long-term equity growth.")
        elif item == "Strong Labor Market":
            summary.append("Employment is strong — economic momentum remains stable.")
        elif item == "Weak Labor Market":
            summary.append("Unemployment rising — signals recession risk.")
        elif item == "Economic Expansion":
            summary.append("GDP trend indicates economic expansion.")
        elif item == "Recession Risk":
            summary.append("GDP is weakening — recession risk increasing.")
        elif item == "Restrictive Monetary Policy":
            summary.append("Interest rates are high — borrowing is expensive, tech stocks usually underperform.")
        elif item == "Accommodative Monetary Policy":
            summary.append("Rates are low — supportive for stock market growth.")
        elif item == "High Market Volatility":
            summary.append("VIX is high — markets are stressed and risky.")
        elif item == "Strong Dollar Environment":
            summary.append("Dollar strong — EM equities and commodities may weaken.")
        elif item == "Weak Dollar Environment":
            summary.append("Dollar weakening — good for global equities and commodities.")
        elif item == "Risk-Off Sentiment (Gold Rising)":
            summary.append("Gold rising — investors are moving to safety.")
        elif item == "Inflationary Pressure (Oil Rising)":
            summary.append("Oil rising — energy costs increasing, inflation pressures up.")

    return summary


# -----------------------------------------------------------
# MAIN MACRO AGENT
# -----------------------------------------------------------
def run_macro_agent():

    print("\n========== MACRO AGENT STARTED ==========\n")

    # Load all macro datasets
    df_cpi = load_macro("cpi")
    df_un = load_macro("unrate")
    df_gdp = load_macro("gdp")
    df_dff = load_macro("dff")
    df_vix = load_macro("vix")
    df_tnx = load_macro("tnx")
    df_dxy = load_macro("dxy")
    df_gold = load_macro("gold")
    df_oil = load_macro("oil")

    # Compute trend scores (0–100)
    scores = {
        "inflation_score": trend_score(df_cpi["cpi"]),
        "unemployment_score": trend_score(df_un["unrate"]),
        "gdp_score": trend_score(df_gdp["gdp"]),
        "interest_rate_score": trend_score(df_dff["dff"]),
        "vix_score": trend_score(df_vix["vix"]),
        "dxy_score": trend_score(df_dxy["dxy"]),
        "gold_score": trend_score(df_gold["gold"]),
        "oil_score": trend_score(df_oil["oil"])
    }

    # Composite score
    macro_score = round(np.mean(list(scores.values())), 2)
    scores["macro_composite"] = macro_score

    # Regime analysis
    regime_list = classify_macro_regime(
        scores["inflation_score"],
        scores["unemployment_score"],
        scores["gdp_score"],
        scores["interest_rate_score"],
        scores["vix_score"],
        scores["dxy_score"],
        scores["gold_score"],
        scores["oil_score"]
    )

    # Human-readable summary
    summary = generate_summary(regime_list)

    # Save macro scores (CSV)
    out_csv = os.path.join(DIRS["macro"], "macro_scores.csv")
    pd.DataFrame([scores]).to_csv(out_csv, index=False)

    # Save macro summary (JSON)
    out_json = os.path.join(DIRS["macro"], "macro_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "macro_scores": scores,
            "macro_regime": regime_list,
            "macro_summary_text": summary
        }, f, indent=4)

    print(f"[OK] Macro scores saved → {out_csv}")
    print(f"[OK] Macro summary saved → {out_json}")

    print("\n========== MACRO AGENT COMPLETE ==========\n")


if __name__ == "__main__":
    run_macro_agent()
