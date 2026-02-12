# ==============================================================
# recommendation_agent.py
# ==============================================================

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS


# -------------------------------------------------------------
# LOAD FUNCTIONS
# -------------------------------------------------------------
def load_predictions():
    path = os.path.join(DIRS["predictions"], "latest_predictions.csv")
    df = pd.read_csv(path)

    df = df.rename(columns={
        "1D": "pred_1D",
        "5D": "pred_5D",
        "1M": "pred_1M",
        "3M": "pred_3M",
        "6M": "pred_6M",
        "1Y": "pred_1Y",
        "3Y": "pred_3Y",
        "5Y": "pred_5Y",
    })
    return df


def load_risk():
    return pd.read_csv(os.path.join(DIRS["risk"], "risk_scores.csv"))


def load_macro_strength():
    df = pd.read_csv(os.path.join(DIRS["macro"], "macro_scores.csv"))
    return float(df["macro_composite"].iloc[0])


def load_portfolio_weights(horizon):
    # NEW: horizon = "1D", "5D", "1M", "1Y"
    fname = f"portfolio_{horizon}.csv"
    path = os.path.join(DIRS["portfolio"], fname)

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]  # normalize
        return df

    return pd.DataFrame(columns=["symbol", "weight"])


# -------------------------------------------------------------
# PREDICTION SELECTION BASED ON HORIZON
# -------------------------------------------------------------
def get_single_horizon_prediction(row, horizon):
    # horizon ∈ ["1D", "5D", "1M", "1Y"]
    col = f"pred_{horizon}"
    return float(row[col]) if col in row else 0.0


# -------------------------------------------------------------
# ACTION CLASSIFICATION
# -------------------------------------------------------------
def classify_action(pred, risk_label, user_risk, macro):
    risk_level = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    user_level = {"Conservative": 1, "Moderate": 2, "Aggressive": 3}

    risk_gap = risk_level[risk_label] - user_level[user_risk]

    if pred > 0 and risk_gap <= 0 and macro >= 50:
        return "BUY"
    if pred < 0 or risk_gap > 1:
        return "SELL"
    return "HOLD"


# -------------------------------------------------------------
# REASONING BLOCKS
# -------------------------------------------------------------
def prediction_reasoning(pred, horizon):
    if pred > 0:
        return f"The {horizon} forecast indicates upward momentum."
    if pred < 0:
        return f"The {horizon} forecast suggests downward pressure."
    return f"The {horizon} forecast is neutral."


def risk_reasoning(label):
    return {
        "LOW": "Asset exhibits low volatility and strong stability.",
        "MEDIUM": "Moderate volatility — suitable for most investors.",
        "HIGH": "High volatility — suitable primarily for aggressive investors."
    }[label]


def macro_reasoning(macro):
    if macro >= 55:
        return "Macro environment supports growth."
    if macro <= 45:
        return "Macro trends reflect uncertainty."
    return "Macro conditions are neutral."


def user_alignment_reasoning(asset_risk, user_risk):
    if user_risk == "Conservative" and asset_risk == "HIGH":
        return "Risk exceeds your conservative tolerance."
    if user_risk == "Aggressive" and asset_risk == "LOW":
        return "Very stable asset — may not match aggressive growth goals."
    return "Risk level aligns with your profile."


def portfolio_reasoning(symbol, weights):
    if weights.empty:
        return "Portfolio optimization unavailable for this horizon."

    if "symbol" not in weights.columns:
        return "Portfolio weights missing symbol column."

    # safe lookup
    w = weights.loc[weights["symbol"] == symbol]
    if w.empty:
        return "This asset is not part of your optimized allocation."
    
    weight = float(w["weight"].iloc[0]) * 100
    return f"Portfolio optimization assigns {weight:.2f}% weight to this asset."


# -------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------
def build_final_summary(action, pred, risk, macro):
    base = {
        "BUY": "Signals indicate a favorable buying opportunity.",
        "HOLD": "Mixed signals — holding is recommended.",
        "SELL": "Indicators suggest risk outweighs reward at this time."
    }[action]

    confidence = 50
    if pred > 0: confidence += 15
    if macro > 50: confidence += 15
    if risk == "LOW": confidence += 10
    if risk == "HIGH": confidence -= 10

    return f"{base} (Confidence: {confidence}/100)"


# -------------------------------------------------------------
# MAIN RECOMMENDATION BUILDER
# -------------------------------------------------------------
def build_recommendations(user_profile):

    preds = load_predictions()
    risks = load_risk()
    macro = load_macro_strength()
    weights = load_portfolio_weights(user_profile["horizon"])

    df = preds.merge(risks, on="symbol", how="left")

    rows = []

    for _, row in df.iterrows():

        pred = get_single_horizon_prediction(row, user_profile["horizon"])
        action = classify_action(pred, row["risk_label"], user_profile["risk_level"], macro)

        rows.append([
            row["symbol"],
            pred,
            row["risk_label"],
            portfolio_reasoning(row["symbol"], weights),
            prediction_reasoning(pred, user_profile["horizon"]),
            risk_reasoning(row["risk_label"]),
            macro_reasoning(macro),
            user_alignment_reasoning(row["risk_label"], user_profile["risk_level"]),
            action,
            build_final_summary(action, pred, row["risk_label"], macro)
        ])

    df_out = pd.DataFrame(rows, columns=[
        "symbol",
        "prediction",
        "risk_label",
        "portfolio_reason",
        "prediction_reason",
        "risk_reason",
        "macro_reason",
        "user_alignment_reason",
        "action",
        "final_summary"
    ])

    out_path = os.path.join(DIRS["portfolio"], "personalized_recommendations.csv")
    df_out.to_csv(out_path, index=False)

    print(f"[OK] Saved personalized recommendations → {out_path}")
    return df_out, out_path


# -------------------------------------------------------------
# run() WRAPPER
# -------------------------------------------------------------
def run(user_profile):
    return build_recommendations(user_profile)


# -------------------------------------------------------------
# EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    profile = {
        "age": 25,
        "risk_level": "Moderate",
        "horizon": "1M",
        "goals": "Wealth Growth",
    }
    run(profile)
