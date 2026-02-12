# ================================================================
#Portfolio Engine
# ================================================================

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS

# ----------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------
def load_predictions():
    df = pd.read_csv(os.path.join(DIRS["predictions"], "latest_predictions.csv"))
    df = df.rename(columns={
        "1D": "pred_1D", "5D": "pred_5D", "1M": "pred_1M",
        "3M": "pred_3M", "6M": "pred_6M", "1Y": "pred_1Y",
        "3Y": "pred_3Y", "5Y": "pred_5Y"
    })
    return df

def load_risk():
    return pd.read_csv(os.path.join(DIRS["risk"], "risk_scores.csv"))

def load_macro():
    return pd.read_csv(os.path.join(DIRS["macro"], "macro_scores.csv"))

# ----------------------------------------------------------------
# Weight Parameters by Risk Type
# ----------------------------------------------------------------
def get_weight_params(risk_type):
    if risk_type == "Conservative":
        return 0.20, 0.60, 0.15, 0.05
    elif risk_type == "Moderate":
        return 0.35, 0.35, 0.20, 0.10
    else:
        return 0.60, 0.20, 0.15, 0.05  # Aggressive

# ----------------------------------------------------------------
# MAIN PORTFOLIO BUILDER
# ----------------------------------------------------------------
def build_portfolio(user_profile):
    preds = load_predictions()
    risks = load_risk()
    macro = load_macro()

    horizon = user_profile["horizon"]
    pred_col = f"pred_{horizon}"

    α, β, γ, δ = get_weight_params(user_profile["risk_level"])

    macro_score = float(macro["macro_composite"].iloc[0]) / 100

    df = preds.merge(risks, on="symbol", how="left")

    # Forecast score = selected horizon only
    df["forecast_score"] = df[pred_col]

    # Remove negative forecast assets
    df = df[df["forecast_score"] > 0]

    df["risk_norm"] = df["final_risk_score"] / 100

    # User age → aggressiveness
    age = user_profile["age"]
    if age < 30: 
        U = 1
    elif age < 50: 
        U = 0.7
    else: 
        U = 0.4

    # Raw weight
    df["raw_weight"] = (
        α * df["forecast_score"] +
        β * (1 - df["risk_norm"]) +
        γ * macro_score +
        δ * U
    )

    # Remove noise assets
    df = df[df["raw_weight"] > df["raw_weight"].sum() * 0.05]

    # Limit assets
    top_n = 6 if user_profile["risk_level"] == "Conservative" else \
            10 if user_profile["risk_level"] == "Moderate" else 12

    df = df.nlargest(top_n, "raw_weight")

    # Final normalization
    df["weight"] = df["raw_weight"] / df["raw_weight"].sum()

    # Save output
    out_path = os.path.join(DIRS["portfolio"], "user_portfolio.csv")
    df.to_csv(out_path, index=False)

    return df, out_path

# ----------------------------------------------------------------
# run() wrapper function for UI and pipeline
# ----------------------------------------------------------------
def run(user_profile):
    """
    Wrapper function:
    - Builds portfolio for given user_profile dict
    - Returns (portfolio_df, file_path)
    """
    portfolio_df, file_path = build_portfolio(user_profile)
    return portfolio_df, file_path


# ----------------------------------------------------------------
# MAIN EXECUTION SUPPORT
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("========== PORTFOLIO AGENT (MANUAL RUN) ==========")

    # Example default profile for terminal testing
    example_user = {
        "age": 25,
        "income": 50000,
        "goal": "Wealth Growth",
        "risk_level": "Moderate",
        "horizon": "1M"
    }

    df, path = run(example_user)

    print(df)
    print(f"\nSaved → {path}")
    print("========== DONE ==========")
