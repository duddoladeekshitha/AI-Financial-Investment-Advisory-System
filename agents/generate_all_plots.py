import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

DATA_DIR = os.path.join(ROOT, "data")
PRED_DIR = os.path.join(DATA_DIR, "predictions")
RISK_DIR = os.path.join(DATA_DIR, "risk")
MACRO_DIR = os.path.join(DATA_DIR, "macro", "processed")
PORT_DIR = os.path.join(DATA_DIR, "portfolio_outputs")

PLOT_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ================================================================
# FIGURE 1 — Multi-Horizon Forecast Plot
# ================================================================
def plot_multi_horizon(symbol="AAPL"):

    df = pd.read_csv(os.path.join(PRED_DIR, "all_predictions.csv"))
    asset_df = df[df["symbol"] == symbol].iloc[0]

    horizons = ["1D","5D","1M","3M","6M","1Y","3Y","5Y"]
    values = [asset_df[h] for h in horizons]

    plt.figure(figsize=(8,5))
    plt.plot(horizons, values, marker='o')
    plt.title(f"Multi-Horizon Forecast for {symbol}")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Predicted Return")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{symbol}_forecast_horizon.png")
    plt.savefig(out_path, dpi=300)
    print(f"[SAVED] {out_path}")
    plt.close()


# ================================================================
# FIGURE 2 — Risk Score Visualization
# ================================================================
def plot_risk_scores(symbol="AAPL"):

    risk = pd.read_csv(os.path.join(RISK_DIR, "risk_scores.csv"))
    row = risk[risk["symbol"] == symbol].iloc[0]

    # AUTO-DETECT columns present
    available_cols = set(row.index)

    candidate_metrics = [
        "volatility",
        "max_drawdown",
        "drawdown",
        "value_at_risk",
        "var_95",
        "beta",
        "risk_score"
    ]

    # use only existing columns
    metrics = [c for c in candidate_metrics if c in available_cols]
    values = [row[m] for m in metrics]

    plt.figure(figsize=(8,5))
    plt.bar(metrics, values)
    plt.title(f"Risk Profile for {symbol}")
    plt.ylabel("Value / Score")
    plt.xticks(rotation=25)
    plt.grid(axis="y")
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{symbol}_risk_profile.png")
    plt.savefig(out_path, dpi=300)
    print(f"[SAVED] {out_path}")
    plt.close()


# ================================================================
# FIGURE 3 — Macro Trends
# ================================================================
def plot_macro_indicators():
    
    # Load macro scores
    import json

    json_path = os.path.join(MACRO_DIR, "..", "macro_summary.json")
    csv_path = os.path.join(MACRO_DIR, "..", "macro_scores.csv")

    with open(json_path, "r") as f:
        macro_json = json.load(f)

    macro_df = pd.read_csv(csv_path)

    # Extract scores
    scores = macro_json["macro_scores"]
    labels = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(10,6))
    plt.bar(labels, values)
    plt.ylabel("Score (0–100)")
    plt.title("Macro-Economic Composite Scores")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, "macro_scores_plot.png")
    plt.savefig(out_path, dpi=300)
    print(f"[SAVED] {out_path}")
    plt.close()

    # Print regime + summary
    print("\n=== MACRO REGIME ===")
    for r in macro_json["macro_regime"]:
        print("-", r)

    print("\n=== MACRO SUMMARY TEXT ===")
    for t in macro_json["macro_summary_text"]:
        print("-", t)


# ================================================================
# FIGURE 4 — Portfolio Allocation Plot
# ================================================================
def plot_portfolio(filename, title_name):

    csv_path = os.path.join(PORT_DIR, filename)

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(7, 7))
    plt.pie(
        df["weight"],
        labels=df["symbol"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 0.7},
        textprops={"fontsize": 10}
    )

    plt.title(title_name, fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{title_name.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=300)
    print(f"[SAVED] {out_path}")
    plt.close()



# ================================================================
# RUN ALL
# ================================================================
if __name__ == "__main__":

    print("Generating All  Figures...")

    plot_multi_horizon("AAPL")
    plot_risk_scores("AAPL")
    plot_macro_indicators()

    for level, title in [
        ("conservative_portfolio.csv", "Conservative Portfolio Allocation"),
        ("moderate_portfolio.csv", "Moderate Portfolio Allocation"),
        ("aggressive_portfolio.csv", "Aggressive Portfolio Allocation"),
        ("short_term_portfolio.csv", "Short-Term Portfolio Allocation"),
        ("mid_term_portfolio.csv", "Mid-Term Portfolio Allocation"),
        ("long_term_portfolio.csv", "Long-Term Portfolio Allocation")
    ]:
        plot_portfolio(level, title)


    print("All Figures Generated Successfully!")
