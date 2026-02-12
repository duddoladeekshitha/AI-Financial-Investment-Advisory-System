import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Root resolution
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS, SYMBOLS

plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")

# ============================================================
# NEW EDA DIRECTORY
# ============================================================

EDA_DIR = os.path.join(DIRS["raw"], "..", "eda_plots")
EDA_DIR = os.path.abspath(EDA_DIR)
os.makedirs(EDA_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(EDA_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved → {path}")


# ============================================================
# Load Indicators Dynamically
# ============================================================
import importlib.util

IND_PATH = os.path.join(ROOT, "utils", "indicators.py")
spec_ind = importlib.util.spec_from_file_location("indicators", IND_PATH)
indicators = importlib.util.module_from_spec(spec_ind)
spec_ind.loader.exec_module(indicators)


# ============================================================
# RAW LOADER
# ============================================================
def load_raw(symbol):
    path = os.path.join(DIRS["raw"], f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"[ERROR] Raw file missing → {symbol}")
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


# ============================================================
# MACRO LOADER
# ============================================================
def load_macro():
    macro_path = os.path.join(DIRS["macro"], "processed")

    if not os.path.exists(macro_path):
        print("[ERROR] No macro data folder found.")
        return pd.DataFrame()

    files = [f for f in os.listdir(macro_path) if f.endswith("_daily.csv")]
    frames = []

    for f in files:
        path = os.path.join(macro_path, f)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        key = f.replace("_daily.csv", "")
        df = df.rename(columns={df.columns[1]: key})
        df = df.set_index("date")
        frames.append(df)

    if not frames:
        print("[WARN] No macro files loaded.")
        return pd.DataFrame()

    return pd.concat(frames, axis=1).sort_index().ffill()


# ============================================================
# EDA PLOTS
# ============================================================

def eda_returns(df, symbol):
    df["returns"] = df["close"].pct_change()

    plt.figure(figsize=(10,5))
    sns.histplot(df["returns"].dropna(), bins=60, kde=True)
    plt.title(f"{symbol} – Daily Returns Distribution")
    plt.xlabel("Daily Returns")
    plt.ylabel("Frequency")

    save_fig(f"{symbol}_returns_distribution.png")


def eda_volatility(df, symbol, window=20):
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(window).std() * np.sqrt(252)

    plt.figure(figsize=(10,5))
    plt.plot(df["volatility"], label="Volatility")
    plt.title(f"{symbol} – {window}-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()

    save_fig(f"{symbol}_volatility.png")


def eda_sma_trends(df, symbol):
    df["SMA50"] = df["close"].rolling(50).mean()
    df["SMA200"] = df["close"].rolling(200).mean()

    plt.figure(figsize=(12,6))
    plt.plot(df["close"], label="Close", alpha=0.5)
    plt.plot(df["SMA50"], label="SMA50", linewidth=2)
    plt.plot(df["SMA200"], label="SMA200", linewidth=2)
    plt.title(f"{symbol} – SMA50 vs SMA200 Trend Regimes")
    plt.legend()

    save_fig(f"{symbol}_sma_crossover.png")


def eda_rsi_macd(df, symbol):
    df = indicators.add_indicators(df)

    fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=True)

    # RSI
    axes[0].plot(df["rsi_14"], color="purple")
    axes[0].axhline(70, linestyle="--", color="red")
    axes[0].axhline(30, linestyle="--", color="green")
    axes[0].set_title(f"{symbol} – RSI")

    # MACD
    axes[1].plot(df["macd"], label="MACD", color="blue")
    axes[1].plot(df["macd_signal"], label="Signal", color="orange")
    axes[1].set_title(f"{symbol} – MACD")
    axes[1].legend()

    plt.tight_layout()
    save_fig(f"{symbol}_rsi_macd.png")


def eda_macro_composite(df_macro):
    df_norm = df_macro.rank(pct=True)

    weights = {
        "cpi": 0.15, "unrate": 0.15, "gdp": 0.20, "dff": 0.10,
        "vix": 0.10, "tnx": 0.10, "dxy": 0.10, "gold": 0.05, "oil": 0.05
    }

    df_score = pd.DataFrame(index=df_macro.index)
    df_score["MacroScore"] = 0

    for col, w in weights.items():
        if col in df_norm:
            df_score["MacroScore"] += df_norm[col] * w

    plt.figure(figsize=(12,5))
    plt.plot(df_score["MacroScore"], linewidth=2)
    plt.title("Composite Macro Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Score")

    save_fig("macro_composite_score.png")


# ============================================================
# MASTER RUNNERS
# ============================================================

def run_eda(symbol):
    print(f"\n===== RUNNING EDA FOR {symbol} =====")
    df = load_raw(symbol)
    if df is None:
        return

    eda_returns(df.copy(), symbol)
    eda_volatility(df.copy(), symbol)
    eda_sma_trends(df.copy(), symbol)
    eda_rsi_macd(df.copy(), symbol)

    print(f"[DONE] {symbol} EDA Complete")


def run_macro_eda():
    print("\n===== RUNNING MACRO EDA =====")
    df_macro = load_macro()
    if df_macro.empty:
        print("[ERROR] No macro data found.")
        return

    eda_macro_composite(df_macro)
    print("[DONE] Macro EDA Complete")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Run EDA for all assets
    for _, symbol in SYMBOLS.items():
        run_eda(symbol)

    # Run EDA for Macro
    run_macro_eda()
