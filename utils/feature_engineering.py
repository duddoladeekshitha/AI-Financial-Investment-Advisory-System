# ====================================================================
# feature_engineering.py
# Combines:
#   - Technical indicators 
#   - Price-based features
#   - Macro indicators (daily)
# Outputs final enriched feature set for each symbol.
# ====================================================================

import os
import sys
import pandas as pd
import numpy as np
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS, SYMBOLS

IND_PATH = os.path.join(ROOT, "utils", "indicators.py")
spec_ind = importlib.util.spec_from_file_location("indicators", IND_PATH)
indicators = importlib.util.module_from_spec(spec_ind)
spec_ind.loader.exec_module(indicators)


# -----------------------------------------------------------
# Load ALL macro indicators (clean + aligned)
# -----------------------------------------------------------
def load_all_macro_daily():

    macro_path = os.path.join(DIRS["macro"], "processed")

    macro_files = {
        "cpi": "cpi_daily.csv",
        "unrate": "unrate_daily.csv",
        "gdp": "gdp_daily.csv",
        "dff": "dff_daily.csv",
        "vix": "vix_daily.csv",
        "tnx": "tnx_daily.csv",
        "dxy": "dxy_daily.csv",
        "gold": "gold_daily.csv",
        "oil": "oil_daily.csv",
    }

    df_macro = None

    for key, filename in macro_files.items():
        path = os.path.join(macro_path, filename)
        if not os.path.exists(path):
            print(f"[WARN] Macro file missing: {path}")
            continue

        df = pd.read_csv(path)

        # Convert date safely
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Identify macro column name (anything except date)
        valcol = [c for c in df.columns if c != "date"][0]

        # Convert values to numeric
        df[valcol] = pd.to_numeric(df[valcol], errors="coerce")

        df = df.dropna(subset=["date", valcol])
        df = df.set_index("date")

        if df_macro is None:
            df_macro = df[[valcol]].rename(columns={valcol: key})
        else:
            df_macro = df_macro.join(df[[valcol]].rename(columns={valcol: key}), how="outer")

    if df_macro is None:
        print("[WARN] No macro data loaded. Returning empty.")
        return pd.DataFrame(columns=["date"])

    df_macro = df_macro.sort_index().ffill().reset_index()
    return df_macro


# -----------------------------------------------------------
# FEATURE BUILDER
# -----------------------------------------------------------
def build_features_for_symbol(symbol):

    raw_path = os.path.join(DIRS["raw"], f"{symbol}.csv")
    if not os.path.exists(raw_path):
        print(f"[SKIP] No raw file for {symbol}")
        return

    df = pd.read_csv(raw_path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if "date" not in df.columns:
        print(f"[SKIP] No date column for {symbol}")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # -------------------------------
    # Add Technical Indicators
    # -------------------------------
    df = indicators.add_indicators(df)

    # -------------------------------
    # Price-based features
    # -------------------------------
    df["return_1d"]   = df["close"].pct_change(1)
    df["return_5d"]   = df["close"].pct_change(5)
    df["return_10d"]  = df["close"].pct_change(10)

    df["log_return"]  = np.log(df["close"] / df["close"].shift(1))

    # Price lag features
    df["close_lag1"]  = df["close"].shift(1)
    df["close_lag2"]  = df["close"].shift(2)
    df["close_lag5"]  = df["close"].shift(5)
    df["close_lag10"] = df["close"].shift(10)

    # Rolling stats
    df["rolling_mean_20"] = df["close"].rolling(20).mean()
    df["rolling_std_20"]  = df["close"].rolling(20).std()
    df["rolling_vol_20"]  = df["return_1d"].rolling(20).std()

    # Change in volume
    df["volume_change"] = df["volume"].pct_change()

    # -------------------------------
    # Load & merge macro data
    # -------------------------------
    df_macro = load_all_macro_daily()

    if not df_macro.empty:
        df_merged = df.merge(df_macro, on="date", how="left")
        df_merged = df_merged.ffill()
    else:
        df_merged = df.copy()

    # -------------------------------
    # Cleanup
    # -------------------------------
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
    df_merged = df_merged.dropna().reset_index(drop=True)

    # -------------------------------
    # SAVE FINAL FEATURES
    # -------------------------------
    out_path = os.path.join(DIRS["features"], f"{symbol}_features.csv")
    df_merged.to_csv(out_path, index=False)

    print(f"[OK] FEATURES saved → {out_path}")


# -----------------------------------------------------------
# RUN ALL
# -----------------------------------------------------------
def run_feature_engineering():

    print("\n=========== BUILDING FINAL FEATURE SETS ===========\n")

    for name, symbol in SYMBOLS.items():
        print(f"[INFO] Building features for → {symbol}")
        build_features_for_symbol(symbol)

    print("\n=========== FEATURE ENGINEERING COMPLETE ===========\n")


if __name__ == "__main__":
    run_feature_engineering()
