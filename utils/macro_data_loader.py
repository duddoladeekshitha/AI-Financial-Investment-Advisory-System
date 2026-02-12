# ========================================================================
# macro_data_loader.py
# Loads macro data from:
#   - FRED (downloaded CSVs)
#   - Yahoo Finance (VIX, TNX, DXY, GOLD, OIL)
# Converts everything to daily frequency, forward-fills missing values,
# and saves clean processed macro data for model training.
# ========================================================================

import os
import sys
import pandas as pd
import yfinance as yf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS

FRED_DIR = os.path.join(DIRS["macro"], "fred_raw")
YAHOO_DIR = os.path.join(DIRS["macro"], "yahoo_raw")
OUT_DIR = os.path.join(DIRS["macro"], "processed")

os.makedirs(YAHOO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ===============================================================
# Load FRED CSV + convert to daily frequency
# ===============================================================
def load_fred(filename, new_colname):
    path = os.path.join(FRED_DIR, filename)
    df = pd.read_csv(path)

    # Identify date column automatically
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    df.rename(columns={date_col: "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={df.columns[1]: new_colname}, inplace=True)

    # Convert to daily frequency
    df = df.set_index("date").resample("D").ffill().reset_index()

    return df


# ===============================================================
# Download Yahoo macro indicators
# ===============================================================
def load_yahoo(ticker, new_colname):
    print(f"[INFO] Downloading {ticker}...")
    df = yf.download(ticker, period="30y", interval="1d", progress=False)

    if df.empty:
        print(f"[WARN] Download failed for {ticker}")
        return None

    df = df.reset_index()[["Date", "Close"]]
    df.rename(columns={"Date": "date", "Close": new_colname}, inplace=True)

    # Save raw Yahoo file
    raw_path = os.path.join(YAHOO_DIR, f"{new_colname}.csv")
    df.to_csv(raw_path, index=False)
    print(f"[OK] Saved Yahoo raw → {raw_path}")

    # Convert to daily frequency
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("D").ffill().reset_index()

    return df


# ===============================================================
# Main Runner
# ===============================================================
def run_macro_loader():
    print("\n================= BUILDING DAILY MACRO DATA =================\n")

    final_frames = []

    # ---- FRED DATA ----
    fred_sources = [
        ("CPIAUCSL.csv", "cpi"),
        ("UNRATE.csv", "unrate"),
        ("GDPC1.csv", "gdp"),
        ("DFF.csv", "dff"),
    ]

    for file, col in fred_sources:
        print(f"[INFO] Loading FRED {col.upper()}...")
        df = load_fred(file, col)
        out_path = os.path.join(OUT_DIR, f"{col}_daily.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved → {out_path}\n")
        final_frames.append(df)

    # ---- YAHOO FINANCE DATA ----
    yahoo_sources = [
        ("^VIX", "vix"),
        ("^TNX", "tnx"),
        ("DX-Y.NYB", "dxy"),
        ("GC=F", "gold"),
        ("CL=F", "oil"),
    ]

    for ticker, col in yahoo_sources:
        df = load_yahoo(ticker, col)
        if df is not None:
            out_path = os.path.join(OUT_DIR, f"{col}_daily.csv")
            df.to_csv(out_path, index=False)
            print(f"[OK] Saved → {out_path}\n")
            final_frames.append(df)

    print("\n================= MACRO DATA PROCESSING COMPLETE =================\n")


if __name__ == "__main__":
    run_macro_loader()
