# ===========================================================
# data_loader.py — RAW DOWNLOADER + CLEANER
# ===========================================================

import os
import sys
import pandas as pd
import yfinance as yf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from config import DIRS, SYMBOLS


def get_period(symbol):
    index_symbols = ["^GSPC", "^DJI", "^IXIC", "^NSEI"]
    return "max" if symbol in index_symbols else "30y"


def load_and_save_raw(symbol):
    period = get_period(symbol)
    print(f"[INFO] Downloading {symbol} ({period})")

    df = yf.download(
        symbol,
        period=period,
        progress=False,
        auto_adjust=False,
        threads=True
    )


    if df is None or df.empty:
        print(f"[ERROR] No data downloaded for {symbol}")
        return None

    df = df.reset_index()

    # --- FIX FOR MULTI-INDEX COLUMNS ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()


    out_path = os.path.join(DIRS["raw"], f"{symbol}.csv")
    df.to_csv(out_path, index=False)

    print(f"[OK] RAW saved → {out_path}")
    return df


def load_all_raw():
    print("\n=========== DOWNLOADING RAW ===========")
    for name, symbol in SYMBOLS.items():
        load_and_save_raw(symbol)
    print("=========== DONE ===========\n")
    
if __name__ == "__main__":
    load_all_raw()

