# ===========================================================
# Technical Indicators
# ===========================================================

import pandas as pd
import numpy as np

# ------------------ Indicator Functions ------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def compute_sma(series, window):
    return series.rolling(window).mean()

def compute_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def compute_momentum(series, period=10):
    return series.pct_change(period)

def compute_volatility(series, window=20):
    return series.pct_change().rolling(window).std()


# ------------------ ADD_INDICATORS WRAPPER ------------------

def add_indicators(df):
    """
    Adds all technical indicators to a DataFrame.
    """

    # --- SMA ---
    df["sma_10"] = compute_sma(df["close"], 10)
    df["sma_20"] = compute_sma(df["close"], 20)
    df["sma_50"] = compute_sma(df["close"], 50)

    # --- EMA ---
    df["ema_10"] = compute_ema(df["close"], 10)
    df["ema_20"] = compute_ema(df["close"], 20)
    df["ema_50"] = compute_ema(df["close"], 50)

    # --- RSI ---
    df["rsi_14"] = compute_rsi(df["close"], 14)

    # --- MACD ---
    df["macd"], df["macd_signal"] = compute_macd(df["close"])

    # --- Momentum ---
    df["momentum_10"] = compute_momentum(df["close"], 10)

    # --- Volatility ---
    df["volatility_20"] = compute_volatility(df["close"], 20)

    return df
