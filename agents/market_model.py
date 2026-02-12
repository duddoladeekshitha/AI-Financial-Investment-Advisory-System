# ===========================================================
# MODEL SUPPORT FUNCTIONS FOR TRAINING
# Auto-detects all technical + macro features
# ===========================================================

import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import DIRS


# -----------------------------------------------------------
# Add future return (target)
# -----------------------------------------------------------
def add_future_return(df, horizon):
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    return df.dropna()


# -----------------------------------------------------------
# Prepare X and y
# -----------------------------------------------------------
def prepare_xy(df):
    # Columns to EXCLUDE
    exclude_cols = [
        "date", "future_return",
        "open", "high", "low", "close", "volume"
    ]
    exclude_cols = [c for c in exclude_cols if c in df.columns]

    # Auto-select numeric features only
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df["future_return"].copy()

    # ------------- CLEANING FIXES -------------
    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN
    df_clean = pd.concat([X, y], axis=1).dropna()

    # Re-split into X and y
    y = df_clean["future_return"]
    X = df_clean.drop(columns=["future_return"])

    # Reset index
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y

