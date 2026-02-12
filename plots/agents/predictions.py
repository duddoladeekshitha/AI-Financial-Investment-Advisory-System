# ================================================================
# prediction_agent.py — Uses FINAL MODELS to generate predictions
# ================================================================

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS, SYMBOLS, FORECAST_HORIZONS

# -----------------------------------------------------------
# Load feature_engineering dynamically
# -----------------------------------------------------------
feat_path = os.path.join(ROOT, "utils", "feature_engineering.py")
spec_f = importlib.util.spec_from_file_location("feature_engineering", feat_path)
feature_engineering = importlib.util.module_from_spec(spec_f)
spec_f.loader.exec_module(feature_engineering)


# -----------------------------------------------------------
# Load model (FINAL → normal)
# -----------------------------------------------------------
def load_model(symbol, horizon_label):
    final_path = os.path.join(DIRS["models"], f"{symbol}_{horizon_label}_FINAL.pkl")
    base_path = os.path.join(DIRS["models"], f"{symbol}_{horizon_label}.pkl")

    if os.path.exists(final_path):
        print(f"[INFO] Using FINAL model → {final_path}")
        return joblib.load(final_path)

    if os.path.exists(base_path):
        print(f"[INFO] Using BASE model → {base_path}")
        return joblib.load(base_path)

    print(f"[WARN] Model not found → {symbol}_{horizon_label}")
    return None


# -----------------------------------------------------------
# Prepare features for prediction
# -----------------------------------------------------------
def load_latest_features(symbol):
    feat_file = os.path.join(DIRS["features"], f"{symbol}_features.csv")
    if not os.path.exists(feat_file):
        print(f"[WARN] No features found for {symbol}")
        return None
    df = pd.read_csv(feat_file)
    return df.iloc[-1:]  # last row → most recent day


# -----------------------------------------------------------
# Predict next-horizon returns
# -----------------------------------------------------------
def predict_for_symbol(symbol):
    print(f"\n[INFO] Predicting for {symbol}")

    last_row = load_latest_features(symbol)
    if last_row is None:
        return None

    symbol_preds = {"symbol": symbol}

    for horizon_label, horizon in FORECAST_HORIZONS.items():

        model = load_model(symbol, horizon_label)
        if model is None:
            symbol_preds[horizon_label] = None
            continue

        X = last_row[model.feature_names_in_]

        pred = model.predict(X)[0]
        symbol_preds[horizon_label] = float(pred)

    return symbol_preds


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------
def run_prediction_agent():

    print("\n========== PREDICTION AGENT STARTED ==========\n")

    predictions_list = []

    for name, symbol in SYMBOLS.items():
        preds = predict_for_symbol(symbol)
        if preds:
            predictions_list.append(preds)

    # Save latest predictions CSV
    latest_path = os.path.join(DIRS["predictions"], "latest_predictions.csv")
    pd.DataFrame(predictions_list).to_csv(latest_path, index=False)
    print(f"[OK] Saved latest predictions → {latest_path}")

    # Save historical predictions (append-style)
    hist_path = os.path.join(DIRS["predictions"], "historical_predictions.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            hist_data = json.load(f)
    else:
        hist_data = []

    hist_data.append(predictions_list)

    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=4)

    print(f"[OK] Saved historical predictions → {hist_path}")

    # Save combined predictions
    all_preds_path = os.path.join(DIRS["predictions"], "all_predictions.csv")
    pd.DataFrame(predictions_list).to_csv(all_preds_path, index=False)
    print(f"[OK] Saved combined predictions → all_predictions.csv")

    print("\n========== PREDICTION AGENT COMPLETE ==========\n")


if __name__ == "__main__":
    run_prediction_agent()
