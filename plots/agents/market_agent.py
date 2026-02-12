# ===========================================================
# MARKET AGENT — TRAINING + BASIC METRICS + OPTIONAL TUNING
# ===========================================================

import os
import sys
import pandas as pd
import joblib
import subprocess
import importlib.util
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import DIRS, SYMBOLS, FORECAST_HORIZONS

# Load market_model functions (future_return + prepare_xy)
model_path = os.path.join(ROOT_DIR, "agents", "market_model.py")
spec_m = importlib.util.spec_from_file_location("market_model", model_path)
market_model = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(market_model)


# -----------------------------------------------------------
# Basic model builder (default params)
# -----------------------------------------------------------
def build_default_model():
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


# -----------------------------------------------------------
# Evaluation 
# -----------------------------------------------------------
def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


# -----------------------------------------------------------
# MAIN TRAINING LOOP
# -----------------------------------------------------------
def run_market_agent():
    print("\n=========== MARKET AGENT (TRAINING ONLY) STARTED ===========\n")

    results = []

    for name, symbol in SYMBOLS.items():
        print(f"\n===== Training: {symbol} =====")

        feat_file = os.path.join(DIRS["features"], f"{symbol}_features.csv")
        if not os.path.exists(feat_file):
            print(f"[SKIP] No features for {symbol}")
            continue

        df = pd.read_csv(feat_file)

        for horizon_label, horizon in FORECAST_HORIZONS.items():

            print(f"\n--- Horizon {horizon_label} ({horizon} days) ---")

            df_h = market_model.add_future_return(df, horizon)

            X, y = market_model.prepare_xy(df_h)
            if len(X) < 200:
                print("[SKIP] Not enough data")
                continue

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Train default model
            model = build_default_model()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate
            mae, rmse, r2 = evaluate(y_test, y_pred)

            print(f"MAE  : {mae:.6f}")
            print(f"RMSE : {rmse:.6f}")
            print(f"R²   : {r2:.6f}")

            # Save model
            out_path = os.path.join(DIRS["models"], f"{symbol}_{horizon_label}_default.pkl")
            joblib.dump(model, out_path)

            # Save metrics
            results.append([symbol, horizon_label, mae, rmse, r2])

    # Save metrics table
    df_res = pd.DataFrame(results, columns=["Symbol", "Horizon", "MAE", "RMSE", "R2"])
    out_csv = os.path.join(DIRS["evaluation"], "default_model_metrics.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[OK] Metrics saved → {out_csv}")

    # -----------------------------------------------------------
    # Ask if tuning is needed
    # -----------------------------------------------------------
    choice = input("\nDo you want to run hyperparameter tuning? (y/n): ").strip().lower()

    if choice == "y":
        print("\n[INFO] Running model_tuner.py ... this may take time.\n")
        subprocess.run(["python", os.path.join("agents", "model_tuner.py")])
        print("\n[OK] Tuning complete. Best params saved.\n")
    else:
        print("\n[INFO] Skipping tuning.\n")

    print("\n=========== MARKET AGENT TRAINING COMPLETE ===========\n")


# Run agent
if __name__ == "__main__":
    run_market_agent()
