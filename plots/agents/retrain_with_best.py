# ===========================================================
# FINAL RETRAIN USING TUNED OPTUNA PARAMS
# ===========================================================

import os
import sys
import json
import joblib
import pandas as pd
import importlib.util
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Safe import
# -----------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS, SYMBOLS, FORECAST_HORIZONS

# Load market_model
mk_path = os.path.join(ROOT, "agents", "market_model.py")
spec_m = importlib.util.spec_from_file_location("market_model", mk_path)
market_model = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(market_model)


# -----------------------------------------------------------
# Load tuned params
# -----------------------------------------------------------
def load_best_params(symbol, horizon_label):
    path = os.path.join(DIRS["models"], f"{symbol}_{horizon_label}_best_params.json")

    if not os.path.exists(path):
        print(f"[WARN] No tuned params found for {symbol} {horizon_label}")
        return None

    with open(path, "r") as f:
        params = json.load(f)

    # Convert float to int where needed
    for key in ["n_estimators", "max_depth", "min_child_weight"]:
        if key in params:
            params[key] = int(params[key])

    return params


# -----------------------------------------------------------
# Train final model
# -----------------------------------------------------------
def train_final_model(X_train, y_train, params):
    if params is None:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    else:
        model = XGBRegressor(**params)

    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


# -----------------------------------------------------------
# Plot results
# -----------------------------------------------------------
def plot_results(symbol, horizon_label, y_test, y_pred):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(f"{symbol} – {horizon_label} Final Model")
    plt.legend()
    plt.grid(True)

    path = os.path.join(DIRS["evaluation"], f"{symbol}_{horizon_label}_final_plot.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[OK] Plot saved → {path}")


# -----------------------------------------------------------
# MAIN FINAL RETRAIN
# -----------------------------------------------------------
def run_final_retraining():

    print("\n========== FINAL RETRAINING (TUNED MODELS) ==========\n")

    results = []

    for name, symbol in SYMBOLS.items():
        feat_file = os.path.join(DIRS["features"], f"{symbol}_features.csv")
        if not os.path.exists(feat_file):
            print(f"[SKIP] Missing features → {symbol}")
            continue

        df = pd.read_csv(feat_file)

        for horizon_label, horizon in FORECAST_HORIZONS.items():

            df_h = market_model.add_future_return(df, horizon)
            X, y = market_model.prepare_xy(df_h)

            if len(X) < 200:
                print(f"[SKIP] Not enough data for {symbol} {horizon_label}")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            params = load_best_params(symbol, horizon_label)
            model = train_final_model(X_train, y_train, params)

            y_pred = model.predict(X_test)
            mae, rmse, r2 = evaluate(y_test, y_pred)

            print(f"\n>>> {symbol} | {horizon_label}")
            print(f"MAE : {mae:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"R²  : {r2:.6f}")

            plot_results(symbol, horizon_label, y_test, y_pred)

            # Save final model
            model_path = os.path.join(DIRS["models"], f"{symbol}_{horizon_label}_FINAL.pkl")
            joblib.dump(model, model_path)

            results.append([symbol, horizon_label, mae, rmse, r2])

    # Save overall metrics
    df_res = pd.DataFrame(results, columns=["Symbol", "Horizon", "MAE", "RMSE", "R2"])
    out_csv = os.path.join(DIRS["evaluation"], "final_metrics.csv")
    df_res.to_csv(out_csv, index=False)

    print("\n========== FINAL RETRAINING COMPLETE ==========\n")


if __name__ == "__main__":
    run_final_retraining()
