# ===========================================================
# MODEL TUNER (OPTUNA)
# ===========================================================

import os
import sys
import joblib
import json
import optuna
import pandas as pd
import importlib.util
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------
# Safe imports
# -----------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config import DIRS, SYMBOLS, FORECAST_HORIZONS

# Load market_model for prepare_xy(), add_future_return()
mk_path = os.path.join(ROOT, "agents", "market_model.py")
spec_m = importlib.util.spec_from_file_location("market_model", mk_path)
market_model = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(market_model)


# -----------------------------------------------------------
# Objective for Optuna
# -----------------------------------------------------------
def objective(trial, df, horizon):

    df_h = market_model.add_future_return(df, horizon)
    X, y = market_model.prepare_xy(df_h)

    if len(X) < 300:
        return 9999  # avoid crashes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return rmse


# -----------------------------------------------------------
# Main tuner
# -----------------------------------------------------------
def run_tuner():
    print("\n=========== RUNNING OPTUNA HYPERPARAMETER TUNING ===========\n")

    os.makedirs(DIRS["models"], exist_ok=True)

    for name, symbol in SYMBOLS.items():
        feat_file = os.path.join(DIRS["features"], f"{symbol}_features.csv")
        if not os.path.exists(feat_file):
            print(f"[SKIP] No feature file for {symbol}")
            continue

        df = pd.read_csv(feat_file)

        for horizon_label, horizon in FORECAST_HORIZONS.items():
            print(f"\n--- Tuning {symbol} | {horizon_label} ---")

            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, df, horizon), n_trials=25)

            best_params = study.best_params

            # Save best params
            out_path = os.path.join(DIRS["models"], f"{symbol}_{horizon_label}_best_params.json")
            with open(out_path, "w") as f:
                json.dump(best_params, f, indent=4)

            print(f"[OK] Saved tuned params → {out_path}")

    print("\n=========== TUNING COMPLETE ===========\n")


if __name__ == "__main__":
    run_tuner()
