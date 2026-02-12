# ============================================================
# CONFIGURATION FILE — AI-DRIVEN FINANCIAL ADVISORY SYSTEM
# Central configuration for directories, assets, settings.
# ============================================================

import os

# ------------------------------------------------------------
# PROJECT ROOT DIRECTORY
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# FOLDER STRUCTURE
# ------------------------------------------------------------
DIRS = {
    "raw": os.path.join(BASE_DIR, "data", "raw"),
    "macro": os.path.join(BASE_DIR, "data", "macro"),
    "features": os.path.join(BASE_DIR, "data", "features"),
    "models": os.path.join(BASE_DIR, "models"),
    "predictions": os.path.join(BASE_DIR, "data", "predictions"),
    "risk": os.path.join(BASE_DIR, "data", "risk"),
    "evaluation": os.path.join(BASE_DIR, "data", "evaluation"),
    "portfolio": os.path.join(BASE_DIR, "data", "portfolio_outputs"),
}

# Create directories
for folder in DIRS.values():
    os.makedirs(folder, exist_ok=True)

RAW_DIR = DIRS["raw"]
FEATURE_DIR = DIRS["features"]
MODEL_DIR = DIRS["models"]
PRED_DIR = DIRS["predictions"]
RISK_DIR = DIRS["risk"]
PORTFOLIO_DIR = DIRS["portfolio"]

# ------------------------------------------------------------
# ASSET UNIVERSE
# ------------------------------------------------------------
SYMBOLS = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "NASDAQ": "^IXIC",
    "NIFTY 50": "^NSEI",

    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",

    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",

    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOG",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Meta": "META",
}

# ------------------------------------------------------------
# MULTI-HORIZON FORECAST WINDOWS (Trading Days)
# ------------------------------------------------------------
FORECAST_HORIZONS = {
    "1D": 1,
    "5D": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 756,
    "5Y": 1260,
}

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ------------------------------------------------------------
# INDICATOR DEFAULTS
# ------------------------------------------------------------
INDICATOR_SETTINGS = {
    "rsi_period": 14,
    "momentum_period": 10,
    "sma_periods": [20, 50, 100, 200],
    "ema_periods": [20, 50, 100, 200],
    "stoch_k": 14,
    "stoch_d": 3,
    "cci_period": 20,
    "atr_period": 14,
    "adx_period": 14,
    "volatility_window": 20,
}

# ------------------------------------------------------------
# RISK SETTINGS
# ------------------------------------------------------------
RISK_FREE_RATE = 0.03 / 252
RISK_LABELS = {
    (0, 25): "LOW RISK",
    (25, 50): "MODERATE RISK",
    (50, 75): "HIGH RISK",
    (75, 101): "VERY HIGH RISK",
}

# ------------------------------------------------------------
# PORTFOLIO SETTINGS
# ------------------------------------------------------------
MAX_PORTFOLIO_ASSETS = 10
PORTFOLIO_CONSTRAINTS = {
    "min_weight": 0.0,
    "max_weight": 0.30,
}

# ------------------------------------------------------------
# UTILITY
# ------------------------------------------------------------
def print_config_summary():
    print("\n========== CONFIG SUMMARY ==========")
    print(f"Assets Loaded     : {len(SYMBOLS)}")
    print(f"Forecast Horizons : {list(FORECAST_HORIZONS.keys())}")
    print("Directories:")
    for k, v in DIRS.items():
        print(f"  - {k}: {v}")
    print("====================================\n")
