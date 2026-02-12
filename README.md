# AI-Driven Financial Investment Advisory System

## Overview
Multi-agent AI-driven financial advisory system integrating market forecasting, macro-economic analysis, risk scoring, and personalized portfolio allocation.

This project is an AI-powered financial advisory system designed to generate data-driven investment recommendations instead of relying on guesswork or isolated indicators.

The system combines:

- Market price trends
- Macroeconomic indicators
- Risk analysis
- Machine learning forecasting
- Personalized portfolio logic

It produces clear:

- BUY / HOLD / SELL recommendations  
- Risk scores (0–100)  
- Personalized investment portfolios  

---

## Problem Statement

Financial markets are influenced by:

- Inflation
- GDP growth
- Interest rate changes
- Market volatility
- Currency strength
- Commodity movements

Retail investors often rely on fragmented information or opinion-based advice.

This project builds a structured AI framework that integrates multiple financial signals into a unified, explainable decision system.

---

## How the System Works (Simple Explanation)

The system behaves like a team of financial experts:

###  Market Agent
Analyzes stock price trends using technical indicators like:
- RSI
- MACD
- Moving averages
- Momentum
- Volatility

It also performs multi-horizon forecasting using XGBoost.

---

###  Macro Agent
Evaluates broader economic conditions using:
- CPI
- GDP
- Unemployment
- Interest rates
- VIX
- Dollar index (DXY)
- Gold and oil signals

This helps understand long-term economic cycles.

---

###  Risk Analysis Agent
Measures asset risk using:
- Volatility
- Maximum drawdown
- Beta
- Value-at-Risk (VaR)

Generates a normalized risk score (0–100).

---

###  Portfolio Recommendation Agent
Combines all signals using an adaptive weighted scoring model:

Total Score = Forecast + Macro + Risk + Portfolio Fit

Based on user risk preference, the system generates:

- Conservative portfolio
- Balanced portfolio
- Aggressive portfolio

---

## Machine Learning Component

- Model Used: XGBoost
- Multi-horizon forecasting (short-term to long-term)
- Time-series aware validation
- Hyperparameter tuning for stability
- Evaluated using MAE, RMSE, and R²

Model tuning improved forecast robustness across different market horizons.

---

## Key Features

 Multi-agent modular architecture  
 Explainable decision framework  
 Risk-aware recommendations  
 Adaptive portfolio personalization  
 Economic + technical signal integration  
 Interactive dashboard built with Streamlit  

---

## Example Outputs

- BUY / HOLD / SELL signal for individual assets  
- Risk score with supporting metrics  
- Portfolio allocation adjusted by user risk tolerance  

---

## Tech Stack

- Python  
- XGBoost  
- Pandas & NumPy  
- Financial Technical Indicators  
- Streamlit  
- Matplotlib / Visualization tools  


