# AI_2026
Jet Engine Predictive Maintenance

NASA CMAPSS FD001 | RUL Prediction

Overview

This project implements a predictive maintenance system for jet engines using the NASA CMAPSS FD001 dataset.
The system predicts Remaining Useful Life (RUL), converts it into a Health Percentage, and visualizes engine degradation through an interactive dashboard.

Problem

Given multivariate sensor data from aircraft engines:

Predict Remaining Useful Life (RUL)

Convert RUL into a Health Score (0–100%)

Visualize engine health for maintenance decision-making

Dataset

NASA CMAPSS – FD001

train_FD001.txt: Run-to-failure training data

test_FD001.txt: Test data without failure labels

Features include:

Engine ID, operating cycles

3 operational settings

21 sensor readings

Approach

Model: Random Forest Regressor

Task: RUL regression (not classification)

Why Random Forest?

Handles noisy sensor data

Captures non-linear degradation

Explainable and stable for live demos

Health Score
Health (%) = (Predicted RUL / Max RUL) × 100


Clamped between 0–100.

Health Zones

🟢 >70% — Healthy

🟡 30–70% — Warning

🔴 <30% — Critical

Dashboard (Streamlit)

Engine ID selector

Predicted RUL

Health percentage

Condition status (Green/Yellow/Red)

Health degradation trend over time

Evaluation

Metric: RMSE (cycles)

Achieves ~20 RMSE, suitable for FD001 with explainable models

Project Structure
jet_engine_pm/
├── app.py
├── model.py
├── data_utils.py
├── requirements.txt
└── data/

Run Locally
pip install -r requirements.txt
streamlit run app.py

Notes

No deep learning used

Fully explainable pipeline

Designed for hackathon demos and real-time interpretation
