# Jet Engine Predictive Maintenance

Predict Remaining Useful Life (RUL) and health status of jet engines using NASA CMAPSS FD001 dataset.

## 🎯 Problem
Predict engine failure and health percentage from multivariate sensor data for proactive maintenance.

## 📊 Dataset
**NASA CMAPSS FD001**
- Run-to-failure training data
- 3 operational settings + 21 sensor readings
- Engine cycles until failure

## 🔧 Approach
- **Model**: Random Forest Regressor
- **Output**: RUL (cycles) → Health % (0-100)
- **Why RF?** Handles noise, non-linear patterns, explainable

## 📈 Health Zones
- 🟢 **>70%** — Healthy
- 🟡 **30-70%** — Warning  
- 🔴 **<30%** — Critical

## 🚀 Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure
```
jet_engine_pm/
├── app.py              # Streamlit dashboard
├── model.py            # RF model training
├── data_utils.py       # Data preprocessing
├── requirements.txt
└── data/
```

## 📊 Performance
- **RMSE**: ~20 cycles
- Explainable, no deep learning
- Real-time predictions

## 🎥 Features
- Engine selector
- RUL prediction
- Health percentage & status
- Degradation trend visualization

---
Built for hackathon demos | Fully interpretable pipeline
