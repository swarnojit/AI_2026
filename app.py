import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from data_utils import load_data, generate_rul
from model_rf import train, predict, health_score

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Jet Engine Predictive Maintenance",
    layout="wide"
)

st.title(" Jet Engine Predictive Maintenance Dashboard")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_all_data():
    train_df = load_data("data/train_FD001.txt")
    train_df = generate_rul(train_df)
    test_df = load_data("data/test_FD001.txt")
    return train_df, test_df

train_df, test_df = load_all_data()

# ------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------
train(train_df)

# ------------------------------------------------
# RMSE (for judge questions)
# ------------------------------------------------
@st.cache_data
def compute_rmse():
    preds = predict(train_df)
    return np.sqrt(mean_squared_error(train_df["RUL"], preds))

rmse_score = compute_rmse()

# ------------------------------------------------
# ENGINE SELECTION
# ------------------------------------------------
engine_id = st.selectbox(
    "Select Engine ID",
    sorted(test_df["engine_id"].unique())
)

engine_df = test_df[test_df["engine_id"] == engine_id].copy()
engine_df["Predicted_RUL"] = predict(engine_df)
engine_df["Health"] = engine_df["Predicted_RUL"].apply(health_score)

latest = engine_df.iloc[-1]

# ------------------------------------------------
# STATUS LOGIC
# ------------------------------------------------
if latest["Health"] > 70:
    status = "🟢 HEALTHY"
elif latest["Health"] > 30:
    status = "🟡 WARNING"
else:
    status = "🔴 CRITICAL"

# ------------------------------------------------
# METRICS
# ------------------------------------------------
st.subheader("📊 Engine Status")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL", f"{latest['Predicted_RUL']:.1f} cycles")
c2.metric("Health", f"{latest['Health']:.1f}%")
c3.metric("Condition", status)

# ------------------------------------------------
# HEALTH TREND WITH COLOR ZONES
# ------------------------------------------------
st.subheader("Health Degradation Over Time")

fig, ax = plt.subplots(figsize=(5, 1))

# Background zones
ax.axhspan(70, 100, color="green", alpha=0.15)
ax.axhspan(30, 70, color="yellow", alpha=0.15)
ax.axhspan(0, 30, color="red", alpha=0.15)

# Health curve
ax.plot(
    engine_df["cycle"],
    engine_df["Health"],
    color="blue",
    linewidth=2,
    label="Health (%)"
)

# Threshold lines
ax.axhline(70, linestyle="--", color="green")
ax.axhline(30, linestyle="--", color="red")

ax.set_xlabel("Cycle")
ax.set_ylabel("Health (%)")
ax.set_title(f"Engine {engine_id} Health Trend")
ax.legend()

st.pyplot(fig)

# ------------------------------------------------
# EXPLANATION (MODEL UNDERSTANDING)
# ------------------------------------------------
with st.expander(" Clarification"):
    st.markdown(f"""
**1. Data:** We use historical run-to-failure sensor data from NASA CMAPSS FD001.  
**2. Model:** A Random Forest regression model learns the relationship between sensor degradation and Remaining Useful Life (RUL).  
**3. Prediction:** The model predicts how many cycles remain before failure.  
**4. Health Score:** RUL is normalized into a 0–100% health score (clamped by design).  
**5. Decision Zones:**  
- 🟢 Green (>70%): Healthy  
- 🟡 Yellow (30–70%): Warning  
- 🔴 Red (<30%): Critical  

**RMSE:** {rmse_score:.2f}  
We prioritize explainability and stability over black-box complexity.
""")
