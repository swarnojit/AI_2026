import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

scaler = MinMaxScaler()

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

def train(df):
    X = scaler.fit_transform(df[SENSORS])
    y = df["RUL"]
    model.fit(X, y)

def predict(df):
    X = scaler.transform(df[SENSORS])
    return model.predict(X)

def health_score(rul, max_rul=130):
    health = (rul / max_rul) * 100
    return np.clip(health, 0, 100)
