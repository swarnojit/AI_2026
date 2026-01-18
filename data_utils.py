import pandas as pd

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)
    df.columns = COLUMNS
    return df

def generate_rul(df):
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df.drop(columns=["max_cycle"])
