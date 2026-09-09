import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing


def load_california():
    # fetch from sklearn — no download needed
    raw = fetch_california_housing(as_frame=True)
    df = raw.frame.copy()
    df.rename(columns={"MedHouseVal": "price"}, inplace=True)
    return df


def load_ames(path: str):
    # expects the raw train.csv from Kaggle
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df
