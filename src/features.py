import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def log_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    # log1p so we don't blow up on zeros
    df = df.copy()
    for col in cols:
        df[col] = np.log1p(df[col])
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # rooms and bedrooms per household are more useful than raw totals
    df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"]
    df["population_per_household"] = df["Population"] / df["AveOccup"]
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # fit on train only — never on test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_skewed_cols(df: pd.DataFrame, threshold: float = 0.75) -> list:
    # returns column names where skewness exceeds the threshold
    numeric = df.select_dtypes(include=[np.number])
    skewness = numeric.apply(lambda x: x.skew()).abs()
    return skewness[skewness > threshold].index.tolist()
