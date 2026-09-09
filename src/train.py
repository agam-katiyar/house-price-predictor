import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import joblib


def train_model(X_train, y_train, model_type: str = "linear", **kwargs):
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(**kwargs),
        "lasso": Lasso(**kwargs),
        "xgboost": XGBRegressor(random_state=42, verbosity=0, **kwargs),
    }

    if model_type not in models:
        raise ValueError(f"unknown model_type '{model_type}'. pick from {list(models.keys())}")

    model = models[model_type]
    model.fit(X_train, y_train)
    return model


def cross_val_rmse(model, X, y, cv: int = 5) -> float:
    # negative MSE is sklearn convention — flip sign before sqrt
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    return float(np.sqrt(-scores.mean()))


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"saved → {path}")


def load_model(path: str):
    return joblib.load(path)
