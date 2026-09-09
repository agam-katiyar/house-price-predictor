import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred, label: str = "model") -> dict:
    r = rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} — RMSE: {r:.4f}  R²: {r2:.4f}")
    return {"model": label, "rmse": round(r, 4), "r2": round(r2, 4)}


def plot_residuals(y_true, y_pred, label: str = "model"):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # scatter: predicted vs actual
    axes[0].scatter(y_pred, y_true, alpha=0.3, s=10)
    axes[0].plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], "r--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"{label} — Predicted vs Actual")

    # scatter: residuals
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"{label} — Residuals")

    plt.tight_layout()
    plt.savefig(f"plots/residuals_{label.lower().replace(' ', '_')}.png", dpi=120)
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    # works for tree-based models with feature_importances_
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.nlargest(top_n).sort_values()

    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="steelblue")
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=120)
    plt.show()
