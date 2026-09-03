# house-price-predictor

Predicting house prices from features like size, location, and number of rooms. Started with California Housing (sklearn) and gradually added more complex models to see how much each one actually helps.

Not trying to win a Kaggle competition — just building something real and understanding what's happening at each step.

---

## what this does

- loads California Housing data (no download needed, straight from sklearn)
- explores distributions, correlations, and outliers
- engineers new features — room ratios, log transforms on skewed columns
- trains Linear Regression → Ridge/Lasso → XGBoost
- evaluates everything with RMSE and R²
- compares all models side by side at the end

---

## datasets

| dataset | source | notes |
|---------|--------|-------|
| California Housing | `sklearn.datasets` | 20,640 rows, 8 features, no missing values |

---

## notebooks

| notebook | what it does |
|----------|-------------|
| `01_data_exploration` | load data, distributions, correlation heatmap |
| `02_data_cleaning` | outlier detection, skewness check |
| `03_feature_engineering` | log transforms, interaction features, scaling |
| `04_linear_regression_baseline` | linear, ridge, lasso — with RMSE and residual plots |
| `05_gradient_boosting` | xgboost, feature importance |
| `06_model_comparison` | final table, best model, notes |

run them in order.

---

## setup

```bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
jupyter notebook
```

---

## results

| model | RMSE | R² |
|-------|------|-----|
| Linear Regression | — | — |
| Ridge | — | — |
| Lasso | — | — |
| XGBoost | — | — |

*will update after running the notebooks*

---

## what i learned

- log-transforming skewed features makes a bigger difference than switching models sometimes
- Ridge beats plain linear regression almost always — the regularization helps with correlated features
- XGBoost is powerful but you can overfit fast without cross-validation
- RMSE alone doesn't tell you much — residual plots show where the model actually struggles

---

## stack

Python 3.11 · scikit-learn · XGBoost · pandas · matplotlib · seaborn
