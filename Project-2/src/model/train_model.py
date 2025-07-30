import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from .evaluate import evaluate_model

def get_X_y(df, target_col='Predicted Load (kW)'):
    """
    Splits the DataFrame into features (X) and target (y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_and_evaluate_models(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    etr_scores = []
    lasso_scores = []

    fold = 1
    for train_idx, val_idx in kf.split(X):
        print(f"\n🔁 Fold {fold} ------------------")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ✅ Extra Trees Regressor
        etr = ExtraTreesRegressor(random_state=42)
        etr.fit(X_train, y_train)
        etr_preds = etr.predict(X_val)
        etr_metrics = evaluate_model(y_val, etr_preds)
        etr_scores.append(etr_metrics)
        print(f"  ETR → R2: {etr_metrics['R2']:.4f}, RMSE: {etr_metrics['RMSE']:.4f}, MAE: {etr_metrics['MAE']:.4f}, MAPE: {etr_metrics['MAPE']:.2f}%, MARD: {etr_metrics['MARD']:.2f}%")

        # ✅ Lasso Regression
        lasso = Lasso(random_state=42)
        lasso.fit(X_train, y_train)
        lasso_preds = lasso.predict(X_val)
        lasso_metrics = evaluate_model(y_val, lasso_preds)
        lasso_scores.append(lasso_metrics)
        print(f"  Lasso → R2: {lasso_metrics['R2']:.4f}, RMSE: {lasso_metrics['RMSE']:.4f}, MAE: {lasso_metrics['MAE']:.4f}, MAPE: {lasso_metrics['MAPE']:.2f}%, MARD: {lasso_metrics['MARD']:.2f}%")

        fold += 1

    return etr_scores, lasso_scores
