import time
import numpy as np
from src.data_loader import load_data
from src.feature_engineering import average_daily
from src.model.train_model import get_X_y, train_and_evaluate_models
import os
from objective_function import objective_et
from src.Optimiser.KOA.koa_optimizer import koa_optimizer
from src.Optimiser.COA.coa_optimizer import coa_optimizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from analysis.wilcoxon_test import perform_wilcoxon_test

DATA_PATH = os.path.join("data", "Data.xlsx")

def objective_lasso(params, X, y):
    alpha = params[0]
    max_iter = int(params[1])
    model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)
    return np.mean(maes)

def main():
    print("🔹 Loading raw data...")
    df = load_data(DATA_PATH)
    print(f"✅ Raw data shape: {df.shape}")

    print("🔹 Aggregating daily averages (96 samples per day)...")
    df_avg = average_daily(df, samples_per_day=96)
    print(f"✅ Aggregated data shape: {df_avg.shape}")
    print("✅ Sample preview:")
    print(df_avg.head())

    print("🔹 Preparing features and target...")
    X, y = get_X_y(df_avg)
    print(f"✅ X shape: {X.shape}, y shape: {y.shape}")

    print("🔹 Training baseline models with 5-Fold CV...")
    start_time = time.time()
    etr_scores, lasso_scores = train_and_evaluate_models(X, y)
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total training and evaluation time: {elapsed_time:.2f} seconds")

    def summarize_metrics(metrics_list):
        return {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}

    avg_etr = summarize_metrics(etr_scores)
    avg_lasso = summarize_metrics(lasso_scores)

    print("🔹 Extra Trees Regressor:")
    for k, v in avg_etr.items():
        print(f"  {k}: {v:.4f}")

    print("\n🔹 Lasso Regression:")
    for k, v in avg_lasso.items():
        print(f"  {k}: {v:.4f}")

    # Parameter bounds for Extra Trees
    lb_et = [10, 5, 2, 1]          # n_estimators, max_depth, min_samples_split, min_samples_leaf
    ub_et = [300, 50, 10, 10]
    dim_et = 4

    # Parameter bounds for Lasso
    lb_lasso = [0.0001, 100]       # alpha, max_iter
    ub_lasso = [1.0, 1000]
    dim_lasso = 2

    n_agents = 3
    max_iter = 5

    # Optimize Extra Trees with KOA
    print("\n🔹 Optimizing Extra Trees Regressor with KOA...")
    best_params_koa_et, best_mae_koa_et = koa_optimizer(objective_et, lb_et, ub_et, dim_et, n_agents, max_iter, X, y)
    print(f"ETR + KOA Best params: {best_params_koa_et}")
    print(f"ETR + KOA Best MAE: {best_mae_koa_et:.5f}")

    # Optimize Extra Trees with COA
    print("\n🔹 Optimizing Extra Trees Regressor with COA...")
    best_params_coa_et, best_mae_coa_et = coa_optimizer(objective_et, lb_et, ub_et, dim_et, n_agents, max_iter, X, y)
    print(f"ETR + COA Best params: {best_params_coa_et}")
    print(f"ETR + COA Best MAE: {best_mae_coa_et:.5f}")

    # Optimize Lasso with KOA
    print("\n🔹 Optimizing Lasso Regression with KOA...")
    best_params_koa_lasso, best_mae_koa_lasso = koa_optimizer(objective_lasso, lb_lasso, ub_lasso, dim_lasso, n_agents, max_iter, X, y)
    print(f"Lasso + KOA Best params: {best_params_koa_lasso}")
    print(f"Lasso + KOA Best MAE: {best_mae_koa_lasso:.5f}")

    # Optimize Lasso with COA
    print("\n🔹 Optimizing Lasso Regression with COA...")
    best_params_coa_lasso, best_mae_coa_lasso = coa_optimizer(objective_lasso, lb_lasso, ub_lasso, dim_lasso, n_agents, max_iter, X, y)
    print(f"Lasso + COA Best params: {best_params_coa_lasso}")
    print(f"Lasso + COA Best MAE: {best_mae_coa_lasso:.5f}")

    # Wilcoxon Test on baseline models
    mae_etr = [score['MAE'] for score in etr_scores]
    mae_lr = [score['MAE'] for score in lasso_scores]
    perform_wilcoxon_test(mae_etr, mae_lr, model_name_1="ETR", model_name_2="LR")

if __name__ == "__main__":
    main()
