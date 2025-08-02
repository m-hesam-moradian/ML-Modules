import time
import numpy as np
from src.data_loader import load_data
from src.feature_engineering import average_daily
from src.model.train_model import (
    get_X_y,
    train_and_evaluate_models,
    get_best_fold_data_from_kf,
)
import os
from objective_function import objective_et, objective_lasso
from src.Optimiser.KOA.koa_optimizer import koa_optimizer
from src.Optimiser.COA.coa_optimizer import coa_optimizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from analysis.wilcoxon_test import perform_wilcoxon_test


DATA_PATH = os.path.join("data", "Data.xlsx")
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
from src.Utils.K_Fold import K_Fold


print("🔹 Performing K-Fold...")
X_train, X_test, y_train, y_test, combined_df = K_Fold(X, y, n_splits=5)
