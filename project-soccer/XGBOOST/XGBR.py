import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from error_code import getAllMetric

# === Load data ===
file_path = "D:/ML/project-soccer/data/85_Soccer_ETR_LGBR_COA_BWO.xlsx"
df = pd.read_excel(file_path, sheet_name="DATA after K-Fold")

# === Train/Test Split ===
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=["markat value"])
y_train = train_df["markat value"]

X_test = test_df.drop(columns=["markat value"])
y_test = test_df["markat value"]

# === Train Model (Default Parameters) ===
model = XGBRegressor(verbosity=0, n_jobs=-1)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# === Evaluation Functions ===
def custom_error(y_true, y_pred):
    return ((y_true / y_pred) - 1) * 100

def evaluate(y_true, y_pred, name="Set"):
    allmetrics = getAllMetric(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    n10_index = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.10) * 100
    si = (rmse / np.mean(y_true)) * 100
    u95 = np.percentile(np.abs(y_true - y_pred), 95)
    r2 = r2_score(y_true, y_pred)

    print(f"/n🔹 {name} Metrics:")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  N10 Index: {n10_index:.2f}%")
    print(f"  SI       : {si:.2f}%")
    print(f"  U95      : {u95:.4f}")
    print(f"  R²       : {r2:.4f}")

    return {
        "RMSE": rmse,
        "N10": n10_index,
        "SI": si,
        "U95": u95,
        "R2": r2
    }

# === Evaluate All Sets ===
train_error = custom_error(y_train, y_train_pred)
test_error = custom_error(y_test, y_test_pred)

train_metrics = evaluate(y_train, y_train_pred, name="Train")
test_metrics = evaluate(y_test, y_test_pred, name="Test")

# Combined Set
y_combined = pd.concat([y_train, y_test])
y_combined_pred = np.concatenate([y_train_pred, y_test_pred])
combined_metrics = evaluate(y_combined, y_combined_pred, name="Combined")

# First & Second Half of Test
midpoint = len(y_test) // 2
first_half_metrics = evaluate(y_test.iloc[:midpoint], y_test_pred[:midpoint], name="Test - First Half (Value 1)")
second_half_metrics = evaluate(y_test.iloc[midpoint:], y_test_pred[midpoint:], name="Test - Second Half (Value 2)")

# === Optional: Store Results ===
train_results = {
    "Real": y_train.values,
    "Prediction": y_train_pred,
    "Error (%)": train_error
}

test_results = {
    "Real": y_test.values,
    "Prediction": y_test_pred,
    "Error (%)": test_error
}
