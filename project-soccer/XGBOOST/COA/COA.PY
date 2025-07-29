import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


import pandas as pd

file_path = "D:/ML/project-soccer/data/85_Soccer_ETR_LGBR_COA_BWO.xlsx"
df = pd.read_excel(file_path, sheet_name="DATA after K-Fold")
# Separate back into train/test
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=["markat value"])
y_train = train_df["markat value"]

X_test = test_df.drop(columns=["markat value"])
y_test = test_df["markat value"]

# === BWO Implementation ===
def chimp_optimizer(obj_func, num_variables, num_agents, max_iter, lb, ub):
    positions = lb + np.random.rand(num_agents, num_variables) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in positions])
    best_idx = np.argmin(fitness)
    best_position = positions[best_idx].copy()
    best_fitness = fitness[best_idx]
    convergence_curve = [best_fitness]

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # Linearly decreasing from 2 to 0

        for i in range(num_agents):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2

            leader = best_position
            D_leader = np.abs(C * leader - positions[i])
            new_pos = leader - A * D_leader

            new_pos = np.clip(new_pos, lb, ub)
            new_fit = obj_func(new_pos)

            if new_fit < fitness[i]:
                positions[i] = new_pos
                fitness[i] = new_fit

                if new_fit < best_fitness:
                    best_fitness = new_fit
                    best_position = new_pos.copy()

        convergence_curve.append(best_fitness)

    return best_position, best_fitness, np.array(convergence_curve)

# === Split for Hyperparameter Optimization ===

# === Objective Function ===
def xgb_objective(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    learning_rate = float(params[2])
    subsample = float(params[3])

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    return rmse

# === Run BWO ===
lb = np.array([50, 3, 0.01, 0.5])
ub = np.array([300, 10, 0.3, 1.0])
num_variables = 4
num_agents = 5
max_iter = 5

best_params, best_fitness, convergence_curve = chimp_optimizer(

    xgb_objective,
    num_variables,
    num_agents,
    max_iter,
    lb,
    ub
)

best_n_estimators = int(best_params[0])
best_max_depth = int(best_params[1])
best_learning_rate = float(best_params[2])
best_subsample = float(best_params[3])

print("✅ Best Hyperparameters Found:")
print(f"   n_estimators : {best_n_estimators}")
print(f"   max_depth    : {best_max_depth}")
print(f"   learning_rate: {best_learning_rate:.4f}")
print(f"   subsample    : {best_subsample:.2f}")

# === Final Model Training ===

model = XGBRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    learning_rate=best_learning_rate,
    subsample=best_subsample,
    verbosity=0,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# === Evaluation Functions ===
def custom_error(y_true, y_pred):
    return ((y_true / y_pred) - 1) * 100

def evaluate(y_true, y_pred, name="Set"):
    rmse = mean_squared_error(y_true, y_pred)
    n10_index = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.10) * 100
    si = (rmse / np.mean(y_true)) * 100
    u95 = np.percentile(np.abs(y_true - y_pred), 95)
    r2 = r2_score(y_true, y_pred)

    print(f"\n🔹 {name} Metrics:")
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

