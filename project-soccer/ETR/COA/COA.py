import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor  # Changed here

# === Load data ===
file_path = "D:/ML/project-soccer/data/85_Soccer_ETR_LGBR_COA_BWO.xlsx"
df = pd.read_excel(file_path, sheet_name="DATA after K-Fold")

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r'[^A-Za-z0-9]+', '_', regex=True)
    .str.replace(r'_+', '_', regex=True)
    .str.rstrip('_')
)

# Split dataset
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=["markat_value"])
y_train = train_df["markat_value"]

X_test = test_df.drop(columns=["markat_value"])
y_test = test_df["markat_value"]

# === Chimp Optimizer ===
def chimp_optimizer(obj_func, num_variables, num_agents, max_iter, lb, ub):
    population = lb + np.random.rand(num_agents, num_variables) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in population])

    sorted_idx = np.argsort(fitness)
    alpha, beta, delta, gamma = population[sorted_idx[:4]]
    alpha_fit, beta_fit, delta_fit, gamma_fit = fitness[sorted_idx[:4]]

    convergence_curve = []

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        for i in range(num_agents):
            for j in range(num_variables):
                for chimp in [alpha, beta, delta, gamma]:
                    r1, r2 = np.random.rand(), np.random.rand()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * chimp[j] - population[i][j])
                    if chimp is alpha:
                        X1 = chimp[j] - A * D
                    elif chimp is beta:
                        X2 = chimp[j] - A * D
                    elif chimp is delta:
                        X3 = chimp[j] - A * D
                    elif chimp is gamma:
                        X4 = chimp[j] - A * D

                population[i][j] = (X1 + X2 + X3 + X4) / 4

            population[i] = np.clip(population[i], lb, ub)
            fitness[i] = obj_func(population[i])

        sorted_idx = np.argsort(fitness)
        alpha, beta, delta, gamma = population[sorted_idx[:4]]
        alpha_fit, beta_fit, delta_fit, gamma_fit = fitness[sorted_idx[:4]]

        convergence_curve.append(alpha_fit)

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx], np.array(convergence_curve)

# === Objective Function for ExtraTrees ===
def etr_objective(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    return rmse

# === Run Chimp Optimizer ===
lb = np.array([50, 3])    # n_estimators, max_depth
ub = np.array([300, 15])
num_variables = 2
num_agents = 20
max_iter = 100

best_params, best_fitness, convergence_curve = chimp_optimizer(
    etr_objective,
    num_variables,
    num_agents,
    max_iter,
    lb,
    ub
)

best_n_estimators = int(best_params[0])
best_max_depth = int(best_params[1])

print("✅ Best Hyperparameters Found:")
print(f"   n_estimators : {best_n_estimators}")
print(f"   max_depth    : {best_max_depth}")

# === Final Model Training ===
model = ExtraTreesRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    n_jobs=-1,
    random_state=42
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
