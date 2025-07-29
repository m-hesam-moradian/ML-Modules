import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from HO import HO

# === 1. Load and preprocess Excel data ===
df = pd.read_excel("../../../data/357_Computer_Hardware_RFE_NBR_KNNR_TSOA_CO_SHAP_M_Nasirianfar_2.xlsx")  # <== Replace with your file path

# Assume last column is the target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Optional: Normalize or scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 2. Define fitness function ===
def RF_fitness(params):
    # Extract integer-valued hyperparameters
    max_depth = int(params[0])
    n_estimators = int(params[1])
    
    # Bound check just in case (can also use np.clip in HO code)
    max_depth = max(1, min(max_depth, 50))
    n_estimators = max(10, min(n_estimators, 300))

    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    
    # Use negative MSE (scikit-learn uses higher is better)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    return -np.mean(scores)

# === 3. Setup search space ===
def fun_info(name):
    if name == "RandomForest":
        lowerbound = [1, 10]        # min max_depth and n_estimators
        upperbound = [50, 300]      # max max_depth and n_estimators
        dimension = 2
        return lowerbound, upperbound, dimension, RF_fitness
    raise ValueError("Function not supported")

# === 4. Run optimization ===
if __name__ == "__main__":
    fun_name = "RandomForest"
    SearchAgents = 10
    Max_iterations = 50

    lowerbound, upperbound, dimension, fitness = fun_info(fun_name)

    # HO expects scalars for lowerbound/upperbound
    Best_score, Best_pos, HO_curve = HO(
        SearchAgents, Max_iterations, lowerbound[0], upperbound[1], dimension, fitness
    )

    best_max_depth = int(Best_pos[0])
    best_n_estimators = int(Best_pos[1])

    print(f"Best max_depth: {best_max_depth}")
    print(f"Best n_estimators: {best_n_estimators}")
    print(f"Best CV MSE: {Best_score}")

    # === 5. Plot convergence ===
    plt.figure()
    plt.semilogy(HO_curve, color="#b28d90", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best score obtained so far")
    plt.title("HO for Random Forest Hyperparameter Tuning")
    plt.grid(True)
    plt.legend(["HO"])
    plt.show()
