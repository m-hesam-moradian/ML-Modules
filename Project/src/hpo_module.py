import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import gamma
import random

def HPO_runner(
    file_path,
    target_column='Fraudulent',
    model=None,  # Accept model object directly
    params=None,  # Accept hyperparameters as a dictionary of ranges
    SearchAgents=5,
    Max_iterations=10,
    task_type=None, 
    test_size=0.2,
    random_state=42
):
    # === Load and prepare data ===
    data = pd.read_excel(file_path)
    X = data.drop(columns=target_column)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # === Levy Flight Function ===
    def levy(n, m, beta=1.5):
        num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma_u = (num / den) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=(n, m))
        v = np.random.normal(0, 1, size=(n, m))
        return u / (np.abs(v) ** (1 / beta))

    # === Build Model Based on Hyperparameters ===
    def build_model(model, param_values):
        # Ensure the parameters are of correct type (int or float)
        for param_name, value in param_values.items():
            if isinstance(value, float):
                # If the parameter is n_estimators, we convert it to an integer
                if param_name == 'n_estimators':
                    param_values[param_name] = int(round(value))  # Round and cast to int
                else:
                    param_values[param_name] = float(value)
            elif isinstance(value, int):
                param_values[param_name] = int(value)
            elif isinstance(value, np.float64):  # If it's a numpy float, convert to regular float
                param_values[param_name] = int(value) if param_name == 'n_estimators' else float(value)
    
        model.set_params(**param_values)
        return model

    # === Model Evaluation Function ===
    from sklearn.metrics import accuracy_score, mean_squared_error

    def evaluate_model(model, X_train, y_train, X_test, y_test, task_type):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == 'regression':
          rmse = np.sqrt(mean_squared_error(y_test, y_pred))
          return rmse  # We return RMSE as the error to minimize
        else:  # classification
          accuracy = accuracy_score(y_test, y_pred)
          return 1 - accuracy  # Error to minimize

    # === HPO Core ===
    def HPO(SearchAgents, Max_iterations, param_ranges, model, X_train, y_train, X_test, y_test):
        # Initialize the positions of the agents (parameter values)
        param_names = list(param_ranges.keys())
        lowerbound = np.array([param_ranges[p][0] for p in param_names])
        upperbound = np.array([param_ranges[p][1] for p in param_names])
        dimension = len(param_names)

        Positions = np.random.uniform(low=lowerbound, high=upperbound, size=(SearchAgents, dimension))
        fitness = np.array([
            evaluate_model(
                build_model(model, {param_names[i]: p[i] for i in range(dimension)}),
                X_train, y_train, X_test, y_test,
                task_type
            )
            for p in Positions
        ])
 
        # Best initial position and fitness
        best_idx = np.argmin(fitness)
        best_pos = Positions[best_idx].copy()
        best_fit = fitness[best_idx]

        # Start optimization loop
        for t in range(Max_iterations):
            steps = levy(SearchAgents, dimension)
            Positions = Positions + 0.01 * steps * (Positions - best_pos)
            Positions = np.clip(Positions, lowerbound, upperbound)
            
            # Evaluate fitness for each agent and update best
            for i in range(SearchAgents):
                new_param_values = {param_names[j]: Positions[i][j] for j in range(dimension)}
                new_model = build_model(model, new_param_values)
                new_fit = evaluate_model(new_model, X_train, y_train, X_test, y_test, task_type)
                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    if new_fit < best_fit:
                        best_fit = new_fit
                        best_pos = Positions[i].copy()

        return best_fit, best_pos, 1 - best_fit  # Return best error (1 - accuracy)

    # === Run Optimization ===
    best_error, best_param_array, best_accuracy = HPO(
        SearchAgents, Max_iterations, 
        param_ranges=params, 
        model=model, 
        X_train=X_train, y_train=y_train, 
        X_test=X_test, y_test=y_test
    )

    # === Output Best Parameters and Accuracy ===
    best_params = {list(params.keys())[i]: best_param_array[i] for i in range(len(params))}
    print(f"Best Parameters for {model.__class__.__name__}: {best_params}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    if task_type == 'regression':
        return best_params, best_accuracy

