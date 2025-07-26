import numpy as np
from scipy.special import gamma
from src.data_loader import load_data
from src.preprocess import split_and_scale

def HPO_runner(
    file_path,
    target_column=None,
    model=None,  # Accept model object directly
    params=None,  # Accept hyperparameters as a dictionary of ranges
    SearchAgents=5,
    Max_iterations=10,
    task_type=None, 
    test_size=0.2,
    random_state=42
):
    # === Load and prepare data ===
    X, y = load_data(file_path, target_column)
    X_train, X_test, y_train, y_test = split_and_scale(X, y, test_size=test_size, random_state=random_state)

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

        convergence_curve = [best_fit]  # Track best error at each iteration

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
            convergence_curve.append(best_fit)  # Save best error after this iteration

        # Return best_fit (RMSE for regression, error for classification), best_pos, convergence_curve
        return best_fit, best_pos, convergence_curve

    # === Run Optimization ===
    best_fit, best_param_array, convergence_curve = HPO(
        SearchAgents, Max_iterations, 
        param_ranges=params, 
        model=model, 
        X_train=X_train, y_train=y_train, 
        X_test=X_test, y_test=y_test
    )

    # --- Convert numpy types to native Python types ---
    def to_native(val):
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, dict):
            return {k: to_native(v) for k, v in val.items()}
        if isinstance(val, (list, tuple, np.ndarray)):
            return [to_native(x) for x in val]
        return val

    best_params = {list(params.keys())[i]: best_param_array[i] for i in range(len(params))}
    best_params = to_native(best_params)
    best_fit = to_native(best_fit)
    convergence_curve = to_native(convergence_curve)

    print(f"Best Parameters for {model.__class__.__name__}: {best_params}")

    if task_type == "regression":
        print(f"Best RMSE: {best_fit:.4f}")
        return best_params, best_fit, convergence_curve
        library=model_library,
        function=model_function,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        attributes=model_params
    else:
        best_accuracy = 1 - best_fit
        print(f"Best Accuracy: {best_accuracy:.4f}")
        # Convert convergence_curve from error to accuracy for user clarity
        accuracy_curve = [1 - e for e in convergence_curve]
        return best_params, best_accuracy, accuracy_curve
