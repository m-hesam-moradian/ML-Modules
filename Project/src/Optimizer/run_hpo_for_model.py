import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.special import gamma


# Levy flight function (used in HPO)
def levy(n, m, beta=1.5):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=(n, m))
    v = np.random.normal(0, 1, size=(n, m))
    step = u / (np.abs(v) ** (1 / beta))
    return step


# Evaluate model accuracy
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# HPO algorithm function
def HPO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, model, X_train, y_train, X_test, y_test):
    def fitness(params):
        model.set_params(n_estimators=int(params[0]))  # Example for RandomForestClassifier
        acc = evaluate_model(model, X_train, y_train, X_test, y_test)
        return 1 - acc  # minimize error

    lowerbound = np.ones(dimension) * lowerbound if np.isscalar(lowerbound) else np.array(lowerbound)
    upperbound = np.ones(dimension) * upperbound if np.isscalar(upperbound) else np.array(upperbound)

    X = lowerbound + np.random.rand(SearchAgents, dimension) * (upperbound - lowerbound)
    fit = np.array([fitness(ind) for ind in X])
    best_so_far = np.zeros(Max_iterations)

    for t in range(Max_iterations):
        best_idx = np.argmin(fit)
        best = fit[best_idx]

        if t == 0 or best < fbest:
            fbest = best
            Xbest = X[best_idx].copy()

        # TODO: Add optimizer update logic (can be customized here)

        best_so_far[t] = fbest

    return fbest, Xbest, best_so_far


# 🔁 Full wrapper function
def run_hpo_on_model(X, y, model, hyperparam_range, dimension=1, SearchAgents=5, Max_iterations=30, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    fbest, best_params, _ = HPO(
        SearchAgents=SearchAgents,
        Max_iterations=Max_iterations,
        lowerbound=hyperparam_range[0],
        upperbound=hyperparam_range[1],
        dimension=dimension,
        model=model,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test
    )

    accuracy = 1 - fbest
    return fbest ,best_params, accuracy
