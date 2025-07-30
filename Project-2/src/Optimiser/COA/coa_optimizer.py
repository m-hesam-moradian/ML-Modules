import numpy as np
from objective_function import objective_et

def p_obj(x, t, T):
    C = 0.2
    Q = 3
    return C * (1 / (np.sqrt(2 * np.pi) * Q)) * np.exp(-((x - 25) ** 2) / (2 * Q ** 2))

def coa_optimizer(objective_function, lb, ub, dim, n_agents, max_iter, X, y):
    lb = np.array(lb)
    ub = np.array(ub)

    # Initialize population
    population = np.random.uniform(lb, ub, (n_agents, dim))
    fitness = np.array([objective_function(ind, X, y) for ind in population])

    # Initialize global best
    best_idx = np.argmin(fitness)
    best_pos = population[best_idx].copy()
    best_fit = fitness[best_idx]

    for t in range(max_iter):
        C = 2 - (t / max_iter)
        temp = np.random.rand() * 15 + 20
        xf = best_pos.copy()
        Xfood = best_pos.copy()
        new_population = np.zeros_like(population)

        for i in range(n_agents):
            if temp > 30:
                if np.random.rand() < 0.5:
                    new_population[i] = population[i] + C * np.random.rand(dim) * (xf - population[i])
                else:
                    z = np.random.randint(n_agents)
                    new_population[i] = population[i] - population[z] + xf
            else:
                F1 = np.sum(np.abs(population[i] - Xfood)) / dim
                P = 3 * np.random.rand() * fitness[i] / (F1 + 1e-9)  # Avoid division by zero

                if P > 2:
                    Xfood = np.exp(-1 / P) * Xfood
                    for j in range(dim):
                        theta = 2 * np.pi * np.random.rand()
                        new_population[i, j] = (
                            population[i, j]
                            + np.cos(theta) * Xfood[j] * p_obj(temp, t, max_iter)
                            - np.sin(theta) * Xfood[j] * p_obj(temp, t, max_iter)
                        )
                else:
                    new_population[i] = (
                        (population[i] - Xfood) * p_obj(temp, t, max_iter)
                        + p_obj(temp, t, max_iter) * np.random.rand(dim) * population[i]
                    )

        # Apply bounds
        new_population = np.clip(new_population, lb, ub)

        # Evaluate new population
        for i in range(n_agents):
            new_fit = objective_function(new_population[i], X, y)
            if new_fit < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fit
                if new_fit < best_fit:
                    best_fit = new_fit
                    best_pos = new_population[i]

        print(f"🔁 Iter {t+1}/{max_iter} - Best MAE: {best_fit:.5f}")

    return best_pos, best_fit
