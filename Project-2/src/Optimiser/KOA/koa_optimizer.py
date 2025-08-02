import numpy as np


def koa_optimizer(objective_function, lb, ub, dim, n_agents, max_iter, X_train, y_train, X_test, y_test):
    pos = np.random.uniform(low=lb, high=ub, size=(n_agents, dim))
    fit = np.array([objective_function(ind, X_train, y_train, X_test, y_test) for ind in pos])

    best_idx = np.argmin(fit)
    best_pos = pos[best_idx].copy()
    best_fit = fit[best_idx]

    for t in range(max_iter):
        for i in range(n_agents):
            rand_idx = np.random.randint(n_agents)
            r = np.random.rand(dim)
            step = r * (pos[rand_idx] - pos[i])
            new_pos = pos[i] + step

            new_pos = np.clip(new_pos, lb, ub)
            new_fit = objective_function(new_pos, X_train, y_train, X_test, y_test)
         

            if new_fit < fit[i]:
                pos[i] = new_pos
                fit[i] = new_fit

                if new_fit < best_fit:
                    best_fit = new_fit
                    best_pos = new_pos

        print(f"🔁 Iter {t+1}/{max_iter} - Best MAE: {best_fit:.5f}")

    return best_pos, best_fit

