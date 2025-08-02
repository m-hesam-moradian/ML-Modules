import numpy as np

def p_obj(x, t, T):
    C = 0.2
    Q = 3
    return C * (1 / (np.sqrt(2 * np.pi) * Q)) * np.exp(-((x - 25) ** 2) / (2 * Q ** 2))

def coa_optimizer(objective_function, lb, ub, dim, n_agents, max_iter, X_train, y_train, X_test, y_test):
    thres = 0.5
    X = np.random.uniform(lb, ub, size=(n_agents, dim))
    fitness_f = np.zeros(n_agents)
    best_fitness = np.inf
    best_position = np.zeros(dim)

    for i in range(n_agents):
        fitness_f[i] = objective_function(X[i], X_train, y_train, X_test, y_test)
        if fitness_f[i] < best_fitness:
            best_fitness = fitness_f[i]
            best_position = X[i].copy()

    global_position = best_position.copy()
    global_fitness = best_fitness

    for t in range(1, max_iter + 1):
        C = 2 - (t / max_iter)
        temp = np.random.rand() * 15 + 20
        xf = (best_position + global_position) / 2
        Xfood = best_position.copy()
        Xnew = np.zeros_like(X)

        for i in range(n_agents):
            if temp > 30:
                if np.random.rand() < 0.5:
                    Xnew[i] = X[i] + C * np.random.rand(dim) * (xf - X[i])
                else:
                    for j in range(dim):
                        z = np.random.randint(n_agents)
                        Xnew[i][j] = X[i][j] - X[z][j] + xf[j]
            else:
                Xfood = np.clip(Xfood, lb, ub)
                F1 = objective_function(Xfood, X_train, y_train, X_test, y_test)
                P = 3 * np.random.rand() * fitness_f[i] / F1
                if P > 2:
                    Xfood = np.exp(-1 / P) * Xfood
                    for j in range(dim):
                        angle = 2 * np.pi * np.random.rand()
                        Xnew[i][j] = (X[i][j] +
                                     np.cos(angle) * Xfood[j] * p_obj(temp, t, max_iter) -
                                     np.sin(angle) * Xfood[j] * p_obj(temp, t, max_iter))
                else:
                    Xnew[i] = ((X[i] - Xfood) * p_obj(temp, t, max_iter) +
                               p_obj(temp, t, max_iter) * np.random.rand(dim) * X[i])

        Xnew = np.clip(Xnew, lb, ub)

        for i in range(n_agents):
            new_fitness = objective_function(Xnew[i], X_train, y_train, X_test, y_test)

            if new_fitness < global_fitness:
                global_fitness = new_fitness
                global_position = Xnew[i].copy()

            if new_fitness < fitness_f[i]:
                fitness_f[i] = new_fitness
                X[i] = Xnew[i].copy()
                if fitness_f[i] < best_fitness:
                    best_fitness = fitness_f[i]
                    best_position = X[i].copy()

        print(f"🦞 Iter {t}/{max_iter} - Best MAE: {best_fitness:.5f}")

    return best_position, best_fitness
