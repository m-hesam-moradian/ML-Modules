import numpy as np

def levy(n, m, beta=1.5):
    # Levy flight distribution generator
    from scipy.special import gamma
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=(n, m))
    v = np.random.normal(0, 1, size=(n, m))
    step = u / (np.abs(v) ** (1 / beta))
    return step

def HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    # اگر lowerbound و upperbound مقادیر اسکالر باشند آنها را به آرایه تبدیل می‌کنیم
    if np.isscalar(lowerbound):
        lowerbound = np.ones(dimension) * lowerbound
    if np.isscalar(upperbound):
        upperbound = np.ones(dimension) * upperbound

    # Initialization
    X = lowerbound + np.random.rand(SearchAgents, dimension) * (upperbound - lowerbound)
    fit = np.array([fitness(ind) for ind in X])

    best_so_far = np.zeros(Max_iterations)

    for t in range(Max_iterations):
        # Update the Best Candidate Solution
        best_idx = np.argmin(fit)
        best = fit[best_idx]

        if t == 0:
            Xbest = X[best_idx].copy()
            fbest = best
        elif best < fbest:
            fbest = best
            Xbest = X[best_idx].copy()

        # Phase 1: Exploration (First half population)
        half = SearchAgents // 2
        for i in range(half):
            Dominant_hippopotamus = Xbest
            I1 = np.random.randint(1, 3)  # 1 or 2
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, 2)  # [0 or 1, 0 or 1]
            RandGroupNumber = np.random.randint(1, SearchAgents + 1)
            RandGroup = np.random.choice(SearchAgents, RandGroupNumber, replace=False)
            MeanGroup = np.mean(X[RandGroup], axis=0) if len(RandGroup) > 1 else X[RandGroup[0]]

            Alfa = [
                (I2 * np.random.rand(dimension) + (~Ip1[0])),
                (2 * np.random.rand(dimension) - 1),
                np.random.rand(dimension),
                (I1 * np.random.rand(dimension) + (~Ip1[1])),
                np.random.rand()
            ]

            A = Alfa[np.random.randint(0, 5)]
            B = Alfa[np.random.randint(0, 5)]

            X_P1 = X[i] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i])

            T = np.exp(- (t + 1) / Max_iterations)
            if T > 0.6:
                X_P2 = X[i] + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i] + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)

            X_P2 = np.clip(X_P2, lowerbound, upperbound)

            F_P1 = fitness(X_P1)
            if F_P1 < fit[i]:
                X[i] = X_P1
                fit[i] = F_P1

            F_P2 = fitness(X_P2)
            if F_P2 < fit[i]:
                X[i] = X_P2
                fit[i] = F_P2

        # Phase 2: Exploration (Second half population)
        for i in range(half, SearchAgents):
            predator = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)
            F_HL = fitness(predator)
            distance2Leader = np.abs(predator - X[i])
            b = np.random.uniform(2, 4)
            c = np.random.uniform(1, 1.5)
            d = np.random.uniform(2, 3)
            l = np.random.uniform(-2 * np.pi, 2 * np.pi)
            RL = 0.05 * levy(1, dimension, 1.5)[0]

            if fit[i] > F_HL:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1 / distance2Leader)
            else:
                X_P3 = RL * predator + (b / (c - d * np.cos(l))) * (1 / (2 * distance2Leader + np.random.rand(dimension)))

            X_P3 = np.clip(X_P3, lowerbound, upperbound)

            F_P3 = fitness(X_P3)
            if F_P3 < fit[i]:
                X[i] = X_P3
                fit[i] = F_P3

        # Phase 3: Exploitation (All population)
        for i in range(SearchAgents):
            LO_LOCAL = lowerbound / (t + 1)
            HI_LOCAL = upperbound / (t + 1)
            Alfa = [
                2 * np.random.rand(dimension) - 1,
                np.random.rand(),
                np.random.randn()
            ]
            D = Alfa[np.random.randint(0, 3)]
            X_P4 = X[i] + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))
            X_P4 = np.clip(X_P4, lowerbound, upperbound)

            F_P4 = fitness(X_P4)
            if F_P4 < fit[i]:
                X[i] = X_P4
                fit[i] = F_P4

        best_so_far[t] = fbest
        print(f"Iteration {t+1}: Best Cost = {best_so_far[t]}")

    return fbest, Xbest, best_so_far
