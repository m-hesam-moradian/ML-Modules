import numpy as np
from levy import levy

def HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    lowerbound = np.ones(dimension) * lowerbound
    upperbound = np.ones(dimension) * upperbound

    X = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    fit = np.array([fitness(x) for x in X])

    fbest = np.min(fit)
    Xbest = X[np.argmin(fit)].copy()
    HO_curve = []

    for t in range(1, Max_iterations + 1):
        # Phase 1 - Exploration
        for i in range(SearchAgents // 2):
            I1, I2 = np.random.randint(1, 3, size=2)
            Ip1 = np.random.randint(0, 2, size=2)
            RandGroup = np.random.choice(SearchAgents, size=np.random.randint(1, SearchAgents+1), replace=False)
            MeanGroup = np.mean(X[RandGroup], axis=0)

            Alfa = [
                I2 * np.random.rand(dimension) + (1 - Ip1[0]),
                2 * np.random.rand(dimension) - 1,
                np.random.rand(dimension),
                I1 * np.random.rand(dimension) + (1 - Ip1[1]),
                np.random.rand()
            ]
            A = Alfa[np.random.randint(5)]
            B = Alfa[np.random.randint(5)]

            X_P1 = X[i] + np.random.rand() * (Xbest - I1 * X[i])
            T = np.exp(-t / Max_iterations)
            X_P2 = (
                X[i] + A * (Xbest - I2 * MeanGroup) if T > 0.6
                else X[i] + B * (MeanGroup - Xbest) if np.random.rand() > 0.5
                else np.random.uniform(lowerbound, upperbound)
            )

            for new_X in [X_P1, X_P2]:
                new_X = np.clip(new_X, lowerbound, upperbound)
                if fitness(new_X) < fit[i]:
                    X[i] = new_X
                    fit[i] = fitness(new_X)

        # Phase 2 - Defense
        RL = 0.05 * levy(SearchAgents, dimension)
        for i in range(SearchAgents // 2, SearchAgents):
            predator = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)
            F_HL = fitness(predator)
            distance2Leader = np.abs(predator - X[i])
            b, c, d, l = np.random.uniform(2, 4), np.random.uniform(1, 1.5), np.random.uniform(2, 3), np.random.uniform(-2*np.pi, 2*np.pi)

            if fit[i] > F_HL:
                X_P3 = RL[i] * predator + (b / (c - d * np.cos(l))) * (1 / distance2Leader)
            else:
                X_P3 = RL[i] * predator + (b / (c - d * np.cos(l))) * (1 / (2 * distance2Leader + np.random.rand(dimension)))

            X_P3 = np.clip(X_P3, lowerbound, upperbound)
            if fitness(X_P3) < fit[i]:
                X[i] = X_P3
                fit[i] = fitness(X_P3)

        # Phase 3 - Escape
        for i in range(SearchAgents):
            LO = lowerbound / t
            HI = upperbound / t
            D = [
                2 * np.random.rand(dimension) - 1,
                np.random.rand(),
                np.random.randn()
            ][np.random.randint(3)]
            X_P4 = X[i] + np.random.rand() * (LO + D * (HI - LO))
            X_P4 = np.clip(X_P4, lowerbound, upperbound)
            if fitness(X_P4) < fit[i]:
                X[i] = X_P4
                fit[i] = fitness(X_P4)

        # Update best
        current_best = np.min(fit)
        if current_best < fbest:
            fbest = current_best
            Xbest = X[np.argmin(fit)].copy()
        HO_curve.append(fbest)
        
        print(f"Iteration {t}: Best Cost = {fbest:.6f}")

    return fbest, Xbest, HO_curve
