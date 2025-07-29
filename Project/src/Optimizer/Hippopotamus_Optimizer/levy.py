import numpy as np
from scipy.special import gamma

def levy(n, m, beta=1.5):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=(n, m))
    v = np.random.normal(0, 1, size=(n, m))
    return u / (np.abs(v) ** (1 / beta))
