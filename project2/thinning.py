import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Base shape of your inhomogeneous rate function λ_base(t)
def lambda_base(t):
    return 5 * np.exp(-0.5 * (t - 10)**2) + 3 * np.exp(-0.5 * ((t - 15)/1.5)**2)

# 2. Scaled lambda to target expected n arrivals
def make_lambda_scaled(n, t_max):
    integral, _ = quad(lambda_base, 0, t_max)
    scale = n / integral
    return lambda t: scale * lambda_base(t), scale

# 3. Thinning-based sampler for inhomogeneous Poisson process
def thinning_sampler(lambda_func, lambda_max, t_max):
    arrivals = []
    t = 0
    while t < t_max:
        t += np.random.exponential(1 / lambda_max)
        if t >= t_max:
            break
        if np.random.uniform(0, 1) < lambda_func(t) / lambda_max:
            arrivals.append(t)
    return np.array(arrivals)