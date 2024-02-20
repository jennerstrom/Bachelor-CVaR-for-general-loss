import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


# Define a cdf with support on 0 to 1 with an atom at 0.5 and 0.75

def f(x):
    if x < 0:
        return 0
    if x >= 1:
        return 1
    elif x < 0.5:
        return x
    elif x == 0.5:
        return 0.6
    elif 0.5 < x < 0.75:
        return x * (0.75 - 0.6) / (0.75 - 0.5) + 0.6 - 0.5 * (0.75 - 0.6) / (0.75 - 0.5)
    elif x == 0.75:
        return 0.8
    elif x < 1:
        return x * (1 - 0.8) / (1 - 0.75) + 0.8 - 0.75 * (1 - 0.8) / (1 - 0.75)


# Plot the cdf
x = np.linspace(0, 1.5, 100)  # 100 points between 0 and 1
y = [f(i) for i in x]

pos = np.where(np.abs(np.diff(y)) >= 0.03)

for i in pos:
    x = np.insert(x, i + 1, np.nan)
    y = np.insert(y, i + 1, np.nan)

# Add a filled point at (0.5, 0.6) and (0.75, 0.8) and a hollow point at (0.5, 0.5) and (0.75, 0.75)
plt.plot(x, y)
plt.plot([0.5, 0.75], [0.6, 0.8], 'bo')
plt.plot([0.5 - 0.01, 0.75], [0.5 - 0.01, 0.75], 'bo', markerfacecolor='white')
plt.show()


# Define the helper function for CVaR
def F_alpha(zeta, alpha):
    atom_expectation = 0.1 * max(0, 0.5 - zeta) + 0.05 * max(0, 0.75 - zeta)
    rest_expectation = 0.5
    return zeta + 1 / (1 - alpha) * (rest_expectation + atom_expectation)


# Fully discrete case - Binomial distribution with p = 1/3, n = 100
def cvar_discrete(alpha):
    cvar_non_scaled = 0
    prob = 0
    z_k_alpha = 0
    k = 0

    while k <= 10:
        prob = prob + binom.pmf(k, 10, 1 / 3)
        if prob >= alpha:
            z_k_alpha = k
            break
        k += 1

    for k in range(0, 11):
        if k <= z_k_alpha:
            cvar_non_scaled = cvar_non_scaled + (binom.pmf(k, 10, 1 / 3)) * z_k_alpha
        else:
            cvar_non_scaled = cvar_non_scaled + binom.pmf(k, 10, 1 / 3) * k

    cvar_non_scaled = cvar_non_scaled - alpha * z_k_alpha

    return cvar_non_scaled / (1 - alpha)

# Plot the cdf
x = np.linspace(0.01, 0.98, 1000)  # 100 points between 0 and 1
y = [cvar_discrete(i) for i in x]

plt.plot(x, y)
plt.xlabel(r'$\alpha$')
plt.ylabel("CVaR")
plt.title(r'CVaR w.r.t. $\alpha$ for a binomial loss with $p = \frac{1}{3}$, $n = 10$')
plt.show()
