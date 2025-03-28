"""
A stock is at $100. Simulate its future dynamics using geometric brownian motion with a rate of 3% and a volatility of 0.40.

Run 20,000 daily simulations for a 1-year period. Assume 30 days in each month.

Estimate the probability that the stock drops below $90 and then rises above $110 in the 1 year period.
"""

import numpy as np


def simulate_gbm(rate, sigma, stock, T, steps, nTrials):
    dt = T / steps
    dW = np.random.normal(0, 1, (steps, nTrials)) * np.sqrt(dt)
    W = np.cumsum(dW, axis=0)
    t = np.linspace(0, T, steps)
    S = stock * np.exp((rate - 0.5 * sigma**2) * t[:, None] + sigma * W)
    return S


def simulate_gbm_with_events(
    rate: float = 0.03,
    sigma: float = 0.4,
    stock: float = 100,
    T: float = 1,
    steps: int = 360,  # 30 days/month * 12 months
    nTrials: int = 20000,
    lower_threshold: float = 90,
    upper_threshold: float = 110,
):
    paths = simulate_gbm(rate, sigma, stock, T, steps, nTrials)
    event_met = np.zeros(nTrials, dtype=bool)

    for i in range(nTrials):
        path = paths[:, i]
        if np.any(path < lower_threshold):
            # Find first time stock drops below lower_threshold
            t_low_indices = np.where(path < lower_threshold)[0]
            first_t_low = t_low_indices[0]
            # Check if stock ever exceeds upper_threshold after first_t_low
            if np.any(path[first_t_low:] > upper_threshold):
                event_met[i] = True

    prob = np.mean(event_met)
    return prob


# Run simulation
probability = simulate_gbm_with_events()
print(f"Estimated probability: {probability:.4f}")
# Estimated probability: 0.3781
