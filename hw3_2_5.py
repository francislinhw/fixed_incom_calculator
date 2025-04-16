import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes formula for European call"""
    d1 = (np.log(S / K + 1e-10) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def monte_carlo_call(S0, K, T, r, sigma, N):
    """Monte Carlo simulation for European call"""
    np.random.seed(42)
    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)


# === 測試參數 ===
S0 = 10
K = 10
T = 0.5  # 小於一年 (t < T)
r = 0.05
sigma = 0.3

# 精確解
bs_price = black_scholes_call(S0, K, T, r, sigma)

# 不同模擬次數下的 Monte Carlo 結果
realizations = [10**2, 10**3, 10**4, 10**5, 10**6]
mc_prices = []
errors = []

for N in realizations:
    price = monte_carlo_call(S0, K, T, r, sigma, N)
    mc_prices.append(price)
    errors.append(abs(price - bs_price))

# 繪圖比較
plt.figure(figsize=(10, 6))
plt.plot(realizations, errors, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Realizations (log scale)")
plt.ylabel("Absolute Error (log scale)")
plt.title("Monte Carlo Error vs Realizations (European Call)")
plt.grid(True)
plt.axhline(errors[0] / 2, color="red", linestyle="--", label="½ Initial Error")
plt.legend()
plt.show()


for i, N in enumerate(realizations):
    print(f"N = {N:>7}, MC = {mc_prices[i]:.6f}, Error = {errors[i]:.6f}")
