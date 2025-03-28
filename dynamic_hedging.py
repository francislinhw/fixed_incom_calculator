import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve


# Black-Scholes functions
def blsprice(S, K, r, T, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return call_price, put_price


def blsdelta(S, K, r, T, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    call_delta = np.exp(-q * T) * norm.cdf(d1)
    put_delta = -np.exp(-q * T) * norm.cdf(-d1)
    return call_delta, put_delta


def blsgamma(S, K, r, T, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


# American Option Pricing using Binomial Model
def american_put_price(S0, K, r, T, sigma, N):
    dt = T / N
    time = np.linspace(0, T, N + 1)
    stock_price = np.full((N + 1, N + 1), np.nan)
    stock_price[0, 0] = S0

    for t in range(1, N + 1):
        stock_price[:t, t] = stock_price[:t, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt)
        )
        stock_price[t, t] = stock_price[t - 1, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt)
        )

    put_value = np.zeros_like(stock_price)
    put_value[:, -1] = np.maximum(K - stock_price[:, -1], 0)

    for t in range(N - 1, -1, -1):
        continuation = (
            np.exp(-r * dt)
            * 0.5
            * (put_value[: t + 1, t + 1] + put_value[1 : t + 2, t + 1])
        )
        exercise = K - stock_price[: t + 1, t]
        put_value[: t + 1, t] = np.maximum(exercise, continuation)

    return put_value[0, 0]


# Example 1: Delta Hedging
sigma = 0.3
r = 0.03
K_call = 1000
T_call = 2
S0 = 1100
Number_of_Stocks = 1000

# Calculate call price and delta
call_price, _ = blsprice(S0, K_call, r, T_call, sigma)
call_delta, _ = blsdelta(S0, K_call, r, T_call, sigma)

# Solve system of equations
A = np.array([[1, call_delta], [S0, call_price]])
B = np.array([0, S0 * Number_of_Stocks])
X = solve(A, B)
stocks_delta = X[0]
calls_delta = X[1]

# Value comparison
S_range = np.arange(S0 - 200, S0 + 201)
call_prices, _ = blsprice(S_range, K_call, r, T_call, sigma)
unhedged = S_range * Number_of_Stocks
delta_hedged = S_range * stocks_delta + call_prices * calls_delta

plt.figure(figsize=(10, 6))
plt.plot(S_range, unhedged, "r", label="Unhedged")
plt.plot(S_range, delta_hedged, "b", label="Delta-Hedged")
plt.xlabel("Stock Price")
plt.ylabel("Portfolio Value")
plt.legend()
plt.title("Unhedged vs Delta-Hedged Portfolio")
plt.show()

# Example 2: Delta-Gamma Hedging
K_put = 700
T_put = 1.5

# Calculate put price, delta, gamma
_, put_price = blsprice(S0, K_put, r, T_put, sigma)
_, put_delta = blsdelta(S0, K_put, r, T_put, sigma)
call_gamma = blsgamma(S0, K_call, r, T_call, sigma)
put_gamma = blsgamma(S0, K_put, r, T_put, sigma)

# Solve system of equations
A = np.array(
    [
        [1, call_delta, put_delta],
        [0, call_gamma, put_gamma],
        [S0, call_price, put_price],
    ]
)
B = np.array([0, 0, S0 * Number_of_Stocks])
X = solve(A, B)
stocks_dg = X[0]
calls_dg = X[1]
puts_dg = X[2]

# Value comparison
_, put_prices = blsprice(S_range, K_put, r, T_put, sigma)
delta_gamma_hedged = S_range * stocks_dg + call_prices * calls_dg + put_prices * puts_dg

plt.figure(figsize=(10, 6))
plt.plot(S_range, unhedged, "r", label="Unhedged")
plt.plot(S_range, delta_hedged, "k", label="Delta-Hedged")
plt.plot(S_range, delta_gamma_hedged, "g", label="Delta-Gamma-Hedged")
plt.xlabel("Stock Price")
plt.ylabel("Portfolio Value")
plt.legend()
plt.title("Comparison of Hedging Strategies")
plt.show()

# Dynamic Delta Hedging
dt_choices = [1 / 12, 1 / 52, 1 / 252]
T_maturity = 2
T_sim = 1
mu = 0.05
sigma_sim = 0.25
r_sim = 0.03
F = 100
n_sim = 10
K_sim = 100

plt.figure(figsize=(15, 10))
for i, dt in enumerate(dt_choices, 1):
    n_steps = int(T_sim / dt)
    time = np.linspace(0, T_sim, n_steps + 1)
    time_to_maturity = T_maturity - time

    dW = np.sqrt(dt) * np.random.randn(n_steps, n_sim)
    log_stock = np.zeros((n_steps + 1, n_sim))
    log_stock[0, :] = np.log(F)

    for t in range(1, n_steps + 1):
        log_stock[t, :] = (
            log_stock[t - 1, :]
            + (mu - 0.5 * sigma_sim**2) * dt
            + sigma_sim * dW[t - 1, :]
        )

    stock_price = np.exp(log_stock)

    call_prices = np.zeros_like(stock_price)
    call_deltas = np.zeros_like(stock_price)
    for t in range(n_steps + 1):
        price, _ = blsprice(
            stock_price[t, :],
            K_sim,
            r_sim,
            np.maximum(time_to_maturity[t], 1e-3),
            sigma_sim,
        )
        delta, _ = blsdelta(
            stock_price[t, :],
            K_sim,
            r_sim,
            np.maximum(time_to_maturity[t], 1e-3),
            sigma_sim,
        )
        call_prices[t, :] = price
        call_deltas[t, :] = delta

    portfolio = np.zeros_like(stock_price)
    portfolio[0, :] = stock_price[0, :]

    for sim in range(n_sim):
        for t in range(n_steps):
            S = stock_price[t, sim]
            C = call_prices[t, sim]
            D = call_deltas[t, sim]
            V = portfolio[t, sim]

            A_mat = np.array([[S, C], [1, D]])
            B_vec = np.array([V, 0])
            try:
                x = solve(A_mat, B_vec)
            except:
                x = np.array([0, 0])

            S_next = stock_price[t + 1, sim]
            C_next = call_prices[t + 1, sim]
            portfolio[t + 1, sim] = x[0] * S_next + x[1] * C_next

    plt.subplot(3, 1, i)
    for sim in range(n_sim):
        plt.plot(time, portfolio[:, sim])
    plt.plot(time, F * np.exp(r_sim * time), "r", linewidth=2)
    plt.title(f"Rebalancing Interval: {dt:.3f}")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.ylim(80, 120)

plt.tight_layout()
plt.show()
