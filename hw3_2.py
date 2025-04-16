import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K + 1e-10) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K + 1e-10) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def crank_nicolson_american(S_max, K, T, sigma, r, M, N, omega=1.2, option_type="put"):
    dS = S_max / M
    dt = T / N
    S = np.linspace(0, S_max, M + 1)
    V = np.maximum(K - S, 0) if option_type == "put" else np.maximum(S - K, 0)

    alpha = 0.25 * dt * (sigma**2 * (S / dS) ** 2 - r * S / dS)
    beta = -0.5 * dt * (sigma**2 * (S / dS) ** 2 + r)
    gamma = 0.25 * dt * (sigma**2 * (S / dS) ** 2 + r * S / dS)

    for n in range(N - 1, -1, -1):
        if option_type == "put":
            V[0] = K * np.exp(-r * (T - n * dt))
            V[-1] = 0
        else:
            V[0] = 0
            V[-1] = S_max - K * np.exp(-r * (T - n * dt))

        error = np.inf
        iteration = 0
        V_old = V.copy()

        while error > 1e-8 and iteration < 10000:
            error = 0.0
            for j in range(1, M):
                y = (V_old[j] + alpha[j] * V[j - 1] + gamma[j] * V[j + 1]) / (
                    1 - beta[j]
                )
                new_V = V[j] + omega * (y - V[j])
                payoff = max(K - S[j], 0) if option_type == "put" else max(S[j] - K, 0)
                new_V = max(payoff, new_V)
                error += (new_V - V[j]) ** 2
                V[j] = new_V

            error = np.sqrt(error)
            iteration += 1

    return S, V


# === Q2a ===
S_list = np.arange(0, 17, 2)
S_max = 20
M, N = 200, 200
K = 10
r = 0.1
sigma = 0.4
T = 4 / 12
omega = 1.2

S_grid, V_am_put = crank_nicolson_american(
    S_max, K, T, sigma, r, M, N, omega, option_type="put"
)
V_am_put_interp = np.interp(S_list, S_grid, V_am_put)
V_eu_put_exact = black_scholes_put(S_list, K, T, r, sigma)

df_puts = pd.DataFrame(
    {"S": S_list, "American Put": V_am_put_interp, "European Put": V_eu_put_exact}
)

# 輸出表格到 CSV
df_puts.to_csv("american_vs_european_put.csv", index=False)
print(df_puts)

# === Q2b ===
S_max_b = 30
M, N = 300, 300
K = 10
r = 0.25
sigma = 0.8
T = 1
D0 = 0.2  # dividend
r_eff = r - D0

S_grid_b, V_am_call = crank_nicolson_american(
    S_max_b, K, T, sigma, r_eff, M, N, omega, option_type="call"
)
V_eu_call_exact = black_scholes_call(S_grid_b, K, T, r_eff, sigma)
payoff_call = np.maximum(S_grid_b - K, 0)


print("Print the call American")
df_call = pd.DataFrame(
    {
        "S": S_grid_b[::10],
        "American Call": V_am_call[::10],
        "European Call": V_eu_call_exact[::10],
    }
)
df_call.to_csv("american_vs_european_call.csv", index=False)
print(df_call)

plt.figure(figsize=(10, 6))
plt.plot(S_grid_b, V_am_call, label="American Call", color="blue")
plt.plot(S_grid_b, V_eu_call_exact, label="European Call", linestyle="--", color="red")
plt.plot(S_grid_b, payoff_call, label="Payoff", linestyle=":", color="gray")
plt.xlabel("Stock Price S")
plt.ylabel("Option Value")
plt.title("Q2b: American vs European Call with Payoff")
plt.legend()
plt.grid(True)
plt.savefig("american_vs_european_call.png", dpi=300)
plt.show()
