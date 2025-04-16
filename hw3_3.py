import numpy as np


def binomial_american_option(S0, K, T, r, q, sigma, N, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # 建立資產價格樹
    asset_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S0 * (u ** (i - j)) * (d**j)

    # 建立 payoff 樹
    option_tree = np.zeros_like(asset_tree)
    for j in range(N + 1):
        if option_type == "call":
            option_tree[j, N] = max(asset_tree[j, N] - K, 0)
        else:
            option_tree[j, N] = max(K - asset_tree[j, N], 0)

    # 倒推回來
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = np.exp(-r * dt) * (
                p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
            )
            if option_type == "call":
                exercise = max(asset_tree[j, i] - K, 0)
            else:
                exercise = max(K - asset_tree[j, i], 0)
            option_tree[j, i] = max(continuation, exercise)

    return option_tree[0, 0], u, d, p


# 參數與(2b)相同
K = 10
r = 0.25
D0 = 0.2
sigma = 0.8
T = 1
N = 200

for S0 in [10, 20, 25]:
    value, u, d, p = binomial_american_option(
        S0, K, T, r, D0, sigma, N, option_type="call"
    )
    print(f"S = {S0}: American Call = {value:.4f}")

print(f"\nu = {u:.4f}, d = {d:.4f}, p = {p:.4f}")
