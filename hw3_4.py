import numpy as np


def binomial_american_option_discrete_dividend(
    S0, K, T, r, sigma, N, dy, td, option_type="call"
):
    """
    Binomial method for American option with discrete dividend at time td.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Determine the dividend step
    dividend_step = int(td / dt + 0.5)

    # Build asset price tree with discrete dividend adjustment
    asset_tree = np.zeros((N + 1, N + 1))
    asset_tree[0, 0] = S0
    for i in range(1, N + 1):
        for j in range(i + 1):
            if j == 0:
                asset_tree[j, i] = asset_tree[j, i - 1] * u
            else:
                asset_tree[j, i] = asset_tree[j - 1, i - 1] * d

            # Apply dividend adjustment after td
            if i == dividend_step:
                asset_tree[j, i] *= 1 - dy

    # Option payoff at maturity
    option_tree = np.zeros_like(asset_tree)
    for j in range(N + 1):
        if option_type == "call":
            option_tree[j, N] = max(asset_tree[j, N] - K, 0)
        else:
            option_tree[j, N] = max(K - asset_tree[j, N], 0)

    # Backward induction
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


# Try this method on a basic test case
result = binomial_american_option_discrete_dividend(
    S0=100, K=100, T=1, r=0.05, sigma=0.2, N=100, dy=0.03, td=0.5, option_type="call"
)
result
