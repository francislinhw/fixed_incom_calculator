import numpy as np
from typing import Optional, List, Dict


def binomial_option_price(
    payoff_func,  # Payoff function taking a price path and optional parameters
    S0: float,  # Initial stock price
    T: float,  # Time to maturity
    delta_t: int,  # Number of time steps (layers)
    r: float,  # Risk-free interest rate
    flavor: str,  # 'European' or 'American'
    u: float = None,  # Up factor
    d: float = None,  # Down factor
    sigma: float = None,  # Volatility (optional if u and d are provided)
    custom_tree: Optional[
        List[List[Dict[str, float]]]
    ] = None,  # Custom tree structure (not implemented)
    **kwargs  # Additional parameters for the payoff function (e.g., K)
):
    if custom_tree is not None:
        raise NotImplementedError("Custom tree functionality is not implemented.")

    dt = T / delta_t  # Time per step

    # Calculate u and d using CRR if not provided
    if u is None or d is None:
        if sigma is None:
            raise ValueError("Either u and d or sigma must be provided.")
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u

    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Build the binomial tree, storing each node's path
    tree = []
    current_layer = [{"path": [S0], "value": 0.0}]
    tree.append(current_layer)

    for step in range(delta_t):
        next_layer = []
        for node in current_layer:
            current_price = node["path"][-1]
            up_price = current_price * u
            down_price = current_price * d
            next_layer.append({"path": node["path"] + [up_price], "value": 0.0})
            next_layer.append({"path": node["path"] + [down_price], "value": 0.0})
        tree.append(next_layer)
        current_layer = next_layer

    # Calculate option values by backward induction
    for i in reversed(range(len(tree))):
        layer = tree[i]
        for j in range(len(layer)):
            node = layer[j]
            if i == delta_t:  # Expiration layer
                node["value"] = payoff_func(node["path"], **kwargs)
            else:
                # Continuation value from children
                child1 = tree[i + 1][2 * j]
                child2 = tree[i + 1][2 * j + 1]
                cont_value = (
                    p * child1["value"] + (1 - p) * child2["value"]
                ) * discount

                if flavor == "American":
                    immediate_value = payoff_func(node["path"], **kwargs)
                    node["value"] = max(immediate_value, cont_value)
                else:
                    node["value"] = cont_value

    return tree[0][0]["value"]


# Example payoff functions
def call_payoff(prices, K):
    return max(prices[-1] - K, 0)


def put_payoff(prices, K):
    return max(K - prices[-1], 0)


def asian_payoff(prices):
    return max(np.mean(prices) - prices[0], 0)  # Example Asian option payoff


price = binomial_option_price(
    call_payoff, 100, 1, 10, 0.05, "European", u=1.1, d=0.9, K=100
)
print(price)
