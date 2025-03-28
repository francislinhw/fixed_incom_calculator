import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt
from scipy.stats import norm


def binprice(S0, K, r, T, dt, sigma, is_call=True):
    """
    Binomial Option Pricing (Cox-Ross-Rubinstein model).

    Parameters:
        S0      : current asset price
        K       : strike price
        r       : risk-free interest rate
        T       : time to maturity
        dt      : size of each time step (so #steps = N = T/dt)
        sigma   : volatility
        is_call : True for call, False for put

    Returns:
        stock_tree : 2D array with stock prices
        option_tree: 2D array with corresponding option prices
    """
    N = int(round(T / dt))  # number of steps
    # Up and down factors (Cox-Ross-Rubinstein version)
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    # Risk-neutral probability
    disc = exp(r * dt)
    p = (disc - d) / (u - d)

    # Initialize stock tree
    stock_tree = np.zeros((N + 1, N + 1))
    stock_tree[0, 0] = S0

    # Fill stock price tree
    for i in range(1, N + 1):
        stock_tree[i, 0] = stock_tree[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_tree[i, j] = stock_tree[i - 1, j - 1] * d

    # Initialize option tree
    option_tree = np.zeros_like(stock_tree)

    # Payoff at maturity
    for j in range(N + 1):
        if is_call:
            option_tree[N, j] = max(stock_tree[N, j] - K, 0.0)
        else:
            option_tree[N, j] = max(K - stock_tree[N, j], 0.0)

    # Backward recursion for option price
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Discounted expected value under risk-neutral measure
            option_tree[i, j] = exp(-r * dt) * (
                p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1]
            )

    return stock_tree, option_tree


def black_scholes_call_put(S, K, r, T, sigma):
    """
    Returns the Black-Scholes price for a Call and a Put.
    """
    from math import log, sqrt, exp

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Helper: cumulative distribution function for standard normal
    def cdf(x):
        return norm.cdf(x)

    call_price = S * cdf(d1) - K * exp(-r * T) * cdf(d2)
    put_price = K * exp(-r * T) * cdf(-d2) - S * cdf(-d1)

    return call_price, put_price


if __name__ == "__main__":
    # Parameters
    S0 = 100.0  # current asset price
    K = 100.0  # strike
    r = 0.05  # risk-free rate
    sigma = 0.4  # volatility
    T = 2.0  # time to maturity (years)
    dt = 0.5  # an initial step-size example (4 steps for T=2)

    # Example with dt=0.5
    # For demonstration, a single call & put tree with 4 steps
    stock_tree_put, put_tree = binprice(S0, K, r, T, dt, sigma, is_call=False)
    stock_tree_call, call_tree = binprice(S0, K, r, T, dt, sigma, is_call=True)

    # Now examine convergence by varying the number of steps (1 to 100)
    steps = 100
    call_price = np.zeros(steps)
    put_price = np.zeros(steps)

    for i in range(1, steps + 1):
        dt_i = T / i  # step size
        _, call_tree_i = binprice(S0, K, r, T, dt_i, sigma, is_call=True)
        _, put_tree_i = binprice(S0, K, r, T, dt_i, sigma, is_call=False)

        # The option price is at the root of the tree (i.e., [0,0])
        call_price[i - 1] = call_tree_i[0, 0]
        put_price[i - 1] = put_tree_i[0, 0]

    # Plot the call and put price convergence
    plt.figure()
    plt.plot(call_price)
    plt.title("Call prices (Binomial) for increasing steps")
    plt.xlabel("Steps")
    plt.ylabel("Option Price")
    plt.show()

    plt.figure()
    plt.plot(put_price)
    plt.title("Put prices (Binomial) for increasing steps")
    plt.xlabel("Steps")
    plt.ylabel("Option Price")
    plt.show()

    # Compare with Black-Scholes
    c_bs, p_bs = black_scholes_call_put(S0, K, r, T, sigma)
    print("Black-Scholes Call: ", c_bs)
    print("Black-Scholes Put : ", p_bs)

    # Show how binomial results converge to Black-Scholes
    x_vals = np.arange(1, steps + 1)

    plt.figure()
    plt.plot(x_vals, call_price, label="Binomial Call")
    plt.plot(x_vals, [c_bs] * steps, label="Black-Scholes Call")
    plt.title("Call price convergence to Black-Scholes")
    plt.xlabel("Steps")
    plt.ylabel("Option Price")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x_vals, put_price, label="Binomial Put")
    plt.plot(x_vals, [p_bs] * steps, label="Black-Scholes Put")
    plt.title("Put price convergence to Black-Scholes")
    plt.xlabel("Steps")
    plt.ylabel("Option Price")
    plt.legend()
    plt.show()
