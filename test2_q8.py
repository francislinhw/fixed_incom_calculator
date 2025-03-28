import numpy as np
from scipy.stats import norm


# Black-Scholes binary option pricer
def binary_option_price(S, K, T, r, sigma, option_type="call"):
    # Calculate d2
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        # Call binary option price
        price = np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        # Put binary option price
        price = np.exp(-r * T) * norm.cdf(-d2)

    return price


# Delta calculation
def binary_option_delta(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Delta for binary call or put
    delta = np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))

    return delta


# Gamma calculation
def binary_option_gamma(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Gamma for binary call or put
    gamma = (
        np.exp(-r * T)
        * (d2 * norm.pdf(d2) - norm.pdf(d2))
        / (S**2 * sigma * np.sqrt(T))
    )

    return gamma


# Vega calculation
def binary_option_vega(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Vega for binary call or put
    vega = np.exp(-r * T) * norm.pdf(d2) * np.sqrt(T) / sigma

    return vega


# Theta calculation
def binary_option_theta(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Theta for binary call or put
    theta = (
        -np.exp(-r * T)
        * norm.pdf(d2)
        * (r * T - np.log(S / K) * np.sqrt(T))
        / (2 * S * sigma * np.sqrt(T))
    )

    return theta


"""
Long binary call with strike price of $50: pays $100 if stock is above $50 and nothing otherwise.

Short binary put with strike price of $60: pays $100 if stock is below $60 and nothing otherwise.

Suppose the stock is at $55 now.
"""

# Parameters for the binary option 1
S = 55  # Current stock price
K = 50  # Strike price
T = 1  # Time to expiration (in years)
r = 0.00  # Risk-free interest rate
sigma = 0.2  # Volatility

# Calculate the binary call option price and Greeks
binary_price = binary_option_price(S, K, T, r, sigma, option_type="call") * 100
delta = binary_option_delta(S, K, T, r, sigma, option_type="call") * 100
gamma = binary_option_gamma(S, K, T, r, sigma, option_type="call") * 100
vega = binary_option_vega(S, K, T, r, sigma, option_type="call") * 100
theta = binary_option_theta(S, K, T, r, sigma, option_type="call") * 100

# Parameters for the binary option 2
S = 55  # Current stock price
K2 = 60  # Strike price
T = 1  # Time to expiration (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility

binary_price_2 = binary_option_price(S, K2, T, r, sigma, option_type="put") * 100
delta_2 = binary_option_delta(S, K2, T, r, sigma, option_type="put") * 100
gamma_2 = binary_option_gamma(S, K2, T, r, sigma, option_type="put") * 100
vega_2 = binary_option_vega(S, K2, T, r, sigma, option_type="put") * 100
theta_2 = binary_option_theta(S, K2, T, r, sigma, option_type="put") * 100

# Output results
print(f"Binary Call Option Price: {binary_price}")
print(f"Delta: {delta}")
print(f"Gamma: {gamma}")
print(f"Vega: {vega}")
print(f"Theta: {theta}")

print(f"Binary Put Option Price: {binary_price_2}")
print(f"Delta: {delta_2}")
print(f"Gamma: {gamma_2}")
print(f"Vega: {vega_2}")
print(f"Theta: {theta_2}")

combine_price = binary_price - binary_price_2
combine_delta = delta - delta_2
combine_gamma = gamma - gamma_2
combine_vega = vega - vega_2
combine_theta = theta - theta_2

print(f"Combined Price: {combine_price}")
print(f"Combined Delta: {combine_delta}")
print(f"Combined Gamma: {combine_gamma}")
print(f"Combined Vega: {combine_vega}")
print(f"Combined Theta: {combine_theta}")
