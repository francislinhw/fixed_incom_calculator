from fixed_income_clculators.simulation_pricers import (
    simulate_gbm_with_events,
    simulate_gbm_barrier_option_pricer,
    price_generic_payoff,
)
from fixed_income_clculators.core import (
    call_delta,
    currency_option_delta,
    currency_option_rho,
    currency_option_theta,
    currency_option_vega,
    gamma,
    currency_option_gamma,
)
import numpy as np

prob = simulate_gbm_with_events(verbose=False)

print(f"Probability of the event: {prob}")

# %% calculating probability that stock crosses
# %% $120 during the first year.

prob = simulate_gbm_with_events(
    rate=0.03,
    sigma=0.4,
    stock=100,
    T=1,
    steps=360,
    nTrials=20000,
    events=[
        [0, 1, 120, "above"],
    ],
    verbose=False,
)

print(f"Probability of the event: {prob}")


# %%

price = simulate_gbm_barrier_option_pricer(
    rate=0.03,
    sigma=0.1,
    stock=100,
    T=1,
    steps=720,
    nTrials=20000,
    barrier=120,
    barrier_type="up_in",
    option_type="call",
    strike=110,
    barrier_start_time=0,
    barrier_end_time=1,
    verbose=False,
)

print(f"Price of the barrier option: {price}")

# %%

strike = 100


def call_payoff(path):
    return max(path[-1] - strike, 0.0)


call_price = price_generic_payoff(payoff_func=call_payoff)
print(f"European Call Price: {call_price:.4f}")


# 4) Price a European put
def put_payoff(path):
    return max(strike - path[-1], 0.0)


put_price = price_generic_payoff(payoff_func=put_payoff)
print(f"European Put Price: {put_price:.4f}")


# 5) Price an Asian call (arithmetic average)
def asian_call_payoff(path):
    avg_price = np.mean(path)
    return max(avg_price - strike, 0.0)


asian_call_price = price_generic_payoff(payoff_func=asian_call_payoff)
print(f"Asian Call Price: {asian_call_price:.4f}")


# 6) Price a square call
def square_call_payoff(path):
    return max((path[-1] - strike) ** 2, 0.0)


square_call_price = price_generic_payoff(payoff_func=square_call_payoff)
print(f"Square Call Price: {square_call_price:.4f}")


# 7) Price a log-contract
def log_contract_payoff(path):
    return max(np.log(path[-1]), 0.0)


log_contract_price = price_generic_payoff(payoff_func=log_contract_payoff)
print(f"Log Contract Price: {log_contract_price:.4f}")


# 8) Price a square root
def square_root_payoff(path):
    return max(np.sqrt(path[-1]), 0.0)


square_root_price = price_generic_payoff(payoff_func=square_root_payoff)
print(f"Square Root Price: {square_root_price:.4f}")


# 9) Price a forward start option
# %% price asian..average price call strike 100.
# %% price a forward starter!
# forward_starter_price = exp(-rate*T)...
# * mean(max(s(end,:)- s(51,:),0));


def forward_starter_payoff(path):
    return max(path[-1] - path[50], 0.0)


forward_starter_price = price_generic_payoff(payoff_func=forward_starter_payoff)
print(f"Forward Starter Price: {forward_starter_price:.4f}")


# %% Price cash barrier pays $75 if the
# %% stock crosses $110
def cash_barrier_payoff(path):
    return 75 if np.any(path > 110) else 0.0


cash_barrier_price = price_generic_payoff(
    rate=0.05,
    T=2,
    sigma=0.4,
    stock=100,
    steps=720,
    nTrials=20000,
    payoff_func=cash_barrier_payoff,
)
print(f"Cash Barrier Price: {cash_barrier_price:.4f}")


"""
A stock is at $100 and moving with a volatility of 0.40. The interest rate is 5% per year. Price a derivative that pays (Smax - Smin) during the period of 1 year. Here Smax and Smin denote the maximum and the minimum of the stock along the path it takes over the year, respectively.

Use a geometric brownian motion dynamics for the stock. Run 20,000 daily simulations for the 1-year period and find the price of the derivative.

The price is approximately
"""


def max_min_payoff(path):
    return np.max(path) - np.min(path)


max_min_price = price_generic_payoff(
    rate=0.05,
    sigma=0.4,
    stock=100,
    T=1,
    steps=360,
    nTrials=20000,
    payoff_func=max_min_payoff,
)
print(f"======= Max Min Price: {max_min_price:.4f}")

# Delta of a call option
delta = call_delta(S=100, K=100, r=0.1, sigma=0.25, T=0.5)
print(f"Call Delta: {delta:.4f}")

# Delta of Dividend paying stock
USD_JPY = 0.8
K = 0.81
r = 0.08
sigma = 0.15
T = 0.5833
rate_foreign = 0.05
delta = currency_option_delta(
    S=USD_JPY, K=K, r=r, sigma=sigma, T=T, rate_foreign=rate_foreign, option_type="call"
)
print(f"Currency Call Delta: {delta:.4f}")

# Gamma of a currency option
gammaValue = currency_option_gamma(
    S=USD_JPY,
    K=K,
    r=r,
    sigma=sigma,
    T=T,
    rate_foreign=rate_foreign,
)
print(f"Currency Call Gamma: {gammaValue:.4f}")

# Vega of a currency option
vegaValue = currency_option_vega(
    S=USD_JPY, K=K, r=r, sigma=sigma, T=T, rate_foreign=rate_foreign
)
print(f"Currency Call Vega: {vegaValue:.4f}")

# Theta of a currency option
thetaValue = currency_option_theta(
    S=USD_JPY, K=K, r=r, sigma=sigma, T=T, rate_foreign=rate_foreign, option_type="call"
)
print(f"Currency Call Theta: {thetaValue:.4f}")

# Theta for one day
oneDayTheta = thetaValue / 365
print(f"Currency Call Theta for one day: {oneDayTheta:.8f}")

# Rho of a currency option
rhoValue = currency_option_rho(
    S=USD_JPY, K=K, r=r, sigma=sigma, T=T, rate_foreign=rate_foreign, option_type="call"
)
print(f"Currency Call Rho: {rhoValue:.4f}")
