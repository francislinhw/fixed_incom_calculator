from fixed_income_clculators.binomial import (
    binomial_custom_tree_option_price,
    binomial_option_price,
)
import numpy as np

# Question 1:
# 100	T	1
# sigma = 0.4	time step	0.5
# 0.05	K	100
# 0
# 1.326896441
# 0.753638316


def call_payoff(prices, K):
    return max(prices[-1] - K, 0)


def put_payoff(prices, K):
    return max(K - prices[-1], 0)


call_option_price = binomial_option_price(
    call_payoff, 100, 1, 2, 0.05, 0, "European", u=1.326896441, d=0.753638316, K=100
)

print("call_option_price", call_option_price)

put_option_price = binomial_option_price(
    put_payoff, 100, 1, 2, 0.05, 0, "European", u=1.326896441, d=0.753638316, K=100
)

print("put_option_price", put_option_price)

call_american_option_price = binomial_option_price(
    call_payoff, 100, 1, 2, 0.05, 0.03, "American", u=1.326896441, d=0.753638316, K=100
)

print("call_american_option_price", call_american_option_price)


def asian_payoff(prices, K):
    return max(np.mean(prices) - K, 0)  # Example Asian option payoff


asian_option_price = binomial_option_price(
    asian_payoff, 100, 1, 2, 0.05, 0, "European", u=1.326896441, d=0.753638316, K=100
)

print("asian_option_price", asian_option_price)


def forward_start_option_payoff(prices):
    return max(prices[-1] - prices[1], 0)


forward_start_option_price = binomial_option_price(
    forward_start_option_payoff,
    100,
    2,
    2,
    0.05,
    0,
    "European",
    u=1.491824698,
    d=0.670320046,
)

print("forward_start_option_price", forward_start_option_price)

# custom tree
# List[List[Dict[str, float]]]
#
# 			240
# 		230
# 	220		220
# 200		210
# 		200
# 	190		190
# 		180
# 			170
# Example usage for the provided custom tree structure
valid_custom_tree = [
    [200],  # Layer 0
    [220, 190],  # Layer 1
    [230, 210, 200, 180],  # Layer 2
    [240, 220, 220, 190, 220, 190, 190, 170],  # Layer 3
]


custom_tree_option_price = binomial_custom_tree_option_price(
    put_payoff, valid_custom_tree, 0.75, 0, 0, "European", K=200
)

print("custom_tree_option_price", custom_tree_option_price)

# Question 2:
# 		160
# 	130
# 100	100
# 	80
# 		60

valid_custom_tree = [
    [100],
    [130, 80],
    [160, 100, 100, 60],
]

custom_tree_option_price = binomial_custom_tree_option_price(
    call_payoff, valid_custom_tree, 1, 0.05, 0, "European", K=100
)

print("custom_tree_option_price", custom_tree_option_price)


# Option on Futures dividend yield = risk free rate is key
call_future_option_price = binomial_option_price(
    call_payoff, 50, 0.5, 2, 0.1, 0.1, "European", u=None, d=None, sigma=0.4, K=50
)

print("call_future_option_price", call_future_option_price)
# official test 1
# test question 1
"""
A non-dividend paying stock is trading at $200. The volatility is at 50%. 
Risk free rate is at 2%. 

Price an at the money 2-year American put option using a 4 step tree. 

Use the CRR procedure. What is the price?

"""
# use python to implement


def american_put_payoff(prices, K):
    return max(K - prices[-1], 0)


american_put_option_price = binomial_option_price(
    american_put_payoff,
    200,
    2,
    4,
    0.02,
    0,
    "American",
    u=None,
    d=None,
    sigma=0.5,
    K=200,
)

print("Test 1american_put_option_price", american_put_option_price)

# test question 2
"""

Elite Investments is using the stock tree shown below for pricing a binary option. The stock is at $120 now. The payoff from binary option at expiration in 2 years, is $200 if the stock is above $120 and $0 otherwise.

The interest rate is constant at 4%.

What is the price of the binary option?

Stock	 	 
         145
    135	
120		115
    110	
        100
"""


def binary_option_payoff(prices):
    return 200 if prices[-1] > 120 else 0


custom_trees = [
    [120],
    [135, 110],
    [145, 115, 115, 100],
]

binary_option_price = binomial_custom_tree_option_price(
    binary_option_payoff,
    custom_trees,
    2,
    0.04,
    0,
    "European",
)

print("Test 2 binary_option_price", binary_option_price)

# test question 7
"""
You have $10,000 to invest. You are interested in a stock that is trading at $90 with a volatility of 20%. You are going to use 2 year options with strikes $85 and $95 in order to maximize your expected profit. You are not willing to sell more than 200 of any of the options. What positions in the two options will achieve this objective? Assume the risk free rate is constant at 4%.

To arrive at a solution create a 4-step CRR tree for the stock. It results in 4 possible outcomes (at the end of 2 years) for the stock along with its probabilities. Price the two options on this CRR tree. Carry out the expected profit maximization with your budget constraint. You may simply use excel (as we did in class) or code in matlab or R or python.

"""
