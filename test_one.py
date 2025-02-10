import numpy as np
from fixed_income_clculators.fixed_income_instrument import FixedIncomeInstrument
from fixed_income_clculators.simple_yield_curve import (
    calculate_yield_curve_from_cashflows,
)
from fixed_income_clculators.duration_hedge import calculate_duration_hedge_ratio
from fixed_income_clculators.bond_durations import calculate_bond_metrics
from fixed_income_clculators.core import (
    PV,
    auction_revenue_with_constraints,
    find_fixed_payment_in_times_to_fit_future_value,
    YTM,
)
from fixed_income_clculators.core import (
    PV_perpetuity,
    find_the_fixed_payment_in_certain_times_to_fit_future_value,
    find_the_fixed_payment_in_certain_times_to_fit_present_value,
    find_fixed_payment_in_times_to_fit_present_value,
    find_the_optimal_weights_to_match_the_target_liability_by_different_cashflows,
    maximize_the_expected_return_of_a_portfolio_with_a_given_constraints,
    optimize_investments,
)


print("\n *** Question 3 *** \n")
"""
Question 3:

You have $100 to invest. The following three investment vehicles are available.
Investment 1: Every dollar invested now yields $0.15 a year from now, and $1.50 three years from now.
Investment 2: Every dollar invested now yields $0.30 a year from now and $1.25 two years from now.
Investment 3: Every dollar invested one year from now yields $2.00 three years from now.
During each year leftover cash yields nothing. The most in each investment should be $60.
How should you invest so as to maximize the cash three years from now?
"""

investment_1 = [-1, 0.15, 0, 1.5]
investment_2 = [-1, 0.3, 1.25, 0]
investment_3 = [0, -1, 0, 2]

cashflows = np.array([investment_1, investment_2, investment_3])

constraints = [60, 60, 60]

investments, final_cash = optimize_investments(
    cashflows=cashflows, budget=100, max_per_investments=constraints
)

print("Optimal Investments:")
for inv in investments:
    print(f"• Invest ${inv[2]:.2f} in Investment {inv[0]+1} at period {inv[1]}")
print(f"\nFinal Cash after 3 years: ${final_cash:.2f}")

# Answer:
# Optimal Investments:
# • Invest $57.14 in Investment 2 at period 0
# • Invest $60.00 in Investment 3 at period 1

# Final Cash after 3 years: $191.43


print("\n *** Question 5 *** \n")
"""
Question 5:

Bonds A, B, C, D, E are for sale. There is 3 units of each bond for sale. The following bids have come in:
• {(A), $8}
• {(B, C, E), $6}
• {(A, E), $4}
• {(C, E), $10}
• {(D, E), $5}
• {(A, B), $7}
• {(A, D), $6}
How should you allocate the bonds to maximize the revenue?


"""

# Reference
# A : 0
# B : 1
# C : 2
# D : 3
# E : 4

# bids form the question
bids = [
    ([0], 8),
    ([1, 2, 4], 6),
    ([0, 4], 4),
    ([2, 4], 10),
    ([3, 4], 5),
    ([0, 1], 7),
    ([0, 3], 6),
]

number_of_bonds_selected = 3

auction_revenue, selected_bids = auction_revenue_with_constraints(
    bids, number_of_bonds_selected
)

print(f"Auction Revenue: ${auction_revenue:.2f}")
print(f"Selected Bids: {selected_bids}")

# Answer:
# Optimal Solution:
# uction Revenue: $42.00
# elected Bids: ['Bid 1', 'Bid 2', 'Bid 4', 'Bid 5', 'Bid 6', 'Bid 7']

# print("\n *** Question 6 *** \n")
# """
# Question 6:
# John has just started working and he is going to save $2500 a year for the first 4 years.
# Then he hopes to get promoted and save a reasonable amount every year for the next 15 years.
# These are his prime working years.
# After that he will begin to wind down and save $4000 for the next 5 years and finally $1200 for the next 3 years after which he will retire.
# He wants to retire and spend $60,000 per year for as long as he needs (perpetuity).
# The rate is assumed to be 4% throughout. How much should he save during his prime working years?

# """

# cashflows = [2500] * 4 + [0] * 15 + [4000] * 5 + [1200] * 3

# fv_of_perpetuity = PV_perpetuity(60000, 0.04)

# perpetuity_cashflow = [0] * 26 + [fv_of_perpetuity]

# PV_of_perpetuity = PV(perpetuity_cashflow, [0.04] * 27, [i for i in range(1, 28)])

# fixed_payment = find_the_fixed_payment_in_certain_times_to_fit_present_value(
#     present_value=PV_of_perpetuity,
#     cashflows=cashflows,
#     discount_rates=[0.04] * 27,
#     fixed_payment_times=[i for i in range(5, 20)],
# )

# print("Fixed Payment: ", fixed_payment)

# # verify that the fixed payment is correct
# verify_cashflows = [2500] * 4 + [fixed_payment] * 15 + [4000] * 5 + [1200] * 3
# present_value = PV(verify_cashflows, [0.04] * 27, [i for i in range(1, 28)])
# print("Present Value: Annuity ", PV_of_perpetuity)
# print("Present Value: Verify ", present_value)

# print("\n *** Question 7 *** \n")
# """
# Question 7:

# BlackRock has a bond portfolio that is funding the retirement of a senior community.
# The value of the bond portfolio is $10 million and the modified duration is 10.
# BlackRock wants to hedge the portfolio using a 12% coupon bond with a Face Value of $100, with 10 years to maturity, and a YTM of 7%.
# How many of these bonds are needed?

# """

# vanguard_portfolio_value = 10000000
# vanguard_portfolio_duration = 10

# bond_metrics = calculate_bond_metrics(
#     discount_rates=[0.07] * 10,
#     instrument=FixedIncomeInstrument(
#         [12] * 9 + [112], [i for i in range(1, 11)], is_treasury=False
#     ),
# )

# value_of_bond_needed = (
#     vanguard_portfolio_value
#     * vanguard_portfolio_duration
#     / bond_metrics.modified_duration
# )

# number_of_bonds_needed = value_of_bond_needed / bond_metrics.price

# print("Number of Bonds Needed: ", number_of_bonds_needed)
