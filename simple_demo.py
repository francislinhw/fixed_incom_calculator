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
)

# 10 year 7% 100 PV coupon bond with price 106

cashflows = [7, 7, 7, 7, 7, 7, 7, 7, 7, 107]

ytm = YTM(
    price=106,
    cashflows=cashflows,
    times=list(range(1, 11)),
)

bond_matrics = calculate_bond_metrics(
    discount_rates=[ytm] * 10,
    instrument=FixedIncomeInstrument(
        cashflows, [i for i in range(1, 11)], is_treasury=False
    ),
)

print("YTM: ", ytm)


"""
 Sally has just started working and she is going to save $2000 a year for
the first 5 years.
• Then she hopes to get promoted and save a reasonable amount every
year for the next 20 years. These are her prime working years.
• After that she will begin to wind down and save $3000 for the next 3
years and finally $1000 for the next 4 years after which she will retire.
• She wants to retire and spend $50,000 per year for as long as she
needs (perpetuity).
• Rate is assumed to be 5% throughout. How much should she save
during her prime working years?
"""

cashflows = [2000] * 5 + [0] * 20 + [3000] * 3 + [1000] * 4

fv_of_perpetuity = PV_perpetuity(50000, 0.05)

perpetuity_cashflow = [0] * 31 + [fv_of_perpetuity]

PV_of_perpetuity = PV(perpetuity_cashflow, [0.05] * 32, [i for i in range(1, 33)])

fixed_payment = find_the_fixed_payment_in_certain_times_to_fit_present_value(
    present_value=PV_of_perpetuity,
    cashflows=cashflows,
    discount_rates=[0.05] * 32,
    fixed_payment_times=[i for i in range(6, 26)],
)

print("Fixed Payment: ", fixed_payment)

# verify that the fixed payment is correct
verify_cashflows = [2000] * 5 + [fixed_payment] * 20 + [3000] * 3 + [1000] * 4
present_value = PV(verify_cashflows, [0.05] * 32, [i for i in range(1, 33)])
print("Present Value: Annuity ", PV_of_perpetuity)
print("Present Value: Verify ", present_value)

fixed_payment = find_the_fixed_payment_in_certain_times_to_fit_future_value(
    future_value=fv_of_perpetuity,
    cashflows=cashflows,
    discount_rates=[0.05] * 32,
    fixed_payment_times=[i for i in range(6, 26)],
)

print("Fixed Payment: ", fixed_payment)

# verify that the fixed payment is correct
verify_cashflows = [2000] * 5 + [fixed_payment] * 20 + [3000] * 3 + [1000] * 4
present_value = PV(verify_cashflows, [0.05] * 32, [i for i in range(1, 33)])
print("Present Value: Annuity ", PV_of_perpetuity)
print("Present Value: Verify ", present_value)


# Example usage of find_the_optimal_weights_to_match_the_target_liability_by_different_cashflows
"""
    • An insurance company has to
    pay out $50,000 every year for
    the next 5 years.
    • The bonds shownare available to
    trade.
    • Find the cheapest solution to the
    liability problem.
    • Excess funds carry over to the
    next year without any interest.
    Bond(FV=100)
    Coupon(a
    nnual) Term
    Price
    ($)
    1 5% 1 98
    2 7% 3 100
    3 4% 4 94
    4 3% 5 92
    5 0% 5 89
"""

list_candidate_bonds = {
    "bond1": [105],
    "bond2": [7, 7, 107],
    "bond3": [4, 4, 4, 104],
    "bond4": [3, 3, 3, 3, 103],
    "bond5": [0, 0, 0, 0, 100],
}

list_candidate_bonds_prices = {
    "bond1": 98,
    "bond2": 100,
    "bond3": 94,
    "bond4": 92,
    "bond5": 89,
}

list_times = [1, 2, 3, 4, 5]

target_liability = [50000, 50000, 50000, 50000, 50000]

optimal_weights = (
    find_the_optimal_weights_to_match_the_target_liability_by_different_cashflows(
        list_candidate_bonds=list_candidate_bonds,
        list_candidate_bonds_prices=list_candidate_bonds_prices,
        list_times=list_times,
        target_liability=target_liability,
    )
)

print("Optimal Weights: ", optimal_weights)

# Calculate the total cost of the optimal weights
total_cost = sum(
    [
        list_candidate_bonds_prices[bond] * optimal_weights[0][bond]
        for bond in list_candidate_bonds
    ]
)

print("Total Cost: ", total_cost)

# hedge ratio
"""
Vanguard has a bond portfolio that is funding the retirement
of a senior community.
• The value of the bond portfolio is $20 million and the
modified duration is 15.
• Vanguard wants to hedge the portfolio using a 6% coupon
bond with Face Value of $100 and 10 years to maturity
whose YTM is 7%.
• How many of these bonds are needed?
• Why would Vanguard want to hedge in the first place?

"""

vanguard_portfolio_value = 20000000
vanguard_portfolio_duration = 15

bond_metrics = calculate_bond_metrics(
    discount_rates=[0.07] * 10,
    instrument=FixedIncomeInstrument(
        [6] * 9 + [106], [i for i in range(1, 11)], is_treasury=False
    ),
)

value_of_bond_needed = (
    vanguard_portfolio_value
    * vanguard_portfolio_duration
    / bond_metrics.modified_duration
)

number_of_bonds_needed = value_of_bond_needed / bond_metrics.price

print("Number of Bonds Needed: ", number_of_bonds_needed)

# Example usage of maximize_the_expected_return_of_a_portfolio_with_a_given_constraints
"""
• You have $100 to invest. The following three investment vehicles are
available.
• Investment 1: Every dollar invested now yields $0.10 a year from now, and
$1.30 three years from now.
• Investment 2: Every dollar invested now yields $0.20 a year from now and
$1.10 two years from now.
• Investment 3: Every dollar invested one year from now yields $1.50 three
years from now.
• During each year leftover cash yields nothing. The most in each investment
should be $50.
• How should you invest so as to maximize the cash three years from now?


"""

# Example usage:
cash_flows = {
    "bond1": [
        0.1,
        0,
        1.3,
    ],  # Bond 1 yields 0.1 in Year 1, 0 in Year 2, and 1.3 in Year 3
    "bond2": [
        0.2,
        1.1,
        0,
    ],  # Bond 2 yields 0.2 in Year 1, 0.2 in Year 2, and 1.1 in Year 3
    "bond3": [0, 0, 1.5],  # Bond 3 yields 0 in Year 1, 1.5 in Year 2, and 0 in Year 3
}

bond_prices = {
    "bond1": 1,  # Price of Bond 1
    "bond2": 1,  # Price of Bond 2
    "bond3": 1,  # Price of Bond 3
}

total_budget = 100  # Total investment available
max_investment_per_bond = 50  # Maximum investment per bond

optimal_allocations, max_return = (
    maximize_the_expected_return_of_a_portfolio_with_a_given_constraints(
        cash_flows, bond_prices, total_budget, max_investment_per_bond
    )
)

print("Optimal Allocations:", optimal_allocations)
print("Maximum Return:", max_return)


# Example usage of auction_revenue_with_constraints
"""
    bids = [
        ([0], 6),  # Bid 1: Item 1, Price $6
        ([1], 3),  # Bid 2: Item 2, Price $3
        ([2, 3], 12),  # Bid 3: Item 3 and 4, Price $12
        ([0, 2], 12),  # Bid 4: Item 1 and 3, Price $12
        ([1, 3], 8),  # Bid 5: Item 2 and 4, Price $8
        ([0, 2, 3], 16),  # Bid 6: Item 1, 3, and 4, Price $16
        ([0, 1, 2], 13),  # Bid 7: Item 1, 2, and 3, Price $13
    ]
"""

bids = [
    ([0], 6),  # Bid 1: Item 1, Price $6
    ([1], 3),  # Bid 2: Item 2, Price $3
    ([2, 3], 12),  # Bid 3: Item 3 and 4, Price $12
    ([0, 2], 12),  # Bid 4: Item 1 and 3, Price $12
    ([1, 3], 8),  # Bid 5: Item 2 and 4, Price $8
    ([0, 2, 3], 16),  # Bid 6: Item 1, 3, and 4, Price $16
    ([0, 1, 2], 13),  # Bid 7: Item 1, 2, and 3, Price $13
]

number_of_bonds_selected = 1

auction_revenue, selected_bids = auction_revenue_with_constraints(
    bids, number_of_bonds_selected
)

print("Auction Revenue: ", auction_revenue)
print("Selected Bids: ", selected_bids)

# practice test
"""
 Bonds A, B, C, D, E are for sale. There are 2 units of each bond
for sale.
The following bids have come in:
• ({A, C, D}, 7)
• ({B, E}, 7)
• ({C}, 3)
• ({A, B, C, E}, 9)
• ({D}, 4)
• ({A, B, C}, 5)
• ({B, D}, 5)
• Select the winning bids so that the revenue is maximized

"""

# reference table
"""
A : 0
B : 1
C : 2
D : 3
E : 4
"""

bids = [
    ([0, 2, 3], 7),
    ([1, 4], 7),
    ([2], 3),
    ([0, 1, 2, 4], 9),
    ([3], 4),
    ([0, 1, 2], 5),
    ([1, 3], 5),
]

number_of_bonds_selected = 1

auction_revenue, selected_bids = auction_revenue_with_constraints(
    bids, number_of_bonds_selected
)

print("Auction Revenue: ", auction_revenue)
print("Selected Bids: ", selected_bids)
