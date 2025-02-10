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


# Test 1 Official
print("\n--------------------------------\n")
print("\nTest 1 Official\n")
print("\n *** Question 1 *** \n")
"""
Question 1:


"""


print("\n *** Question 2 *** \n")
"""
Question 2:


"""

print("\n *** Question 3 *** \n")
"""
Question 3:


"""

print("\n *** Question 4 *** \n")
"""
Question 4:


"""

print("\n *** Question 5 *** \n")
"""
Question 5:


"""

print("\n *** Question 6 *** \n")
"""
Question 6:


"""

print("\n *** Question 7 *** \n")
"""
Question 7:


"""
