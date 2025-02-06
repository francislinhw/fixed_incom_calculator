from fixed_income_clculators.fixed_income_instrument import FixedIncomeInstrument
from fixed_income_clculators.simple_yield_curve import (
    calculate_yield_curve_from_cashflows,
)
from fixed_income_clculators.duration_hedge import calculate_duration_hedge_ratio
from fixed_income_clculators.bond_durations import calculate_bond_metrics
from fixed_income_clculators.core import (
    PV,
    find_fixed_payment_in_times_to_fit_future_value,
    YTM,
)
from fixed_income_clculators.core import (
    PV_perpetuity,
    find_the_fixed_payment_in_certain_times_to_fit_future_value,
    find_the_fixed_payment_in_certain_times_to_fit_present_value,
    find_fixed_payment_in_times_to_fit_present_value,
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
