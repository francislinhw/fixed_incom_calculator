from fixed_income_clculators.fixed_income_instrument import FixedIncomeInstrument
from fixed_income_clculators.simple_yield_curve import (
    calculate_yield_curve_from_cashflows,
)
from fixed_income_clculators.duration_hedge import calculate_duration_hedge_ratio
from fixed_income_clculators.bond_durations import calculate_bond_metrics

treasury_price = [98, 99, 102]

treasury_bonds = [
    FixedIncomeInstrument([104], [1], True),
    FixedIncomeInstrument([5, 105], [1, 2], True),
    FixedIncomeInstrument([7, 7, 107], [1, 2, 3], True),
]

# set prices
for i, bond in enumerate(treasury_bonds):
    bond.instrument_information.price = treasury_price[i]

# Calculate the yield curve
spot_rates = calculate_yield_curve_from_cashflows(treasury_bonds)
print("Spot Rates:", spot_rates)


# Coupon Bond
print("Coupon Bond")
discount_rates_coupon_bond = [0.03, 0.033, 0.035, 0.041]
coupon_bond = FixedIncomeInstrument([3, 3, 3, 103], [1, 2, 3, 4], False)

coupon_bond_info = calculate_bond_metrics(discount_rates_coupon_bond, coupon_bond)

print("Zero Coupon Bond")
discount_rates_zcb = [0.03, 0.033, 0.035, 0.041]
zcb = FixedIncomeInstrument([0, 0, 0, 100], [1, 2, 3, 4], False)

zcb_info = calculate_bond_metrics(discount_rates_zcb, zcb)


print("Aromotizing Bond")
discount_rates_aromotizing_bond = [
    0.05,
    0.053,
    0.056,
    0.059,
    0.062,
    0.065,
    0.068,
    0.071,
    0.074,
    0.077,
    0.08,
    0.083,
    0.086,
    0.089,
    0.092,
    0.095,
    0.098,
    0.101,
    0.104,
    0.107,
]


aromotizing_bond = FixedIncomeInstrument(
    [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    False,
)

aromotizing_bond_info = calculate_bond_metrics(
    discount_rates_aromotizing_bond, aromotizing_bond
)


print("Backloaded Bond")
discount_rates_backloaded_bond = [0.03, 0.033, 0.035, 0.041]

backloaded_bond = FixedIncomeInstrument([5, 5, 10, 90], [1, 2, 3, 4], False)

backloaded_bond_info = calculate_bond_metrics(
    discount_rates_backloaded_bond, backloaded_bond
)


print("Frontloaded Bond")
discount_rates_frontloaded_bond = [0.03, 0.033, 0.035, 0.041]

frontloaded_bond = FixedIncomeInstrument([90, 10, 5, 5], [1, 2, 3, 4], False)

frontloaded_bond_info = calculate_bond_metrics(
    discount_rates_frontloaded_bond, frontloaded_bond
)


print("Hedge Ratio")
ratio = calculate_duration_hedge_ratio(
    coupon_bond,
    aromotizing_bond,
    discount_rates_coupon_bond,
    discount_rates_aromotizing_bond,
)

ratio_backloaded = calculate_duration_hedge_ratio(
    backloaded_bond,
    aromotizing_bond,
    discount_rates_backloaded_bond,
    discount_rates_aromotizing_bond,
)

ratio_frontloaded = calculate_duration_hedge_ratio(
    frontloaded_bond,
    aromotizing_bond,
    discount_rates_frontloaded_bond,
    discount_rates_aromotizing_bond,
)

ratio_zcb = calculate_duration_hedge_ratio(
    zcb,
    aromotizing_bond,
    discount_rates_zcb,
    discount_rates_aromotizing_bond,
)

print("Aromotizing Bond Hedged by Coupon Bond:", ratio)
print("Aromotizing Bond Hedged by Backloaded Bond:", ratio_backloaded)
print("Aromotizing Bond Hedged by Frontloaded Bond:", ratio_frontloaded)
print("Aromotizing Bond Hedged by Zero Coupon Bond:", ratio_zcb)

print("\n\n")

# New Bond
print("*** New Bond ***")
new_bond = FixedIncomeInstrument([8, 8, 8, 8, 108], [1, 2, 3, 4, 5], False)
discount_rates_new_bond = [0.059, 0.062, 0.069, 0.071, 0.075]
new_bond_info = calculate_bond_metrics(discount_rates_new_bond, new_bond)

print("New Bond:", new_bond_info)

"""
Government wants to raise money for School projects by issuing bond of 
Face value $100 that pays a coupon on 6% per year annually with maturity of 4 years. 
Find the duration and modified duration of this bond. The yield curve is shown below.

Treasury Yield Curve:

Time(Yrs)
1 2 3 4

Rate
0.057 0.063 0.068 0.072
"""

print("*** Government Bond ***")
government_bond = FixedIncomeInstrument([6, 6, 6, 106], [1, 2, 3, 4], False)
discount_rates_government_bond = [0.057, 0.063, 0.068, 0.072]

government_bond_info = calculate_bond_metrics(
    discount_rates_government_bond, government_bond
)

print("\n\n")
"""
Bond 1 is a 8% coupon bond with Face value of $100 and 9 years to maturity.
Bond 2 is a 4% coupon bond with a face value of $100 and 8 years to maturity.
If both bonds have the same YTM of 8% which has higher duration?

"""
print("*** Bond 1 ***")
bond_1 = FixedIncomeInstrument(
    [8, 8, 8, 8, 8, 8, 8, 8, 108], [1, 2, 3, 4, 5, 6, 7, 8, 9], False
)
discount_rates_bond_1 = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]

bond_1_info = calculate_bond_metrics(discount_rates_bond_1, bond_1)

print("*** Bond 2 ***")
bond_2 = FixedIncomeInstrument(
    [4, 4, 4, 4, 4, 4, 4, 104], [1, 2, 3, 4, 5, 6, 7, 8], False
)
discount_rates_bond_2 = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]

bond_2_info = calculate_bond_metrics(discount_rates_bond_2, bond_2)
