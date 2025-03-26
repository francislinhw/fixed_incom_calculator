from datetime import datetime, timedelta

import numpy as np
from fixed_income_clculators.bond_durations import calculate_bond_metrics
from fixed_income_clculators.core import (
    calculate_bond_price_by_binomial_tree,
    continuous_forward_rate,
    continuous_discount_factor,
    convert_continuous_to_simple,
    forward_rate_from_binomial_tree,
    simple_forward_rate,
)
from fixed_income_clculators.bond_futures import (
    calculate_portfolio_hedge_ratio_by_bond_futures,
    conversion_factor,
    find_cheapest_to_deliver,
    theoretical_futures_price_fixed_accrual,
    implied_futures_price_from_rate,
    eurodollar_contract_value,
    convexity_adjusted_forward_rate,
    compute_dirty_price,
    calculate_futures_dollar_duration,
)
from fixed_income_clculators.fixed_income_instrument import FixedIncomeInstrument
from fixed_income_clculators.rates_derivatives import calculate_fixed_swap_rate

# 教材範例測試
dirty_price_dict = compute_dirty_price(
    quote="95-16",
    last_coupon_date="2010-01-10",
    next_coupon_date="2010-07-10",
    today="2010-03-05",
    semiannual_coupon=5.50,
)

print(f"教材範例測試：dirty price: {dirty_price_dict['dirty_price']:.4f}")
print(f"教材範例測試：clean price: {dirty_price_dict['clean_price']:.4f}")
print(f"教材範例測試：accrued interest: {dirty_price_dict['accrued_interest']:.4f}")


# 測試範例一：教材說應該是 1.4623
cf1 = conversion_factor(
    face_value=100,
    coupon_rate=0.10,
    assume_maturity_years=20.0,  # 教材說假設交割日當下還有 20 年整
    discount_rate=0.06,
    periods_per_year=2,
    months_to_delivery=None,  # 沒有從交割日折回今天
    discount_to_today=False,  # ❌ 不折現，教材直接從交割日當估值日
)

print(f"範例一：轉換因子: {cf1:.4f}")  # ✅ 預期為 1.4623

# 教材範例 2：應該是 1.2199
cf2 = conversion_factor(
    face_value=100,
    coupon_rate=0.08,
    assume_maturity_years=18.25,
    discount_rate=0.06,
    periods_per_year=2,
    months_to_delivery=3,
    discount_to_today=True,
)

print(f"範例二：轉換因子: {cf2:.4f}")  # ✅ 預期為 1.2199

# 測試教材範例：
bond_data = [(99.50, 1.0382), (143.50, 1.5188), (119.75, 1.2615)]
futures_price = 93.25

ctd_index, ctd_cost = find_cheapest_to_deliver(bond_data, futures_price)
print(f"最便宜可交割債券 (CTD) 的索引值: {ctd_index}")
print(f"CTD 的交割成本: {ctd_cost:.2f}")


# 教材測試
pricedict = theoretical_futures_price_fixed_accrual(
    quoted_price=115,
    days_since_last_coupon=60,
    days_to_next_coupon=122,
    coupon_amount=6,
    annual_continuous_rate=0.10,
    days_to_delivery=270,
    conversion_factor=1.6,
    accrued_days_at_delivery=148,
    next_coupon_period_days=148 + 35,
)

print(f"教材測試：理論期貨價格: {pricedict['futures_price_cash']:.4f}")
print(f"教材測試：標準化期貨價格: {pricedict['standardized_futures_price']:.4f}")

# Eurodollar futures 報價 Q 對應年化利率
# 測試教材中的利率對應報價
q1 = implied_futures_price_from_rate(0.69)  # 應為 99.3100
q2 = implied_futures_price_from_rate(0.47)  # 應為 99.5300

print(f"教材測試：年化利率 0.69% 對應報價: {q1:.4f}")
print(f"教材測試：年化利率 0.47% 對應報價: {q2:.4f}")

# 測試教材中兩個例子
value1 = eurodollar_contract_value(99.3100)  # 預期應為 998275.0
value2 = eurodollar_contract_value(99.5300)  # 預期應為 998825.0

print(f"教材測試：年化利率 0.69% 對應合約價值: {value1:.4f}")
print(f"教材測試：年化利率 0.47% 對應合約價值: {value2:.4f}")


# convexity adjusted forward rate
# 教材範例：sigma = 0.012, T1 = 8, T2 = 8.25, futures rate = 6.038% = 0.06038
adjusted_forward_rate = convexity_adjusted_forward_rate(
    futures_rate=0.06038, sigma=0.012, T1=8, T2=8.25
)

print(
    f"教材測試：凸性調整後的 forward rate: {adjusted_forward_rate['adjusted_forward_rate']:.4f}"
)

"""
Question: 3
• Today is 10/08/2024
• 30-Year bond issued on 6/15/2000
• Face value $100 with coupon rate of 2.5%
• Quoted price is $98-3
• Settlement is 1 day for Treasury
What is the dirty price?
"""
last_coupon_date = datetime.strptime("2024-06-15", "%Y-%m-%d").strftime("%Y-%m-%d")
next_coupon_date = (
    datetime.strptime(last_coupon_date, "%Y-%m-%d") + timedelta(days=360)
).strftime("%Y-%m-%d")

dirty_price_dict = compute_dirty_price(
    quote="98-3",
    last_coupon_date=last_coupon_date,
    next_coupon_date=next_coupon_date,
    today="2024-10-08",
    semiannual_coupon=2.5,
    settlement_days=1,
)

print(f"題目三：dirty price: {dirty_price_dict['dirty_price']:.4f}")
print(f"題目三：clean price: {dirty_price_dict['clean_price']:.4f}")
print(f"題目三：accrued interest: {dirty_price_dict['accrued_interest']:.4f}")

# Forward Rate
# Example of usage
rates_map = {  # All rates are assume to start at 0
    1: 0.052,  # 5.2% spot rate for year 1
    2: 0.058,  # 5.8% spot rate for year 2
    3: 0.061,  # 6.1% spot rate for year 3
}

# Example: Calculate the forward rate from year 1 to year 2
forward_rate_1_2 = continuous_forward_rate(1, 2, rates_map)
forward_rate_2_3 = continuous_forward_rate(2, 3, rates_map)

discount_factor_0_1 = continuous_discount_factor(0, 1, rates_map)
discount_factor_0_2 = continuous_discount_factor(0, 2, rates_map)
discount_factor_0_3 = continuous_discount_factor(0, 3, rates_map)

print(f"Continuous forward rate from year 1 to year 2: {forward_rate_1_2:.4f}")
print(f"Continuous forward rate from year 2 to year 3: {forward_rate_2_3:.4f}")

print(f"Continuous discount factor from year 0 to year 1: {discount_factor_0_1:.4f}")
print(f"Continuous discount factor from year 0 to year 2: {discount_factor_0_2:.4f}")
print(f"Continuous discount factor from year 0 to year 3: {discount_factor_0_3:.4f}")


# Calculate Bond Futures Dollar Duration Given the delevery bond info
# Find the dollar duration of a forward contract with 2-years of maturity. At maturity, the forward contract
# delivers a 1-year to maturity annual-coupon bond with coupon rate of 6%. You are given the following
# term structure of spot rates, where R(i, j) is the annual rate from year i to year j:
# R(0, 1) = 5.2% R(0, 2) = 5.8% R(0, 3) = 6.1%

continuous_rates_map = {  # continuous spot rates
    1: 0.052,
    2: 0.058,
    3: 0.061,
}

delivery_bond_years_to_maturity_at_delivery = 1

dollar_duration = calculate_futures_dollar_duration(
    continuous_rates_map,
    delivery_year=2,
    delivery_bond_coupon_rate=0.06,
    delivery_bond_years_to_maturity_at_delivery=1,
)

print(f"Dollar duration of the forward contract today: {dollar_duration:.4f}")


# Example usage:
zero_rates = {
    1: 0.04,  # R(0, 1)
    2: 0.042,  # R(0, 2)
    3: 0.045,  # R(0, 3)
    4: 0.048,  # R(0, 4)
    5: 0.051,  # R(0, 5)
}

# Calculate the fixed swap rate with continuous compounding
fixed_swap_rate = calculate_fixed_swap_rate(
    zero_rates, swap_maturity=5, compounding_method="continuous"
)
print(f"Fixed swap rate: {fixed_swap_rate:.8f}")


# Duration Hedge Ratio

hedge_ratio = calculate_portfolio_hedge_ratio_by_bond_futures(
    portfolio_value=15000000,
    portfolio_duration=5.9,
    bond_futures_market_quote="92-4",
    cheapest_to_deliver_bond_duration=7.2,
    target_duration=0,
)

hedge_ratio_target_duration_1_5 = calculate_portfolio_hedge_ratio_by_bond_futures(
    portfolio_value=15000000,
    portfolio_duration=5.9,
    bond_futures_market_quote="92-4",
    cheapest_to_deliver_bond_duration=7.2,
    target_duration=1.5,
)
print(f"Hedge ratio: {hedge_ratio:.4f}")
print(f"Hedge ratio with target duration 1.5: {hedge_ratio_target_duration_1_5:.4f}")


interest_rate_tree = [
    [0.045],  # Year 0 -> 1 parent
    [0.08, 0.035],  # Year 1 -> 2 left, right
    [0.09, 0.045, 0.045, 0.032],  # Year 2 -> 3 left, right, left, right
]


# Calculate the bond price
bond_price = calculate_bond_price_by_binomial_tree(interest_rate_tree)
print(f"The price of the zero-coupon bond at time 0 is: {bond_price:.4f}")

bond_price_2 = calculate_bond_price_by_binomial_tree(interest_rate_tree, maturity=2)
print(f"The price of the zero-coupon bond at time 0 is: {bond_price_2:.4f}")

bond_price_1_year = calculate_bond_price_by_binomial_tree(
    interest_rate_tree, maturity=1
)
print(f"The price of the zero-coupon bond at time 0 is: {bond_price_1_year:.4f}")

# Calculate the forward rate from year 1 to year 2

forward_rate_1_2 = forward_rate_from_binomial_tree(1, 2, interest_rate_tree)
print(f"The forward rate from year 1 to year 2: {forward_rate_1_2:.6f}")


# Problem 1
"""

Suppose the zero rates with continuous compounding are as follows:

 

Maturity( years)

Rate (% per annum)

1

2.0

2

3.0

3

3.7

4

4.2

5

4.5

What is the forward rate for the one year period between the 4th and the 5th year?

"""

zero_rates = {
    1: 0.02,
    2: 0.03,
    3: 0.037,
    4: 0.042,
    5: 0.045,
}

forward_rate = continuous_forward_rate(4, 5, zero_rates)
print(
    f"The forward rate for the one year period between the 4th and the 5th year: {forward_rate:.4f}"
)

# Problem 2
"""
Assuming that the forward rate between year 1 and year 3, f(0, 1, 3), is 5% and the forward rate between
year 3 and year 4, f(0, 3, 4), is 7%. The current four-year zero rate is, R(0, 4), is 6%. What is the current
one-year zero rate R(0, 1)? Assume continuous compounding, f(i, j, k) is the forward rate between year
j and k computed in year i.

"""

forward_rate_1_3 = 0.05
forward_rate_3_4 = 0.07
zero_rate_4 = 0.06

# solve for rate_0_1
# exp^(rate_0_1* 1) * exp^(forward_rate_1_3 * 2) * exp^(forward_rate_3_4 * 1) = exp^(zero_rate_4 * 4)

rate_0_1 = (
    np.log(np.exp(zero_rate_4 * 4))
    - np.log(np.exp(forward_rate_1_3 * 2) * np.exp(forward_rate_3_4 * 1))
) / 1

print(f"The current one-year zero rate: {rate_0_1:.4f}")

# Verify the solution
print(
    f"Verify the solution: {np.exp(rate_0_1 * 1) * np.exp(forward_rate_1_3 * 2) * np.exp(forward_rate_3_4 * 1):.4f}"
)
print(f"Verify the solution: {np.exp(zero_rate_4 * 4):.4f}")

# Problem 3
"""
When the zero-yield curve is flat, the forward interest rate is: 

Group of answer choices

Less than the 4 year zero rate

Greater than the 1-year zero rate

None of these

Equal to the 5-year zero rate

"""

# Assume the zero-yield curve is flat at 5% for all maturities 1, 2, 3, 4, 5
zero_rates = {
    1: 0.05,
    2: 0.05,
    3: 0.05,
    4: 0.05,
    5: 0.05,
}

forward_rate = continuous_forward_rate(1, 2, zero_rates)
print(
    f"The forward rate for the one year period between the 1st and the 2nd year: {forward_rate:.4f}"
)

forward_rate = continuous_forward_rate(2, 3, zero_rates)
print(
    f"The forward rate for the one year period between the 1st and the 3rd year: {forward_rate:.4f}"
)

forward_rate = continuous_forward_rate(3, 4, zero_rates)
print(
    f"The forward rate for the one year period between the 1st and the 4th year: {forward_rate:.4f}"
)

forward_rate = continuous_forward_rate(4, 5, zero_rates)
print(
    f"The forward rate for the one year period between the 1st and the 5th year: {forward_rate:.4f}"
)


# a. Less than the 4 year zero rate
# b. Greater than the 1-year zero rate
# c. None of these
# d. Equal to the 5-year zero rate

# The forward rate is equal to the 5-year zero rate

# Problem 4
"""

Consider the given interest rate tree. A bond has a maturity of 4 years, and the interest rate evolves
according to the provided tree, with each step representing a one-year period. The risk-neutral probability
of the interest rate increasing in each step is 50%. Calculate the price of the zero-coupon bond at time
zero, assuming no arbitrage conditions and a face value of $100. Round your answer to the nearest
integer.

 
"""

interest_rate_tree = [
    [0.03],
    [0.042, 0.028],
    [0.047, 0.03, 0.024],
    [0.05, 0.042, 0.028, 0.021],
]

bond_price = calculate_bond_price_by_binomial_tree(interest_rate_tree, maturity=4)
print(f"The price of the zero-coupon bond at time 0 is: {bond_price:.4f}")


"""

Consider a bond with 30-year bond issued on 12/15/2005 that pays an annual coupon at 2.75%. 
The quoted price of this bond on the Bloomberg terminal is currently $98-11. 
The last time that this bond issued a coupon payment was on 12/15/2024. 
Assume that there are 360 days per year, today is 02/08/2025 and the principal amount is $100. What is the current dirty bond price given 5 settlement days?
"""

last_coupon_date = datetime.strptime("2024-12-15", "%Y-%m-%d").strftime("%Y-%m-%d")
next_coupon_date = (
    datetime.strptime(last_coupon_date, "%Y-%m-%d") + timedelta(days=360)
).strftime("%Y-%m-%d")

dirty_price_dict = compute_dirty_price(
    quote="98-11",
    last_coupon_date=last_coupon_date,
    next_coupon_date=next_coupon_date,
    today="2025-02-08",
    semiannual_coupon=2.75,
    settlement_days=5,
)

print(f"題目四：dirty price: {dirty_price_dict['dirty_price']:.4f}")
print(f"題目四：clean price: {dirty_price_dict['clean_price']:.4f}")
print(f"題目四：accrued interest: {dirty_price_dict['accrued_interest']:.4f}")
