from typing import List, Optional, Tuple
import math
from fixed_income_clculators.bond_durations import calculate_bond_metrics
from fixed_income_clculators.core import (
    convert_continuous_to_simple,
    simple_forward_rate,
    continuous_discount_factor,
)
from fixed_income_clculators.fixed_income_instrument import FixedIncomeInstrument


def eurodollar_contract_value(q: float, notional: int = 10_000) -> float:
    """
    計算 Eurodollar 期貨合約的理論價格（以報價 Q 為基礎）

    Parameters:
        q (float): 報價（如 99.3100）
        notional (int): 合約本金（預設 $10,000）

    Returns:
        float: 合約價值（理論交割價）
    """
    implied_rate = 100 - q  # 由報價倒推出利率 (%)
    interest_for_3_months = 0.25 * implied_rate
    price = 100 - interest_for_3_months  # 理論債券價格
    contract_value = price * notional
    return round(contract_value, 3)


def implied_futures_price_from_rate(interest_rate: float) -> float:
    """
    根據市場利率（年化）轉換成 Eurodollar futures 報價 Q。

    Parameters:
        interest_rate (float): 年化利率（百分比形式，如 0.69 表示 0.69%）

    Returns:
        float: 對應的期貨報價 Q（如 99.3100）
    """
    return round(100 - interest_rate, 4)


def theoretical_futures_price_fixed_accrual(
    quoted_price: float,
    days_since_last_coupon: int,
    days_to_next_coupon: int,
    coupon_amount: float,
    annual_continuous_rate: float,
    days_to_delivery: int,
    conversion_factor: float,
    accrued_days_at_delivery: int,
    next_coupon_period_days: int,
) -> dict:
    """
    修正版：使用教材定義的應計利息比例公式 (60 / (60+122)) 以對齊現金價格計算。
    """

    # Step 1: 應計利息以「過去 + 未來」的週期計算
    full_coupon_cycle = days_since_last_coupon + days_to_next_coupon
    accrued_interest_today = (
        days_since_last_coupon / full_coupon_cycle
    ) * coupon_amount
    cash_price = quoted_price + accrued_interest_today

    # Step 2: 折現未來的息票
    time_to_coupon_years = days_to_next_coupon / 365
    pv_coupon = coupon_amount * math.exp(-annual_continuous_rate * time_to_coupon_years)

    # Step 3: 現金期貨價格（加上 carry cost）
    T = days_to_delivery / 365
    futures_price_cash = (cash_price - pv_coupon) * math.exp(annual_continuous_rate * T)

    # Step 4: 扣掉交割日的應計利息
    accrued_at_delivery = coupon_amount * (
        accrued_days_at_delivery / next_coupon_period_days
    )
    quoted_futures_price = futures_price_cash - accrued_at_delivery

    # Step 5: 標準化期貨價格（除以 conversion factor）
    standardized_price = quoted_futures_price / conversion_factor

    return {
        "cash_price": round(cash_price, 3),
        "pv_coupon": round(pv_coupon, 3),
        "futures_price_cash": round(futures_price_cash, 3),
        "quoted_futures_price": round(quoted_futures_price, 3),
        "standardized_futures_price": round(standardized_price, 3),
    }


def find_cheapest_to_deliver(
    bonds: List[Tuple[float, float]], futures_price: float
) -> Tuple[int, float]:
    """
    給定債券列表與期貨價格，找出最便宜可交割債券 (Cheapest to Deliver, CTD)

    Parameters:
        bonds: List of tuples, 每個元素為 (quoted_price, conversion_factor)
        futures_price: 當前期貨結算價格 (如 93.25)

    Returns:
        Tuple: (CTD 的索引值（從1開始）, 對應的交割成本)
    """
    costs = []
    for quoted_price, conversion_factor in bonds:
        invoice_price = futures_price * conversion_factor
        delivery_cost = quoted_price - invoice_price
        costs.append(delivery_cost)

    min_cost = min(costs)
    ctd_index = costs.index(min_cost) + 1  # +1 為了讓 index 從 1 開始
    return ctd_index, round(min_cost, 2)


def conversion_factor(
    face_value,
    coupon_rate,
    assume_maturity_years,
    discount_rate,
    periods_per_year=2,
    months_to_delivery=None,
    discount_to_today=False,
):
    """
    最終正確版本，完全符合教材邏輯（例如補全息票 + 折現回今天 + 扣應計利息）
    """

    coupon = face_value * coupon_rate / periods_per_year
    period_discount_rate = discount_rate / periods_per_year
    step = 1 / periods_per_year

    # 將假設到期時間拆成整數部分 + 非整數部分
    rounded_years = int(assume_maturity_years / step) * step
    remaining_fraction = assume_maturity_years - rounded_years
    total_periods = int(rounded_years * periods_per_year)

    # Step 1: 折現息票與本金 (從1期開始)
    pv_at_delivery = sum(
        coupon / (1 + period_discount_rate) ** i for i in range(1, total_periods + 1)
    )
    pv_at_delivery += face_value / (1 + period_discount_rate) ** total_periods

    # Step 2: 如果 maturity 不是付息週期的整數倍，加一整期息票（教材邏輯）
    if remaining_fraction > 0:
        pv_at_delivery += coupon  # 補「+4」或「+5」那筆錢（教材做法）

    # Step 3: 折回今天（教材說 3 個月用 sqrt(1.03)）
    if discount_to_today and months_to_delivery is not None:
        if months_to_delivery == 3:
            back_discount = (1 + period_discount_rate) ** (months_to_delivery / 6)
        else:
            back_discount = (1 + period_discount_rate) ** (months_to_delivery / 6)
        pv_today = pv_at_delivery / back_discount
    else:
        pv_today = pv_at_delivery

    # Step 4: 應計利息（只針對不是整數期時才扣）
    accrued_interest = (
        coupon * (remaining_fraction / step) if remaining_fraction > 0 else 0
    )

    # Step 5: 轉換因子
    clean_price = pv_today - accrued_interest
    cf = clean_price / face_value
    return round(cf, 4)


def convexity_adjusted_forward_rate(
    futures_rate: float, sigma: float, T1: float, T2: float
):
    """
    根據凸性調整公式計算 forward rate：
    Forward Rate = Futures Rate - (1/2) * σ^2 * T1 * T2

    Parameters:
        futures_rate (float): 期貨利率（連續複利，百分比形式）
        sigma (float): 利率的年化標準差（如 0.012 表示 1.2%）
        T1 (float): Futures 剩餘年數（例如 8）
        T2 (float): Forward 合約對應的年數（例如 8.25）

    Returns:
        dict: 包含調整值（% 與 bps）、調整後 forward rate
    """
    convexity_adjustment = 0.5 * sigma**2 * T1 * T2
    forward_rate = futures_rate - convexity_adjustment

    return {
        "futures_rate": round(futures_rate * 100, 4),  # %
        "convexity_adjustment (%)": round(convexity_adjustment * 100, 4),
        "convexity_adjustment (bps)": round(convexity_adjustment * 10000, 1),
        "adjusted_forward_rate": round(forward_rate * 100, 4),  # %
    }


from datetime import datetime


def parse_quoted_price(quote: str) -> float:
    """
    將類似 95-16 的報價轉為十進位價格
    """
    parts = quote.split("-")
    whole = int(parts[0])
    fraction = int(parts[1]) if len(parts) > 1 else 0
    return whole + fraction / 32


def compute_dirty_price(
    quote: str,
    last_coupon_date: str,
    next_coupon_date: str,
    today: str,
    semiannual_coupon: float,
    settlement_days: int = 1,
) -> dict:
    """
    根據報價與應計利息計算 dirty price（實際付款價格）上一期的利息歸屬沒問題 是在調整下一期的利息

    Parameters:
        quote (str): 報價字串，如 "95-16"
        last_coupon_date (str): 上次付息日，格式 "YYYY-MM-DD"
        next_coupon_date (str): 下次付息日，格式 "YYYY-MM-DD"
        today (str): 今日日期，格式 "YYYY-MM-DD"
        semiannual_coupon (float): 半年息票金額（如 5.50）

    Returns:
        dict: 包含 clean price、應計利息、dirty price
    """
    clean_price = parse_quoted_price(quote)
    d0 = datetime.strptime(last_coupon_date, "%Y-%m-%d")
    d1 = datetime.strptime(today, "%Y-%m-%d")
    d2 = datetime.strptime(next_coupon_date, "%Y-%m-%d")

    days_since_last = (d1 - d0).days
    days_in_period = (d2 - d0).days

    accrued_interest = (
        (days_since_last + settlement_days) / (days_in_period + settlement_days)
    ) * semiannual_coupon
    dirty_price = clean_price + accrued_interest

    return {
        "clean_price": round(clean_price, 3),
        "accrued_interest": round(accrued_interest, 3),
        "dirty_price": round(dirty_price, 3),
    }


def calculate_futures_dollar_duration(
    continuous_rates_map: dict,
    delivery_year: int,
    delivery_bond_coupon_rate: float,
    delivery_bond_years_to_maturity_at_delivery: int,
):
    # check if Len of continuous_rates_map is equal to delivery_bond_years_to_maturity_at_delivery
    if (
        len(continuous_rates_map)
        < delivery_bond_years_to_maturity_at_delivery + delivery_year
    ):
        raise ValueError(
            "Length of continuous_rates_map must be equal to delivery_bond_years_to_maturity_at_delivery"
        )

    delivery_bond = FixedIncomeInstrument(
        cashflows=[
            100 * delivery_bond_coupon_rate
            for _ in range(delivery_bond_years_to_maturity_at_delivery - 1)
        ]
        + [100 + 100 * delivery_bond_coupon_rate],
        times=[1 + i for i in range(delivery_bond_years_to_maturity_at_delivery)],
        is_treasury=False,
    )

    simple_rates_map = convert_continuous_to_simple(continuous_rates_map)

    discount_factor = []
    for i in range(delivery_bond_years_to_maturity_at_delivery):
        discount_factor.append(
            simple_forward_rate(
                delivery_year + i, delivery_year + i + 1, simple_rates_map
            )
        )

    bond_info = calculate_bond_metrics(
        discount_factor, delivery_bond, continuous_rates_map
    )

    bond_price = bond_info.price
    bond_duration = bond_info.duration

    discount_factor = continuous_discount_factor(0, delivery_year, continuous_rates_map)

    dollar_duration_today = bond_duration * bond_price * discount_factor

    print(f"Bond price: {bond_price:.4f}")
    print(f"Bond duration: {bond_duration:.4f}")
    print(f"Discount factor from year 0 to year 2: {discount_factor:.4f}")

    return dollar_duration_today


def calculate_portfolio_hedge_ratio_by_bond_futures(
    portfolio_value: float,
    portfolio_duration: float,
    bond_futures_market_quote: str,  # 例如 "95-16"
    cheapest_to_deliver_bond_duration: float,
    target_duration: Optional[float] = 0,
    how_many_bonds_per_futures: Optional[int] = 1000,
):
    # convert bond_futures_market_quote to futures_price
    futures_price = parse_quoted_price(bond_futures_market_quote)

    # calculate hedge ratio
    hedge_ratio = (
        (portfolio_duration - target_duration)
        * portfolio_value
        / (
            futures_price
            * how_many_bonds_per_futures
            * cheapest_to_deliver_bond_duration
        )
    )

    return hedge_ratio
