from scipy.optimize import minimize
from fixed_income_clculators.fixed_income_instrument import FixedIncomeInstrument


def calculate_yield_curve_from_cashflows(treasury_bonds: list[FixedIncomeInstrument]):
    """
    Calculate the yield curve (spot rates) directly from cash flows and their timings.

    Parameters:
    - prices: A list of bond prices (one for each bond).
    - cashflows: A list of lists, where each sublist contains the cash flows of a bond.
    - times_list: A list of lists, where each sublist contains the times corresponding to the cash flows.

    Returns:
    - spot_rates: A dictionary with spot rates for each time point across all bonds.
    """
    prices, cashflows, times_list = [], [], []
    for bond in treasury_bonds:
        prices.append(bond.instrument_information.price)
        cashflows.append(bond.cashflows)
        times_list.append(bond.times)

    spot_rates = {}  # Store spot rates for specific times

    # Process each bond
    for price, cashflows, times in zip(prices, cashflows, times_list):
        # Solve for missing spot rates one by one
        for t, cf in zip(times, cashflows):
            if t in spot_rates:
                continue  # Skip if the rate for this time is already known

            # Objective function to solve for the spot rate at time t
            def spot_rate_objective(r_guess):
                r = r_guess[0]
                total_price = 0
                for i, cf_i in enumerate(cashflows):
                    time = times[i]
                    if time < t:
                        # Use known spot rates for earlier cash flows
                        total_price += cf_i / (1 + spot_rates[time]) ** time
                    elif time == t:
                        # Use the guessed rate for the current time
                        total_price += cf / (1 + r) ** t
                    else:
                        break
                return abs(total_price - price)

            # Optimize to find the spot rate for this time
            result = minimize(spot_rate_objective, [0.05], bounds=[(0, 1)])
            spot_rates[t] = result.x[0]

    # Interpolate for missing rates
    sorted_times = sorted(spot_rates.keys())
    for t in range(1, max(sorted_times) + 1):
        if t not in spot_rates:
            # Find the nearest known rates for interpolation
            lower_time = max([time for time in sorted_times if time < t], default=None)
            upper_time = min([time for time in sorted_times if time > t], default=None)
            if lower_time is None or upper_time is None:
                raise ValueError(f"Cannot interpolate spot rate for time {t}")

            lower_rate = spot_rates[lower_time]
            upper_rate = spot_rates[upper_time]
            spot_rates[t] = lower_rate + (upper_rate - lower_rate) * (
                t - lower_time
            ) / (upper_time - lower_time)

    return spot_rates
