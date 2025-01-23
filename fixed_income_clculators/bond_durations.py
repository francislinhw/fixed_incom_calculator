from scipy.optimize import minimize
from fixed_income_clculators.fixed_income_instrument import (
    InstrumentInformation,
    FixedIncomeInstrument,
)


def calculate_bond_metrics(
    discount_rates, instrument: FixedIncomeInstrument, verbose: bool = True
):
    """
    Calculate the bond's YTM, price, Macaulay Duration, and Modified Duration.

    Parameters:
    - discount_rates: A list of initial yield rates for each period (used for discounting).
    - payments: A list of cash flow payments corresponding to each period.
    - times: A list of time periods corresponding to the cash flows.

    Returns:
    - ytm: The unified Yield to Maturity (YTM) for all periods.
    - price: The calculated bond price using the optimized YTM.
    - duration: The Macaulay Duration of the bond.
    - modified_duration: The Modified Duration of the bond.
    """
    payments, times = instrument.cashflows, instrument.times
    # Ensure inputs have matching lengths
    if len(discount_rates) != len(payments) or len(payments) != len(times):
        raise ValueError("Lengths of discount_rates, payments, and times must match.")

    # Objective function to find a unified YTM
    def ytm_objective(ytm_guess):
        ytm = ytm_guess[0]
        calculated_price = sum(cf / (1 + ytm) ** t for cf, t in zip(payments, times))
        initial_price = sum(
            cf / (1 + r) ** t for cf, r, t in zip(payments, discount_rates, times)
        )
        return abs(calculated_price - initial_price)

    # Initial guess for YTM (average of discount rates)
    initial_guess = [sum(discount_rates) / len(discount_rates)]

    # Optimize to find the unified YTM
    result = minimize(ytm_objective, initial_guess, bounds=[(0, 1)])
    ytm = result.x[0]

    # Calculate the bond price using the optimized YTM
    price = sum(cf / (1 + ytm) ** t for cf, t in zip(payments, times))

    # Calculate Macaulay Duration
    discounted_cash_flows = [
        (t * cf) / (1 + ytm) ** t for cf, t in zip(payments, times)
    ]
    duration = sum(discounted_cash_flows) / price

    # Calculate Modified Duration
    modified_duration = duration / (1 + ytm)

    instrument_information = InstrumentInformation(
        price=price,
        yield_to_maturity=ytm,
        duration=duration,
        modified_duration=modified_duration,
    )

    if verbose:
        print(f"YTM: {ytm:.4f}")
        print(f"Price: {price:.4f}")
        print(f"Macaulay Duration: {duration:.4f}")
        print(f"Modified Duration: {modified_duration:.4f}")

    return instrument_information
