from typing import List, Optional
import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import pandas as pd


def PV_perpetuity(cashflow: float, discount_rate: float) -> float:
    return cashflow / discount_rate


def PV(
    cashflows: list[float], discount_rates: list[float], times: Optional[list[float]]
) -> float:
    if times is None:
        times = list(range(1, len(cashflows) + 1))
    return sum(
        cashflow / (1 + discount_rate) ** time
        for cashflow, time, discount_rate in zip(cashflows, times, discount_rates)
    )


def YTM(
    price: float,
    cashflows: List[float],
    times: List[float],
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> float:
    """
    Solve for Yield to Maturity (YTM) using Newton-Raphson method.

    :param price: Market price of the bond.
    :param cashflows: List of cash flows (coupons and face value).
    :param times: List of time periods for cash flows.
    :param tol: Convergence tolerance.
    :param max_iter: Maximum number of iterations.
    :return: Yield to Maturity (YTM).
    """
    # Initial guess: Approximate YTM
    guess = sum(cashflows) / price / len(cashflows)  # Simple heuristic

    for _ in range(max_iter):
        # Compute present value with current YTM guess
        pv = PV(cashflows, [guess] * len(times), times)

        # Compute derivative of PV (sensitivity to YTM)
        pv_derivative = -sum(
            t * cf / (1 + guess) ** (t + 1) for cf, t in zip(cashflows, times)
        )

        # Newton-Raphson update
        new_guess = guess - (pv - price) / pv_derivative

        # Convergence check
        if abs(new_guess - guess) < tol:
            return new_guess

        guess = new_guess  # Update guess

    raise ValueError("Newton-Raphson method did not converge")


def FV(
    cashflows: list[float],
    discount_rates: list[float],
    future_point_of_time: Optional[float] = None,
    times: Optional[list[float]] = None,
) -> float:
    if times is None:
        times = list(range(1, len(cashflows) + 1))
    if future_point_of_time is None:
        future_point_of_time = times[-1]

        return sum(
            cf * (1 + r) ** (future_point_of_time - t)
            for cf, r, t in zip(cashflows, discount_rates, times)
        )


def find_fixed_payment_in_times_to_fit_present_value(
    present_value: float,
    discount_rates: list[float],
    times: Optional[list[float]] = None,
) -> float:
    """
    present value = 105.1
    times = [1, 2, 3]
    discount_rates = [0.03, 0.033, 0.035]
    """
    if times is None:
        times = list(range(1, len(discount_rates) + 1))

    guess_cashflow = 10

    cashflows = [guess_cashflow] * len(times)

    price_by_guess = PV(cashflows, discount_rates, times)

    while True:
        if abs(price_by_guess - present_value) < 1e-6:
            return guess_cashflow
        guess_cashflow = (price_by_guess - present_value) / present_value
        cashflows = [guess_cashflow] * len(times)
        price_by_guess = PV(cashflows, discount_rates, times)


def find_fixed_payment_in_times_to_fit_future_value(
    future_value: float,
    discount_rates: list[float],
    times: Optional[list[float]] = None,
    future_point_of_time: Optional[float] = None,
) -> float:
    """
    future value = 105.1
    times = [1, 2, 3]
    discount_rates = [0.03, 0.033, 0.035]
    """
    if times is None:
        times = list(range(1, len(discount_rates) + 1))
    if future_point_of_time is None:
        future_point_of_time = times[-1]

    guess_cashflow = 10

    cashflows = [guess_cashflow] * len(times)

    future_value_by_guess = FV(cashflows, discount_rates, times)

    while True:
        if abs(future_value_by_guess - future_value) < 1e-6:
            return guess_cashflow
        guess_cashflow = (future_value_by_guess - future_value) / future_value
        cashflows = [guess_cashflow] * len(times)
        future_value_by_guess = FV(cashflows, discount_rates, times)


def find_the_fixed_payment_in_certain_times_to_fit_present_value(
    present_value: float,
    cashflows: List[float],
    discount_rates: List[float],
    fixed_payment_times: List[float],
    times: Optional[List[float]] = None,
) -> float:
    """
    Find the necessary fixed payments at given times to make the present value match.

    :param present_value: Target present value.
    :param cashflows: List of known cashflows (excluding the fixed payment).
    :param discount_rates: Corresponding discount rates.
    :param fixed_payment_times: Times where the fixed payments need to be solved.
    :param times: List of all times including known and fixed payment times.
    :return: Fixed payment amount (single value that should be paid at each fixed_payment_time).

    :raises ValueError: If inputs are invalid or inconsistent.
    """
    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [present_value] + cashflows + discount_rates + fixed_payment_times
    ):
        raise ValueError("All numerical inputs must be integers or floats")

    if any(r <= -1 for r in discount_rates):
        raise ValueError("Discount rates must be greater than -1")

    if len(cashflows) != len(discount_rates):
        raise ValueError("Length of cashflows must match length of discount_rates")

    if times is None:
        times = list(range(1, len(cashflows) + 1))

    # Validate times
    if len(times) != len(cashflows):
        raise ValueError("Length of times must match length of cashflows")
    if not all(t1 <= t2 for t1, t2 in zip(times[:-1], times[1:])):
        raise ValueError("Times must be in ascending order")

    # Identify indices of fixed payment times
    try:
        fixed_indices = [times.index(t) for t in fixed_payment_times]
    except ValueError:
        raise ValueError("All fixed_payment_times must exist in times list")

    def objective_function(payment: float) -> float:
        """Calculate difference between target PV and actual PV for a given payment."""
        # Create cashflow array with the fixed payment inserted at specified times
        test_cashflows = cashflows.copy()
        for idx in fixed_indices:
            test_cashflows[idx] = payment

        # Calculate present value
        computed_pv = PV(test_cashflows, discount_rates, times)

        # if (computed_pv - present_value) ** 2 < 1e-6:
        #     print("Converged")

        return (computed_pv - present_value) ** 2

    # Use scipy's optimization to find the payment that minimizes the difference
    result = optimize.minimize_scalar(
        objective_function, method="brent", options={"xtol": 1e-8}
    )

    if not result.success:
        raise ValueError("Optimization failed to converge")

    return float(result.x)


def find_the_fixed_payment_in_certain_times_to_fit_future_value(
    future_value: float,
    cashflows: List[float],
    discount_rates: List[float],
    fixed_payment_times: List[float],
    times: Optional[List[float]] = None,
    future_point_of_time: Optional[float] = None,
) -> float:
    """
    Find the necessary fixed payments at given times to make the future value (FV) match.

    :param future_value: Target future value.
    :param cashflows: List of known cashflows (excluding the fixed payment).
    :param discount_rates: Corresponding discount rates.
    :param fixed_payment_times: Times where the fixed payments need to be solved.
    :param times: List of all times including known and fixed payment times.
    :param future_point_of_time: Time at which the FV should match.
    :return: Fixed payment amount (single value that should be paid at each fixed_payment_time).

    Example:
    future_value = 120
    cashflows = [54, 56, 34]  # Some known cashflows
    discount_rates = [0.03, 0.033, 0.035]  # Discount rates
    fixed_payment_times = [2]  # Need to find CF at time 2
    times = [1, 2, 3]  # Time periods
    future_point_of_time = 4  # FV calculated at time 4
    """
    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [future_value] + cashflows + discount_rates + fixed_payment_times
    ):
        raise ValueError("All numerical inputs must be integers or floats")

    if any(r <= -1 for r in discount_rates):
        raise ValueError("Discount rates must be greater than -1")

    if len(cashflows) != len(discount_rates):
        raise ValueError("Length of cashflows must match length of discount_rates")

    if times is None:
        times = list(range(1, len(cashflows) + 1))

    if future_point_of_time is None:
        future_point_of_time = max(times)

    # Validate times
    if len(times) != len(cashflows):
        raise ValueError("Length of times must match length of cashflows")
    if not all(t1 <= t2 for t1, t2 in zip(times[:-1], times[1:])):
        raise ValueError("Times must be in ascending order")
    if future_point_of_time < max(times):
        raise ValueError(
            "Future point of time must be greater than or equal to the maximum time in the series"
        )

    # Identify indices of fixed payment times
    try:
        fixed_indices = [times.index(t) for t in fixed_payment_times]
    except ValueError:
        raise ValueError("All fixed_payment_times must exist in times list")

    def calculate_future_value(
        cashflows: List[float],
        discount_rates: List[float],
        future_point: float,
        times: List[float],
    ) -> float:
        """Calculate the future value of a series of cashflows."""
        return sum(
            cf * (1 + r) ** (future_point - t)
            for cf, r, t in zip(cashflows, discount_rates, times)
        )

    def objective_function(payment: float) -> float:
        """Calculate squared difference between target FV and actual FV for a given payment."""
        # Create cashflow array with the fixed payment inserted at specified times
        test_cashflows = cashflows.copy()
        for idx in fixed_indices:
            test_cashflows[idx] = payment

        # Calculate future value
        computed_fv = calculate_future_value(
            test_cashflows, discount_rates, future_point_of_time, times
        )

        if (computed_fv - future_value) ** 2 < 1e-6:
            print("Converged")

        return (computed_fv - future_value) ** 2

    # Use scipy's optimization to find the payment that minimizes the difference
    result = optimize.minimize_scalar(
        objective_function, method="brent", options={"xtol": 1e-8}
    )

    if not result.success:
        raise ValueError("Optimization failed to converge")

    return float(result.x)
