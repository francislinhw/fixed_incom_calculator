from fixed_income_clculators.core import simple_forward_rate, continuous_forward_rate
from typing import Optional, Literal
import math


def calculate_fixed_swap_rate(
    zero_rates: dict,
    swap_maturity: int = 5,
    compounding_method: Optional[Literal["simple", "continuous"]] = "continuous",
    verbose: bool = False,
):
    """
    Calculate the fixed swap rate by matching the present value of the fixed leg
    to the present value of the floating leg.

    Parameters:
    - zero_rates (dict): A dictionary of zero-coupon rates for each year.
      Example: {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05}
    - compounding_method (str): The compounding method to use.
    - maturity (int): The maturity period for the swap (default is 5 years).
        Example: 5 (cannot exceed the length of zero_rates)

    Returns:
    - fixed_swap_rate (float): The fixed swap rate.
    """
    if swap_maturity > len(zero_rates):
        raise ValueError("Maturity cannot be greater than the length of zero_rates")

    # Step 1: Calculate the forward rates from the zero-coupon rates
    forward_rates = {}
    for i in range(0, swap_maturity):
        if compounding_method == "simple":
            forward_rates[f"{i}-{i+1}"] = simple_forward_rate(i, i + 1, zero_rates)
        elif compounding_method == "continuous":
            forward_rates[f"{i}-{i+1}"] = continuous_forward_rate(i, i + 1, zero_rates)
            if verbose:
                print(
                    f"Forward rate from {i} to {i+1}: {forward_rates[f'{i}-{i+1}']:.4f}"
                )
        else:
            raise ValueError("Invalid compounding method")

    # Step 2: Calculate the present value of the floating leg
    floating_leg_pv = 0
    for t in range(0, swap_maturity):
        if compounding_method == "simple":
            # Discount floating leg at simple rate
            floating_leg_pv += forward_rates[f"{t}-{t+1}"] / (
                1 + zero_rates[t + 1]
            ) ** (t + 1)
        elif compounding_method == "continuous":
            # Discount floating leg at continuous rate
            floating_leg_pv += forward_rates[f"{t}-{t+1}"] * math.exp(
                -zero_rates[t + 1] * (t + 1)
            )

    if verbose:
        print(f"Floating leg present value: {floating_leg_pv:.4f}")

    # Step 3: Calculate the present value of the fixed leg
    if compounding_method == "simple":
        fixed_rate = floating_leg_pv / sum(
            [1 / (1 + zero_rates[t]) ** t for t in range(1, swap_maturity + 1)]
        )

    elif compounding_method == "continuous":
        fixed_rate = floating_leg_pv / sum(
            [math.exp(-zero_rates[t] * t) for t in range(1, swap_maturity + 1)]
        )

    return fixed_rate
