from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import optimize
from scipy.optimize import fsolve, linprog, minimize
import pandas as pd
from ortools.linear_solver import pywraplp
from itertools import product
import pulp
import math


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


def find_the_optimal_weights_to_match_the_target_liability_by_different_cashflows(
    list_candidate_bonds: dict[str, list[float]],
    list_candidate_bonds_prices: dict[str, float],
    list_times: List[float],
    target_liability: list[float],
) -> tuple[dict[str, float], float]:
    """
    Example questions:
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

    """

    num_bonds = len(list_candidate_bonds)
    num_periods = len(list_times)

    # Construct cashflow matrix
    cashflow_matrix = np.zeros((num_periods, num_bonds))
    bond_prices = np.array(
        [list_candidate_bonds_prices[bond] for bond in list_candidate_bonds]
    )
    bond_names = list(list_candidate_bonds.keys())

    for j, bond in enumerate(bond_names):
        cashflows = list_candidate_bonds[bond]
        for i in range(len(cashflows)):
            if i < num_periods:
                cashflow_matrix[i, j] = cashflows[i]

    # Use inequality constraints (Ax >= b) instead of strict equality (Ax = b)
    A_ub = -cashflow_matrix  # Convert to Ax >= b format
    b_ub = -np.array(target_liability)  # Flip signs to match Ax >= b formulation

    # Introduce slack variables to allow small deviations
    slack_penalty = 1e3  # Adjust this if needed (higher = stricter matching)
    slack_matrix = np.eye(num_periods)  # Identity matrix for slack variables
    A_ub = np.hstack((A_ub, slack_matrix))  # Add slack variables to constraints
    c = np.concatenate(
        (bond_prices, np.ones(num_periods) * slack_penalty)
    )  # Penalize slack usage

    # Update bounds (bonds non-negative, slack variables unrestricted)
    bounds = [(0, None)] * num_bonds + [(0, None)] * num_periods

    # Solve using HiGHS with dual simplex for stability
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

    if result.success:
        optimal_bond_allocations = {
            bond_names[i]: result.x[i] for i in range(num_bonds)
        }
        total_cost = sum(
            optimal_bond_allocations[bond] * bond_prices[i]
            for i, bond in enumerate(bond_names)
        )
        return optimal_bond_allocations, total_cost
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def maximize_the_expected_return_of_a_portfolio_with_a_given_constraints(
    cash_flows: dict[str, list[float]],  # cash flows for each bond over years
    bond_prices: dict[str, float],  # prices for each bond
    total_budget: float,  # Total available investment
    max_investment_per_bond: float,  # Maximum amount to invest in any bond
):
    """
    Brute force approach to maximize expected return by exploring different investment allocations.
    """

    # Extract bond names and cash flow data
    bond_names = list(cash_flows.keys())
    cash_flow_matrix = np.array(
        list(cash_flows.values())
    )  # Investment returns over the years
    bond_prices = np.array([bond_prices[name] for name in bond_names])  # Bond prices

    # Initialize the best solution
    best_return = 0
    best_allocations = {}

    # Generate all possible combinations of investment amounts for each bond
    # We create a range from 0 to max_investment_per_bond with a step size of 1
    ranges = [range(0, int(max_investment_per_bond) + 1) for _ in bond_names]

    # Iterate over all possible combinations of investments
    for allocation in product(*ranges):
        # Calculate total investment
        total_investment = sum(
            allocation[i] * bond_prices[i] for i in range(len(bond_names))
        )

        # Skip allocations that exceed the total budget
        if total_investment > total_budget:
            continue

        # Calculate the total return at Year 3
        total_return = sum(
            allocation[i] * cash_flow_matrix[i, -1] for i in range(len(bond_names))
        )

        # If this combination is better, store the result
        if total_return > best_return:
            best_return = total_return
            best_allocations = {
                bond_names[i]: allocation[i] for i in range(len(bond_names))
            }

    return best_allocations, best_return


def auction_revenue_with_constraints(
    bids: list[tuple[list[int], float]],  # [selected bonds, price]
    max_bonds_selected: int,  # max # of bonds that can be selected
) -> tuple[float, list[str]]:
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
    # Example Bid Matrix B (7 bids, 4 bonds)
    # make list of candidate bonds from bids
    list_candidate_bonds = list(set(sum((bid[0] for bid in bids), [])))
    # Make B from bids
    B = np.zeros((len(bids), len(list_candidate_bonds)))
    for i, bid in enumerate(bids):
        for bond in bid[0]:
            B[i, bond] = 1

    # B = np.array(
    #     [
    #         [1, 0, 1, 0],  # Bid 1: Bonds 1 and 3 selected
    #         [0, 1, 0, 1],  # Bid 2: Bonds 2 and 4 selected
    #         [1, 1, 0, 1],  # Bid 3: Bonds 1, 2, and 4 selected
    #         [1, 0, 1, 0],  # Bid 4: Bonds 1 and 3 selected
    #         [0, 1, 1, 0],  # Bid 5: Bonds 2 and 3 selected
    #         [1, 1, 1, 0],  # Bid 6: Bonds 1, 2, and 3 selected
    #         [1, 1, 1, 0],  # Bid 7: Bonds 1, 2, and 3 selected
    #     ]
    # )

    # find prices from bids
    prices = [bid[1] for bid in bids]

    # Number of bids (n) and number of bonds (m)
    # Number of bids (n) and number of bonds (m)
    n, m = B.shape

    # Create the solver
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Solver not created.")
        return None

    # Decision variables: S[i] (binary, indicating whether bid i is selected)
    S = []
    for i in range(n):
        S.append(solver.BoolVar(f"S_{i}"))

    # Objective function: maximize sum of selected bids' prices
    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(S[i], prices[i])
    objective.SetMaximization()

    # Constraints: if a bid is selected, all bonds in that bid must be selected
    for j in range(m):
        constraint = solver.Constraint(
            0, max_bonds_selected
        )  # each bond can be selected at most once
        for i in range(n):
            if B[i][j] == 1:  # If the bond is selected in bid i
                constraint.SetCoefficient(S[i], 1)

    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal Solution:")
        selected_bids = []
        total_revenue = 0
        for i in range(n):
            if S[i].solution_value() > 0:
                selected_bids.append(f"Bid {i+1}")
                total_revenue += prices[i]

        return total_revenue, selected_bids

    else:
        print("No optimal solution found.")


def optimize_investments(
    cashflows: np.ndarray, budget: float, max_per_investments: list[float]
):
    """
    Optimize investments with per-investment maximum limits.

    Args:
        cashflows: 2D array where each row represents an investment's cash flow
        budget: Initial available cash
        max_per_investments: List of maximum allowed amounts for each investment

    Returns:
        (investment_decisions, final_cash)
    """
    n_investments, total_periods = cashflows.shape
    prob = pulp.LpProblem("Investment_Optimization", pulp.LpMaximize)

    # Validate input dimensions
    if len(max_per_investments) != n_investments:
        raise ValueError("max_per_investments must have one value per investment")

    # Decision variables with per-investment max limits
    x = []
    for i in range(n_investments):
        x_i = []
        for t in range(total_periods):
            if cashflows[i][t] < 0:  # Investment cost occurs at this period
                x_var = pulp.LpVariable(
                    f"x_{i}_{t}",
                    lowBound=0,
                    upBound=max_per_investments[i],  # Individual max here
                )
                x_i.append(x_var)
            else:
                x_i.append(None)
        x.append(x_i)

    # Cash balance variables
    cash = [pulp.LpVariable(f"cash_{t}", lowBound=0) for t in range(total_periods + 1)]

    # Objective: Maximize final cash
    prob += cash[total_periods]

    # Initial cash constraint
    initial_outflows = sum(
        x[i][0] * (-cashflows[i][0])
        for i in range(n_investments)
        if x[i][0] is not None
    )
    prob += cash[0] == budget - initial_outflows

    # Cash flow dynamics
    for t in range(1, total_periods + 1):
        inflows = 0
        for i in range(n_investments):
            for s in range(t):
                if s < total_periods and (t - s) < cashflows.shape[1]:
                    return_period = t - s
                    if x[i][s] is not None and cashflows[i][return_period] > 0:
                        inflows += x[i][s] * cashflows[i][return_period]

        outflows = 0
        if t < total_periods:
            outflows = sum(
                x[i][t] * (-cashflows[i][t])
                for i in range(n_investments)
                if x[i][t] is not None
            )

        prob += cash[t] == cash[t - 1] + inflows - outflows

    # Solve and extract results
    prob.solve()

    investment_decisions = []
    for i in range(n_investments):
        for t in range(total_periods):
            if x[i][t] is not None:
                val = x[i][t].varValue
                if val is not None and val > 1e-6:
                    investment_decisions.append((i, t, round(val, 2)))

    return investment_decisions, pulp.value(cash[total_periods])


def continuous_forward_rate(target_start: int, target_end: int, rates_map: dict):
    """
    Calculate the continuous forward rate between two time periods.

    Parameters:
    - target_start (int): The start period for the forward rate (e.g., 1 for year 1).
    - target_end (int): The end period for the forward rate (e.g., 2 for year 2).
    # - current (int): The current period (e.g., 0 for the present).
    - rates_map (dict): A dictionary with spot rates for different periods, where the key is the time period and the value is the rate.
    # rate(0, 1), ....

    Returns:
    - forward_rate (float): The calculated continuous forward rate between target_start and target_end.
    """
    # if time is 0, rate is 0
    rates_map[0] = 0

    # Spot rates at target_start and target_end
    R_start = rates_map.get(target_start)
    R_end = rates_map.get(target_end)

    if R_start is None or R_end is None:
        raise ValueError(f"Spot rates for {target_start} or {target_end} are missing.")

    # Use the formula for the forward rate with continuous compounding
    forward_rate = (R_end * target_end - R_start * target_start) / (
        target_end - target_start
    )

    return forward_rate


def simple_forward_rate(target_start: int, target_end: int, rates_map: dict):
    """
    Calculate the simple forward rate between two time periods.

    Parameters:
    - target_start (int): The start period for the forward rate (e.g., 1 for year 1).
    - target_end (int): The end period for the forward rate (e.g., 2 for year 2).
    - rates_map (dict): A dictionary with spot rates for different periods, where the key is the time period and the value is the rate.

    Returns:
    - forward_rate (float): The calculated simple forward rate between target_start and target_end.
    """
    # if time is 0, rate is 0
    rates_map[0] = 0

    # Spot rates at target_start and target_end
    R_start = rates_map.get(target_start)
    R_end = rates_map.get(target_end)

    if R_start is None or R_end is None:
        raise ValueError(f"Spot rates for {target_start} or {target_end} are missing.")

    # Use the formula for the simple forward rate
    forward_rate = ((1 + R_end) ** target_end / (1 + R_start) ** target_start - 1) / (
        target_end - target_start
    )

    return forward_rate


def continuous_discount_factor(target_start: int, target_end: int, rates_map: dict):
    """
    Calculate the continuous discount factor between two time periods.
    """
    forward_rate = continuous_forward_rate(target_start, target_end, rates_map)
    return np.exp(-forward_rate * (target_end - target_start))


def convert_continuous_to_simple(rates_map):
    """
    Converts continuous compounding rates to simple rates using the relationship:
    e^(r_t) = (1 + r_t).

    Parameters:
    - rates_map (dict): A dictionary with time periods as keys and continuous compounding rates as values.

    Returns:
    - simple_rates_map (dict): A dictionary with time periods as keys and simple rates as values.
    """
    simple_rates_map = {}

    for time, r_continuous in rates_map.items():
        # Convert continuous rate to simple rate using the formula
        r_simple = math.exp(r_continuous) - 1
        simple_rates_map[time] = r_simple

    return simple_rates_map


def calculate_bond_price_by_binomial_tree(
    interest_rate_tree, face_value=100, maturity=3, probability=0.5
):
    """
    Calculate the price of a zero-coupon bond using a binomial interest rate tree.

    Parameters:
    - interest_rate_tree (list of lists): The interest rate tree with rates for each node.
    - face_value (float): The face value of the bond (default is 100).
    - maturity (int): The maturity period of the bond in years (default is 3).
    - probability (float): The risk-neutral probability of interest rate increase (default is 50%).

    Returns:
    - bond_price (float): The price of the bond at time 0.
    """
    if maturity == 0:
        return face_value
    # Step 1: Initialize the bond price tree with face value at maturity
    bond_tree = []
    for i in range(maturity + 1):  # One extra layer to store the maturity values
        bond_tree.append([0] * (2**i))  # Create each layer of the tree

    # Set the bond value at maturity (all nodes in the last row are 100)
    for i in range(len(bond_tree[maturity])):
        bond_tree[maturity][i] = face_value

    # Step 2: Calculate the bond prices for previous layers
    for t in range(maturity - 1, -1, -1):  # Start from maturity-1 and go back to time 0
        for i in range(
            len(interest_rate_tree[t])
        ):  # Iterate through each node at this time
            r = interest_rate_tree[t][i]  # The interest rate at this node
            # Use the formula to calculate the bond price at this node
            # The bond price is the risk-neutral discounted expected value of the next step
            bond_tree[t][i] = (
                probability * bond_tree[t + 1][2 * i]  # 2 * i means left
                + (1 - probability)
                * bond_tree[t + 1][2 * i + 1]  # 2 * i + 1 means right
            ) * math.exp(-r)

    # Step 3: Return the price of the bond at time 0 (the value at the root of the tree)
    return bond_tree[0][0]


def forward_rate_from_binomial_tree(
    from_year, to_year, interest_rate_tree, probability: Optional[float] = 0.5
):
    """
    Calculate the forward rate from year i to year j using the binomial tree
    """
    if from_year == to_year:
        return 0
    else:
        # calculate the bond price for from_year
        bond_price_from_year = (
            calculate_bond_price_by_binomial_tree(
                interest_rate_tree, maturity=from_year, probability=probability
            )
            / 100
        )
        # calculate the bond price for to_year
        bond_price_to_year = (
            calculate_bond_price_by_binomial_tree(
                interest_rate_tree, maturity=to_year, probability=probability
            )
            / 100
        )
        # calculate the forward rate
        bond_price_from_year_zero_rate = -math.log(bond_price_from_year) / from_year
        bond_price_to_year_zero_rate = -math.log(bond_price_to_year) / to_year
        forward_rate = (
            bond_price_to_year_zero_rate * to_year
            - bond_price_from_year_zero_rate * from_year
        ) / (to_year - from_year)
        return forward_rate
