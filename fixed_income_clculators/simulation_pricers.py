import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)


# Function to simulate GBM with drift equal to risk-free rate
def simulate_gbm(rate, sigma, stock, T, steps, nTrials):
    dt = T / steps
    # Simulate the GBM paths
    dW = np.random.normal(0, 1, (steps, nTrials)) * np.sqrt(dt)  # Brownian increments
    W = np.cumsum(dW, axis=0)  # Cumulative sum to get the Wiener process
    t = np.linspace(0, T, steps)
    S = stock * np.exp((rate - 0.5 * sigma**2) * t[:, None] + sigma * W)
    return S


def simulate_gbm_with_events(
    rate: float = 0.03,
    sigma: float = 0.1,
    stock: float = 100,
    T: float = 2,
    steps: int = 360 * 2,
    nTrials: int = 20000,
    events: list = [
        [0, 0.25, 110, "below"],  # [start, end, value, condition]
        [0.25, 0.5, 105, "above"],
    ],
    verbose: bool = False,
):

    # Simulate paths
    nTrials = 20000
    paths = simulate_gbm(rate, sigma, stock, T, steps, nTrials)

    if verbose:
        # Plot paths
        plt.plot(paths)
        plt.title("Simulated GBM Paths")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.show()

    pathInOneYear = steps / T
    # Probability calculations
    eventlists = []
    # Event 1: Stock is below $110 in the first 3 months (first 91 steps)
    for i in range(len(events)):
        eventstepsStart = int(pathInOneYear * events[i][0])
        eventstepsEnd = int(pathInOneYear * events[i][1]) + 1
        if events[i][3] == "below":
            event = np.any(
                paths[eventstepsStart:eventstepsEnd, :] < events[i][2], axis=0
            )
        elif events[i][3] == "above":
            event = np.any(
                paths[eventstepsStart:eventstepsEnd, :] > events[i][2], axis=0
            )
        eventlists.append(event)

    # Combined event (both conditions must be true)
    combined_event = np.all(eventlists, axis=0)

    # Calculate probability
    prob_full_event = np.count_nonzero(combined_event) / nTrials
    if verbose:
        print(f"Probability of the event: {prob_full_event}")
    return prob_full_event


def simulate_gbm_barrier_option_pricer(
    rate: float = 0.03,
    sigma: float = 0.1,
    stock: float = 100,
    T: float = 2,
    steps: int = 360 * 2,
    nTrials: int = 20000,
    barrier: float = 120,
    barrier_type: str = "up_in",  # "up_in", "up_out", "down_in", "down_out"
    barrier_start_time: float = 0,
    barrier_end_time: float = 1,
    option_type: str = "call",  # "call" or "put"
    strike: float = 100,
    seed: int = None,
    verbose: bool = False,
):
    """
    Simulates a barrier option (knock-in or knock-out) under GBM dynamics.

    barrier_type can be:
      - "up_in":   Option *activates* if path goes above barrier
      - "up_out":  Option becomes worthless if path goes above barrier
      - "down_in": Option *activates* if path goes below barrier
      - "down_out":Option becomes worthless if path goes below barrier

    barrier_start_time, barrier_end_time : fraction of total time T
        e.g., 0 to 1 means the first year if T=2
    """
    # 1) Simulate underlying paths
    paths = simulate_gbm(
        rate=rate,
        sigma=sigma,
        stock=stock,
        T=T,
        steps=steps,
        nTrials=nTrials,
    )  # shape: (steps+1, nTrials)

    # 2) Identify barrier inspection steps
    #    e.g., if barrier_end_time=1, that means 1 year out of T=2 -> steps/2
    barrier_start_idx = int(barrier_start_time / T * steps)
    barrier_end_idx = int(barrier_end_time / T * steps) + 1

    # 3) Check if the barrier was touched/hit in [barrier_start_idx : barrier_end_idx]
    barrier_region = paths[
        barrier_start_idx:barrier_end_idx, :
    ]  # shape: (some_steps, nTrials)

    if "up" in barrier_type:
        # "Touched" means any step above barrier
        touched = np.any(barrier_region > barrier, axis=0)  # shape: (nTrials,)
    else:  # "down" in barrier_type
        # "Touched" means any step below barrier
        touched = np.any(barrier_region < barrier, axis=0)

    # 4) Compute payoff at maturity (last step => index = steps)
    final_prices = paths[steps - 1, :]
    if option_type == "call":
        payoffs = np.maximum(final_prices - strike, 0.0)
    else:  # put
        payoffs = np.maximum(strike - final_prices, 0.0)

    # 5) Apply barrier logic
    #    - For "knock-in":  payoff = payoff only if touched, else 0
    #    - For "knock-out": payoff = payoff only if NOT touched, else 0
    if barrier_type.endswith("_in"):
        # Only pay if the barrier was touched
        effective_payoffs = payoffs * touched
    else:
        # Only pay if the barrier was NOT touched
        effective_payoffs = payoffs * (~touched)

    # 6) Discount and compute mean
    discounted_payoffs = np.exp(-rate * T) * effective_payoffs
    price_estimate = np.mean(discounted_payoffs)

    if verbose:
        print(f"Barrier Option: {barrier_type} {option_type} with barrier={barrier}")
        print(f"Simulation: {nTrials} trials, steps={steps}, T={T} years")
        print(f"Mean discounted payoff: {price_estimate:.4f}")

        # Optional: Plot some paths
        sample_paths = min(20, nTrials)  # just plot a few
        time_grid = np.linspace(0, T, steps)
        for i in range(sample_paths):
            plt.plot(time_grid, paths[:, i])
        plt.axhline(y=barrier, linestyle="--")
        plt.title("Simulated GBM Paths (Barrier Shown)")
        plt.xlabel("Time (Years)")
        plt.ylabel("Underlying Price")
        plt.show()

    return price_estimate


def price_generic_payoff(
    rate: float = 0.03,
    sigma: float = 0.1,
    stock: float = 100,
    T: float = 2,
    steps: int = 360 * 2,
    nTrials: int = 20000,
    payoff_func: callable = None,
):
    """
    Parameters:
    -----------
    paths       : ndarray of shape (steps+1, nTrials)
        Simulated paths of the underlying. Each column is a single path.
    r           : float
        Risk-free rate (annualized).
    T           : float
        Total time in years for these paths.
    payoff_func : callable
        A function that takes a single path (1D ndarray) and returns a scalar payoff.

    Returns:
    --------
    price       : float
        The Monte Carlo estimate of E[e^{-rT} * payoff(path)].
    """
    paths = simulate_gbm(rate, sigma, stock, T, steps, nTrials)
    nTrials = paths.shape[1]
    payoffs = np.zeros(nTrials)

    # Evaluate the payoff function on each path
    for j in range(nTrials):
        single_path = paths[:, j]  # shape: (steps+1,)
        payoffs[j] = payoff_func(single_path)

    # Discount and take the average
    discounted_payoffs = np.exp(-rate * T) * payoffs
    price = np.mean(discounted_payoffs)

    return price
