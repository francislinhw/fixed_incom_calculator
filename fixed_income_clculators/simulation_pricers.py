import numpy as np
import matplotlib.pyplot as plt


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
            event = np.all(
                paths[eventstepsStart:eventstepsEnd, :] < events[i][2], axis=0
            )
        elif events[i][3] == "above":
            event = np.all(
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
