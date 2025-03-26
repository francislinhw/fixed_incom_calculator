from fixed_income_clculators.simulation_pricers import simulate_gbm_with_events


prob = simulate_gbm_with_events(verbose=False)

print(f"Probability of the event: {prob}")

# %% calculating probability that stock crosses
# %% $120 during the first year.

prob = simulate_gbm_with_events(
    events=[
        [0, 1, 120, "above"],
    ],
    verbose=False,
)

print(f"Probability of the event: {prob}")
