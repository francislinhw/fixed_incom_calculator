from fixed_income_clculators.fixed_income_instrument import (
    FixedIncomeInstrument,
    InstrumentInformation,
)
from fixed_income_clculators.bond_durations import calculate_bond_metrics


def calculate_duration_hedge_ratio(
    hedging_instrument: FixedIncomeInstrument,  # the instrument used to hedge
    hedged_instrument: FixedIncomeInstrument,  # the instrument being hedged
    discount_rates_hedging_instrument: list[float],
    discount_rates_hedged_instrument: list[float],
):
    """
    return hedging ratio of floating instrument to hedge against fixed income instrument
    """
    hedging_instrument_info: InstrumentInformation = calculate_bond_metrics(
        discount_rates_hedging_instrument,
        hedging_instrument,
        verbose=False,
    )
    hedged_instrument_info: InstrumentInformation = calculate_bond_metrics(
        discount_rates_hedged_instrument,
        hedged_instrument,
        verbose=False,
    )
    hedging_ratio = (
        hedged_instrument_info.modified_duration * hedged_instrument_info.price
    ) / (hedging_instrument_info.modified_duration * hedging_instrument_info.price)
    return hedging_ratio
