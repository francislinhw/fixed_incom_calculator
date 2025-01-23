from typing import Optional


class InstrumentInformation:
    def __init__(
        self,
        price: Optional[float] = None,
        yield_to_maturity: Optional[float] = None,
        duration: Optional[float] = None,
        modified_duration: Optional[float] = None,
    ):
        self.price = price
        self.yield_to_maturity = yield_to_maturity
        self.duration = duration
        self.modified_duration = modified_duration


class FixedIncomeInstrument:
    def __init__(self, cashflows, times, is_treasury):
        self.cashflows: list[list[float]] = cashflows
        self.times: list[float] = times
        self.is_treasury: bool = is_treasury

        self.instrument_information: InstrumentInformation = InstrumentInformation(
            price=None,
            yield_to_maturity=None,
            duration=None,
            modified_duration=None,
        )
