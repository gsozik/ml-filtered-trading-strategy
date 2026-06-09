import pandas as pd

from strategy.base import BaseStrategy, StrategyOrders


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Канальная breakout-стратегия.

    Long:
        close пробивает максимум за entry_window свечей.

    Short:
        close пробивает минимум за entry_window свечей.

    Exit:
        long закрывается при пробое минимума за exit_window;
        short закрывается при пробое максимума за exit_window.

    rolling high/low сдвинуты на 1 свечу,
    чтобы текущая свеча не участвовала в расчете уровня пробоя.
    """

    name = "donchian_breakout"

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        allow_short: bool = True,
    ):
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.allow_short = allow_short

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        self._validate_columns(df, {"high", "low", "close"})

        entry_high = df["high"].rolling(self.entry_window).max().shift(1)
        entry_low = df["low"].rolling(self.entry_window).min().shift(1)

        exit_low = df["low"].rolling(self.exit_window).min().shift(1)
        exit_high = df["high"].rolling(self.exit_window).max().shift(1)

        entries = df["close"] > entry_high
        exits = df["close"] < exit_low

        if self.allow_short:
            short_entries = df["close"] < entry_low
            short_exits = df["close"] > exit_high
        else:
            short_entries = pd.Series(False, index=df.index)
            short_exits = pd.Series(False, index=df.index)

        return StrategyOrders(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
            short_entries=short_entries.fillna(False).astype(bool),
            short_exits=short_exits.fillna(False).astype(bool),
        )