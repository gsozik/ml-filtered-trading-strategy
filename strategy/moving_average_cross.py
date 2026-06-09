import pandas as pd

from strategy.base import BaseStrategy, StrategyOrders


class MovingAverageCrossStrategy(BaseStrategy):
    """
    Классическая трендовая стратегия пересечения скользящих средних.

    Long:
        fast MA > slow MA

    Short:
        fast MA < slow MA

    Входы сделаны событийными:
        вход происходит только в момент смены состояния.
    """

    name = "ma_cross"

    def __init__(
        self,
        fast_col: str = "ema_50",
        slow_col: str = "ema_200",
        allow_short: bool = True,
    ):
        self.fast_col = fast_col
        self.slow_col = slow_col
        self.allow_short = allow_short

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        self._validate_columns(df, {self.fast_col, self.slow_col})

        fast = df[self.fast_col]
        slow = df[self.slow_col]

        long_state = fast > slow
        short_state = fast < slow

        prev_long_state = long_state.shift(1, fill_value=False)
        prev_short_state = short_state.shift(1, fill_value=False)

        entries = long_state & ~prev_long_state
        exits = prev_long_state & ~long_state

        if self.allow_short:
            short_entries = short_state & ~prev_short_state
            short_exits = prev_short_state & ~short_state
        else:
            short_entries = pd.Series(False, index=df.index)
            short_exits = pd.Series(False, index=df.index)

        return StrategyOrders(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
            short_entries=short_entries.fillna(False).astype(bool),
            short_exits=short_exits.fillna(False).astype(bool),
        )