import pandas as pd

from strategy.base import BaseStrategy, StrategyOrders


class MACDTrendStrategy(BaseStrategy):
    """
    Трендовая MACD-стратегия.

    Long:
        MACD пересекает signal снизу вверх.

    Short:
        MACD пересекает signal сверху вниз.
    """

    name = "macd_trend"

    def __init__(
        self,
        macd_col: str = "macd",
        signal_col: str = "macd_signal",
        allow_short: bool = True,
    ):
        self.macd_col = macd_col
        self.signal_col = signal_col
        self.allow_short = allow_short

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        self._validate_columns(df, {self.macd_col, self.signal_col})

        macd = df[self.macd_col]
        signal = df[self.signal_col]

        prev_macd = macd.shift(1)
        prev_signal = signal.shift(1)

        entries = (prev_macd <= prev_signal) & (macd > signal)
        exits = (prev_macd >= prev_signal) & (macd < signal)

        if self.allow_short:
            short_entries = (prev_macd >= prev_signal) & (macd < signal)
            short_exits = (prev_macd <= prev_signal) & (macd > signal)
        else:
            short_entries = pd.Series(False, index=df.index)
            short_exits = pd.Series(False, index=df.index)

        return StrategyOrders(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
            short_entries=short_entries.fillna(False).astype(bool),
            short_exits=short_exits.fillna(False).astype(bool),
        )