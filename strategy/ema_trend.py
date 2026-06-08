import pandas as pd

from strategy.base import BaseStrategy, StrategyOrders


class EMATrendStrategy(BaseStrategy):
    name = "ema_trend"

    def __init__(
        self,
        fast_ema: str = "ema_8",
        slow_ema: str = "ema_200",
        rsi_col: str = "rsi_50",
        long_rsi_min: float = 50,
        short_rsi_max: float = 50,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_col = rsi_col
        self.long_rsi_min = long_rsi_min
        self.short_rsi_max = short_rsi_max

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        self._validate_columns(
            df,
            {
                self.fast_ema,
                self.slow_ema,
                self.rsi_col,
            },
        )

        long_state = (
            (df[self.fast_ema] > df[self.slow_ema])
            & (df[self.rsi_col] > self.long_rsi_min)
        )

        short_state = (
            (df[self.fast_ema] < df[self.slow_ema])
            & (df[self.rsi_col] < self.short_rsi_max)
        )

        prev_long_state = long_state.shift(1).fillna(False)
        prev_short_state = short_state.shift(1).fillna(False)

        entries = long_state & ~prev_long_state
        exits = prev_long_state & ~long_state

        short_entries = short_state & ~prev_short_state
        short_exits = prev_short_state & ~short_state

        return StrategyOrders(
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=short_exits.astype(bool),
        )