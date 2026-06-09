import pandas as pd

from strategy.base import BaseStrategy, StrategyOrders


class RSIReversalStrategy(BaseStrategy):
    """
    Контртрендовая RSI-стратегия.

    Long:
        RSI выходит вверх из зоны перепроданности.

    Short:
        RSI выходит вниз из зоны перекупленности.

    Exit:
        позиция закрывается при возврате RSI к нейтральному уровню.
    """

    name = "rsi_reversal"

    def __init__(
        self,
        rsi_col: str = "rsi_14",
        oversold: float = 30,
        overbought: float = 70,
        exit_level: float = 50,
        allow_short: bool = True,
    ):
        self.rsi_col = rsi_col
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level
        self.allow_short = allow_short

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        self._validate_columns(df, {self.rsi_col})

        rsi = df[self.rsi_col]
        prev_rsi = rsi.shift(1)

        entries = (prev_rsi < self.oversold) & (rsi >= self.oversold)
        exits = rsi >= self.exit_level

        if self.allow_short:
            short_entries = (prev_rsi > self.overbought) & (rsi <= self.overbought)
            short_exits = rsi <= self.exit_level
        else:
            short_entries = pd.Series(False, index=df.index)
            short_exits = pd.Series(False, index=df.index)

        return StrategyOrders(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
            short_entries=short_entries.fillna(False).astype(bool),
            short_exits=short_exits.fillna(False).astype(bool),
        )