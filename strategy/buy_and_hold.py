import pandas as pd

from strategy.base import BaseStrategy, StrategyOrders


class BuyAndHoldStrategy(BaseStrategy):
    name = "buy_and_hold"

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        short_entries = pd.Series(False, index=df.index)
        short_exits = pd.Series(False, index=df.index)

        if len(df) > 0:
            entries.iloc[0] = True
            exits.iloc[-1] = True

        return StrategyOrders(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )