from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyOrders:
    entries: pd.Series
    exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series


class BaseStrategy:
    name: str = "base_strategy"

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        raise NotImplementedError

    @staticmethod
    def _empty_orders(df: pd.DataFrame) -> StrategyOrders:
        empty = pd.Series(False, index=df.index)

        return StrategyOrders(
            entries=empty.copy(),
            exits=empty.copy(),
            short_entries=empty.copy(),
            short_exits=empty.copy(),
        )

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")