import pandas as pd

from strategy.base import BaseStrategy, StrategyOrders


class RobustTrendStrategy(BaseStrategy):
    name = "robust_trend"

    def generate_orders(self, df: pd.DataFrame) -> StrategyOrders:
        self._validate_columns(
            df,
            {
                "close",
                "ema_8",
                "ema_200",
                "rsi_50",
                "ema_dist",
                "ema_dist_bearish_3",
                "ema_dist_bullish_3",
                "trend_6",
                "trend_20",
                "trend_100",
            },
        )

        entries = (
            (df["ema_8"] > df["ema_200"])
            & (df["rsi_50"] > 30)
            & (df["trend_100"])
            & (df["trend_20"])
        )

        exits = (
            (df["rsi_50"] > 80)
            | (df["close"] < df["ema_200"])
            | (df["ema_dist"] >= 0.08)
            | (df["ema_dist_bearish_3"])
            | (~df["trend_100"])
            | (~df["trend_6"])
        )

        short_entries = (
            (df["ema_8"] < df["ema_200"])
            & (df["rsi_50"] > 30)
            & (df["trend_100"])
            & (df["trend_20"])
        )

        short_exits = (
            (df["rsi_50"] > 80)
            | (df["close"] > df["ema_200"])
            | (df["ema_dist"] <= -0.08)
            | (df["ema_dist_bullish_3"])
            | (~df["trend_100"])
            | (~df["trend_6"])
        )

        return StrategyOrders(
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=short_exits.astype(bool),
        )