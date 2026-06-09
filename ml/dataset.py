import pandas as pd


class DirectionTargetBuilder:
    """
    Target для ML-фильтра направления.

    1  = future_return > long_threshold
    0  = движение слабое
    -1 = future_return < short_threshold
    """

    def __init__(
        self,
        horizon: int = 1,
        long_threshold: float = 0.003,
        short_threshold: float = -0.003,
    ):
        self.horizon = horizon
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def build(self, df: pd.DataFrame) -> pd.Series:
        future_return = df["close"].shift(-self.horizon) / df["close"] - 1

        target = pd.Series(index=df.index, dtype="float64", name="target")

        target[(future_return > self.long_threshold) & future_return.notna()] = 1
        target[(future_return < self.short_threshold) & future_return.notna()] = -1

        target[
            (future_return <= self.long_threshold)
            & (future_return >= self.short_threshold)
            & future_return.notna()
        ] = 0

        return target


def get_feature_columns(
    df: pd.DataFrame,
    use_raw_ohlcv: bool = False,
) -> list[str]:
    forbidden = {
        "target",
        "future_return",
        "ml_filter",
        "is_ml_predicted",
        "base_signal",
        "final_signal",
        "predicted_close",
        "predicted_return",
    }

    if not use_raw_ohlcv:
        forbidden.update({"open", "high", "low", "close", "volume"})

    return [
        col
        for col in df.columns
        if col not in forbidden
        and pd.api.types.is_numeric_dtype(df[col])
    ]