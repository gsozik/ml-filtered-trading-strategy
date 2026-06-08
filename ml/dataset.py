import pandas as pd


class DirectionTargetBuilder:
    """
    Создает target для обучения модели.

    Target:
        1  = future return выше long_threshold
        0  = движение слабое / stay
        -1 = future return ниже short_threshold

    ВАЖНО:
    target создается только внутри ML-модуля.
    В backtest future_return и target не передаются.
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
        target[
            (future_return.notna())
            & (future_return > self.long_threshold)
        ] = 1
        target[
            (future_return.notna())
            & (future_return < self.short_threshold)
        ] = -1
        target[
            (future_return.notna())
            & (future_return <= self.long_threshold)
            & (future_return >= self.short_threshold)
        ] = 0
        return target


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Берем только текущие признаки.
    Будущих данных здесь быть не должно.
    """

    forbidden = {
        "target",
        "future_return",
        "ml_filter",
        "base_signal",
        "final_signal",
    }

    return [
        col for col in df.columns
        if col not in forbidden
        and pd.api.types.is_numeric_dtype(df[col])
    ]