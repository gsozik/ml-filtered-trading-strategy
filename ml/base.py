import pandas as pd


class BaseMLFilter:
    name: str = "base_ml_filter"

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def predict_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает Series с индексом df.index.

        Значения:
            1  = long
            0  = stay
            -1 = short
        """
        raise NotImplementedError