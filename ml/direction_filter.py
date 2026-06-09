import pandas as pd

from ml.base import BaseMLFilter
from ml.dataset import DirectionTargetBuilder, get_feature_columns
from ml.models import create_model


class DirectionMLFilter(BaseMLFilter):
    """
    Единый ML-фильтр направления.

    Модель обучается на train_df.
    Потом predict_filter(test_df) возвращает:
        1  = long
        0  = stay
        -1 = short
    """

    def __init__(
        self,
        model_name: str,
        horizon: int = 1,
        long_threshold: float = 0.003,
        short_threshold: float = -0.003,
        use_raw_ohlcv: bool = False,
        random_state: int = 42,
        params: dict | None = None,
    ):
        self.name = model_name
        self.model_name = model_name
        self.target_builder = DirectionTargetBuilder(
            horizon=horizon,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )
        self.use_raw_ohlcv = use_raw_ohlcv
        self.random_state = random_state
        self.params = params

        self.model = create_model(
            model_name=model_name,
            random_state=random_state,
            params=params,
        )

        self.feature_columns: list[str] = []

    def fit(self, df: pd.DataFrame):
        df = df.copy().sort_index()

        self.feature_columns = get_feature_columns(
            df,
            use_raw_ohlcv=self.use_raw_ohlcv,
        )

        y = self.target_builder.build(df)
        valid_mask = y.notna()

        X_train = df.loc[valid_mask, self.feature_columns]
        y_train = y.loc[valid_mask].astype(int)

        self.model.fit(X_train, y_train)

        return self

    def predict_filter(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy().sort_index()

        X = df[self.feature_columns]
        pred = self.model.predict(X)

        return pd.Series(
            pred.ravel(),
            index=df.index,
            name=self.name,
        ).astype(int)