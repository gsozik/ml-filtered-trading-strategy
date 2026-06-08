import pandas as pd

from ta.indicators import (
    add_returns,
    add_sma,
    add_ema,
    add_rsi,
    add_macd,
    add_atr,
    add_volume_features,
    add_price_distance_features,
    add_trend_features,
)


class TechnicalAnalysisPipeline:
    def __init__(
        self,
        sma_windows: list[int] | None = None,
        ema_windows: list[int] | None = None,
        rsi_windows: list[int] | None = None,
        volume_windows: list[int] | None = None,
        atr_window: int = 14,
        dropna: bool = True,
    ):
        self.sma_windows = sma_windows or [8, 14, 50, 200]
        self.ema_windows = ema_windows or [8, 14, 50, 200]
        self.rsi_windows = rsi_windows or [14, 50]
        self.volume_windows = volume_windows or [50, 200]
        self.atr_window = atr_window
        self.dropna = dropna

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_ohlcv(df)

        df = df.copy()
        df = df.sort_index()

        df = add_returns(df)
        df = add_sma(df, self.sma_windows)
        df = add_ema(df, self.ema_windows)
        df = add_rsi(df, self.rsi_windows)
        df = add_macd(df)
        df = add_atr(df, self.atr_window)
        df = add_volume_features(df, self.volume_windows)
        df = add_price_distance_features(df)
        df = add_trend_features(df)

        if self.dropna:
            df = df.dropna()

        return df

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> None:
        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"Missing OHLCV columns: {missing_columns}")

        if df.empty:
            raise ValueError("Input DataFrame is empty")