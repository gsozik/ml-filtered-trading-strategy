import pandas as pd
import numpy as np

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_sma(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
    return df


def add_ema(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for window in windows:
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        df[f"rsi_{window}"] = 100 - (100 / (1 + rs))

    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    df = df.copy()

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df[f"atr_{window}"] = true_range.rolling(window).mean()

    return df


def add_volume_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()

    for window in windows:
        df[f"volume_ema_{window}"] = df["volume"].ewm(span=window, adjust=False).mean()
        df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_ema_{window}"]

    return df


def add_price_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col.startswith("ema_") or col.startswith("sma_"):
            df[f"close_to_{col}"] = (df["close"] - df[col]) / df[col]

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ema_8" in df.columns and "ema_50" in df.columns:
        df["trend_fast"] = (df["ema_8"] > df["ema_50"]).astype(int)

    if "ema_50" in df.columns and "ema_200" in df.columns:
        df["trend_slow"] = (df["ema_50"] > df["ema_200"]).astype(int)

    if "ema_8" in df.columns and "ema_200" in df.columns:
        df["trend_global"] = (df["ema_8"] > df["ema_200"]).astype(int)

        df["ema_dist"] = (df["ema_8"] - df["ema_200"]) / df["ema_200"]

        df["ema_dist_bearish_3"] = (
            df["ema_dist"]
            .rolling(3)
            .apply(lambda x: x[0] > x[1] > x[2], raw=True)
            .fillna(0)
            .astype(bool)
        )

        df["ema_dist_bullish_3"] = (
            df["ema_dist"]
            .rolling(3)
            .apply(lambda x: x[0] < x[1] < x[2], raw=True)
            .fillna(0)
            .astype(bool)
        )

    df["trend_6"] = (
        (df["close"].rolling(6).max() - df["close"].rolling(6).min())
        / df["close"]
    ) > 0.05

    df["trend_20"] = (
        (df["close"].rolling(20).max() - df["close"].rolling(20).min())
        / df["close"]
    ) > 0.08

    df["trend_100"] = (
        (df["close"].rolling(100).max() - df["close"].rolling(100).min())
        / df["close"]
    ) > 0.12

    return df