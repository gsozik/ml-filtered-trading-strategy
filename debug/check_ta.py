import pandas as pd
import sys
from pathlib import Path

from ta import TechnicalAnalysisPipeline



df = pd.read_csv("storage/2024-01-01-2024-12-31.csv")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

pipeline = TechnicalAnalysisPipeline()
df_features = pipeline.transform(df)

print(df_features.head())
print(df_features.tail())
print(df_features.columns)
print(df_features.isna().sum().sum())

needed_cols = [
    "ema_dist",
    "ema_dist_bearish_3",
    "ema_dist_bullish_3",
    "trend_6",
    "trend_20",
    "trend_100",
]

print(print(df_features[df_features["trend_100"] == True]))
print(df_features[needed_cols].dtypes)