import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline
from ml import WalkForwardMLFilter


df = pd.read_csv('storage/2024-01-01-2024-12-31.csv')

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

df = df.sort_index()

# 1. Считаем индикаторы
ta_pipeline = TechnicalAnalysisPipeline()
df_features = ta_pipeline.transform(df)

# 2. Строим честный walk-forward ML filter
ml_filter = WalkForwardMLFilter(
    train_window=1500,
    test_window=200,
    horizon=1,
    long_threshold=0.003,
    short_threshold=-0.003,
)

df_ml = ml_filter.transform(df_features)

print("ML metrics:")
print(ml_filter.get_metrics())

print("\nML filter distribution:")
print(df_ml["ml_filter"].value_counts().sort_index())

print("\nPredicted rows:")
print(df_ml["is_ml_predicted"].value_counts())

print("\nLast rows:")
print(df_ml[["close", "ml_filter", "is_ml_predicted"]].tail(20))

print("\nColumns with forbidden future info:")
for col in ["future_return", "target", "ml_target"]:
    print(col, col in df_ml.columns)