import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.ohlcv.bybit_loader import BybitOHLCVLoader
from ta import TechnicalAnalysisPipeline
from strategy import RobustTrendStrategy
from backtest import VectorBTBacktester


SYMBOL = "TON/USDT"
TIMEFRAME = "4h"
START = "2024-01-01"
END = "2024-12-31"

SAVE_PATH = PROJECT_ROOT / "storage" / "TON_USDT_4h_2024.csv"


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    df = df.sort_index()

    needed = ["open", "high", "low", "close", "volume"]
    missing = set(needed) - set(df.columns)

    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    return df[needed]


df = BybitOHLCVLoader(symbol=SYMBOL, timeframe=TIMEFRAME, start=START, end=END).load()
df = normalize_ohlcv(df)
df.to_csv(SAVE_PATH)