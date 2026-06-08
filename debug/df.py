import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.ohlcv.bybit_loader import BybitOHLCVLoader
from ta import TechnicalAnalysisPipeline
from strategy import RobustTrendStrategy
from backtest import VectorBTBacktester


SYMBOL = "ETH/USDT"
TIMEFRAME = "4h"
START = "2024-01-01"
END = "2024-12-31"

SAVE_PATH = PROJECT_ROOT / "storage" / "ETH_USDT_4h_2024.csv"


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


def main():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = BybitOHLCVLoader(symbol=SYMBOL, timeframe=TIMEFRAME, start=START, end=END).load()


    df = normalize_ohlcv(df)

    df.to_csv(SAVE_PATH)

    print("\nDATA SAVED")
    print("=" * 80)
    print(f"Path: {SAVE_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Start: {df.index.min()}")
    print(f"End:   {df.index.max()}")

    print("\nTIMEFRAME CHECK")
    print("=" * 80)
    print(df.index.to_series().diff().value_counts().head(10))

    print("\nHEAD")
    print(df.head())

    print("\nTAIL")
    print(df.tail())

    # TA
    df_features = TechnicalAnalysisPipeline().transform(df)

    print("\nFEATURES")
    print("=" * 80)
    print(f"Rows after TA dropna: {len(df_features)}")
    print(df_features[[
        "close",
        "ema_8",
        "ema_200",
        "rsi_50",
        "ema_dist",
        "trend_6",
        "trend_20",
        "trend_100",
    ]].tail())

    # Strategy orders
    strategy = RobustTrendStrategy()
    orders = strategy.generate_orders(df_features)

    print("\nRAW STRATEGY ORDERS")
    print("=" * 80)
    print("Long entries: ", int(orders.entries.sum()))
    print("Long exits:   ", int(orders.exits.sum()))
    print("Short entries:", int(orders.short_entries.sum()))
    print("Short exits:  ", int(orders.short_exits.sum()))

    # Backtest full period, without ML
    backtester = VectorBTBacktester(
        init_cash=10_000,
        fees=0.001,
        slippage=0.0008,
        freq=TIMEFRAME,
        result_dir="backtest_result",
    )

    result = backtester.run(
        df=df_features,
        strategy=strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="ETH_USDT_4h_2024_robust_no_ml",
    )

    print("\nBACKTEST RESULT WITHOUT ML")
    print("=" * 80)
    print(result["metrics"])

    print("\nTRADES")
    print("=" * 80)
    print(result["trades"].to_string(index=False))


if __name__ == "__main__":
    main()