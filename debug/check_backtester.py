import sys
from pathlib import Path

import pandas as pd

from ta import TechnicalAnalysisPipeline

from backtest import VectorBTBacktester
from strategy import RobustTrendStrategy
from strategy import MovingAverageCrossStrategy
from strategy import DonchianBreakoutStrategy
from strategy import RSIReversalStrategy
from strategy import MACDTrendStrategy

from ml import DirectionMLFilter


DATA_PATH = "storage/BTC_USDT_4h_2020_2022.csv"


df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp").sort_index()

df = TechnicalAnalysisPipeline().transform(df)

strategy = RobustTrendStrategy()
#strategy = MovingAverageCrossStrategy(fast_col="ema_50", slow_col="ema_200")
#strategy = DonchianBreakoutStrategy(entry_window=20, exit_window=10)
#strategy = RSIReversalStrategy(rsi_col="rsi_14")
#strategy = MACDTrendStrategy(macd_col="macd", signal_col="macd_signal")

backtester = VectorBTBacktester(
    init_cash=10_000,
    fees=0.001,
    slippage=0.0008,
    freq="4h",
    shift_orders=True,
    logging=True,
)

train_df = df.iloc[:1000].copy()
test_df = df.iloc[1000:].copy()

catboost_filter = DirectionMLFilter(
    model_name="catboost",
    horizon=1,
    long_threshold=0.003,
    short_threshold=-0.003,
    use_raw_ohlcv=False,
)
catboost_filter.fit(train_df)
test_df["catboost_filter"] = catboost_filter.predict_filter(test_df)

result = backtester.run_comparison(
    df=df,
    strategy=strategy,
    strategy_name=strategy.name,
    include_buy_and_hold=True,
    include_base_strategy=True,
    ml_filters={"catboost": test_df["catboost_filter"]},
    entry_mode="strict"
)

print(result.to_string(index=False))