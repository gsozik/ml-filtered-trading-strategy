import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline

from backtest import VectorBTBacktester
from strategy import RobustTrendStrategy
from strategy import MovingAverageCrossStrategy
from strategy import DonchianBreakoutStrategy
from strategy import RSIReversalStrategy
from strategy import MACDTrendStrategy

from ml import DirectionMLFilter, LSTMDirectionFilter


DATA_PATH = "storage/PEPE_USDT_4h_2024_2025.csv"


df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp").sort_index()

df = TechnicalAnalysisPipeline().transform(df)

strategy = RobustTrendStrategy()
#strategy = MovingAverageCrossStrategy(fast_col="ema_50", slow_col="ema_200")
#strategy = DonchianBreakoutStrategy(entry_window=20, exit_window=10)
#strategy = RSIReversalStrategy(rsi_col="rsi_14")
#strategy = MACDTrendStrategy(macd_col="macd", signal_col="macd_signal")

# инициализация бэктеста с параметрами
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

lstm_filter = LSTMDirectionFilter(
    model_name= 'lstm',
    horizon=32,
    long_threshold=0.02,
    short_threshold=-0.02,

    tune=True,
    n_trials=100,
    val_size=0.2,
    normalizer="auto",

    random_state=42,
    verbose=True,
)

lstm_filter.fit(train_df)
test_df["lstm_filter"] = lstm_filter.predict_filter(test_df)

# Запуск бекстеста
result = backtester.run_comparison(
    df=test_df,
    strategy=strategy,
    strategy_name=strategy.name,
    include_buy_and_hold=True,
    include_base_strategy=True,
    ml_filters={"lstm": test_df["lstm_filter"]},
    entry_mode="strict"
)

print(result.to_string(index=False))