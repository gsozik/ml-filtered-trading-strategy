from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from data.ohlcv.bybit_loader import BybitOHLCVLoader

from ta import TechnicalAnalysisPipeline
from backtest import VectorBTBacktester
from strategy import (
    RobustTrendStrategy,
    MovingAverageCrossStrategy,
    DonchianBreakoutStrategy,
    RSIReversalStrategy,
    MACDTrendStrategy,
)
from ml import DirectionMLFilter, LSTMDirectionFilter


START = "2024-01-01"
END = "2025-12-31"
TIMEFRAME = "4h"
TRAIN_SIZE = 1000
N_TRIALS = 100

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "TON/USDT",
    "KAVA/USDT",
    "COMP/USDT",
    "XLM/USDT",
    "SUI/USDT",
    "DOT/USDT",
    "ATOM/USDT",
    "FIL/USDT",
    "APT/USDT",
    "ARB/USDT"
]

RESULT_DIR = PROJECT_ROOT / "storage" / "final_research"
PNG_DIR = RESULT_DIR / "png"
MODEL_DIR = RESULT_DIR / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


strategies = [
    RobustTrendStrategy(),
    MovingAverageCrossStrategy(fast_col="ema_50", slow_col="ema_200"),
    DonchianBreakoutStrategy(entry_window=20, exit_window=10),
    RSIReversalStrategy(rsi_col="rsi_14"),
    MACDTrendStrategy(macd_col="macd", signal_col="macd_signal"),
]

classic_ml_models = [
    "logistic_regression",
    "random_forest",
    "catboost",
]

backtester = VectorBTBacktester(
    init_cash=10_000,
    fees=0.001,
    slippage=0.0008,
    freq=TIMEFRAME,
    shift_orders=True,
    logging=False,
)

all_results = []


for symbol in SYMBOLS:
    safe_symbol = symbol.replace("/", "_")
    print(f"\n===== {symbol} =====")

    try:
        raw_df = BybitOHLCVLoader(
            symbol=symbol,
            timeframe=TIMEFRAME,
            start=START,
            end=END,
            storage_path=str(PROJECT_ROOT / "storage"),
        ).load()

        df = TechnicalAnalysisPipeline().transform(raw_df)

        if len(df) <= TRAIN_SIZE + 100:
            print(f"Skip {symbol}: not enough rows")
            continue

        train_df = df.iloc[:TRAIN_SIZE].copy()
        test_df = df.iloc[TRAIN_SIZE:].copy()

        ml_filters = {}

        for model_name in classic_ml_models:
            print(f"Train ML: {symbol} | {model_name}")

            model = DirectionMLFilter(
                model_name=model_name,
                horizon=32,
                long_threshold=0.02,
                short_threshold=-0.02,
                use_raw_ohlcv=False,
                normalizer="auto",
                tune=True,
                n_trials=N_TRIALS,
                val_size=0.2,
                save_model=True,
                model_dir=str(MODEL_DIR),
                model_save_name=f"{safe_symbol}_{TIMEFRAME}_{model_name}",
                random_state=42,
                verbose=False,
            )

            model.fit(train_df)
            ml_filters[model_name] = model.predict_filter(test_df)

        print(f"Train ML: {symbol} | lstm")

        lstm = LSTMDirectionFilter(
            horizon=32,
            long_threshold=0.02,
            short_threshold=-0.02,
            normalizer="auto",
            tune=True,
            n_trials=N_TRIALS,
            val_size=0.2,
            save_weights=True,
            model_dir=str(MODEL_DIR),
            model_name=f"{safe_symbol}_{TIMEFRAME}_lstm",
            random_state=42,
            verbose=False,
        )

        lstm.fit(train_df)
        ml_filters["lstm"] = lstm.predict_filter(test_df)

        for strategy in strategies:
            print(f"Backtest: {symbol} | {strategy.name}")

            result = backtester.run_comparison(
                df=test_df,
                strategy=strategy,
                strategy_name=strategy.name,
                include_buy_and_hold=True,
                include_base_strategy=True,
                ml_filters=ml_filters,
                entry_mode="strict",
            )

            result.insert(0, "symbol", symbol)
            result.insert(1, "timeframe", TIMEFRAME)
            result.insert(2, "strategy_base", strategy.name)
            result.insert(3, "train_start", train_df.index.min())
            result.insert(4, "train_end", train_df.index.max())
            result.insert(5, "test_start", test_df.index.min())
            result.insert(6, "test_end", test_df.index.max())

            all_results.append(result)

            pd.concat(all_results, ignore_index=True).to_csv(
                RESULT_DIR / "checkpoint_backtest_results.csv",
                index=False,
            )

            for name in result["strategy"]:
                if name == "B&H":
                    continue

                if "&ML_" in name:
                    ml_name = name.split("&ML_")[-1]
                    bt = backtester.run_strategy(
                        df=test_df,
                        strategy=strategy,
                        name=name,
                        ml_filter=ml_filters[ml_name],
                        entry_mode="strict",
                    )
                else:
                    bt = backtester.run_strategy(
                        df=test_df,
                        strategy=strategy,
                        name=name,
                    )

                out = PNG_DIR / f"{safe_symbol}_{TIMEFRAME}_{name}.png"

                try:
                    bt["portfolio"].plot().write_image(
                        str(out),
                        width=1400,
                        height=1000,
                    )
                except Exception:
                    bt["portfolio"].plot().write_html(str(out.with_suffix(".html")))

    except Exception as e:
        print(f"ERROR {symbol}: {e}")


if not all_results:
    raise ValueError("No results collected. Check data loading and backtests.")

final_df = pd.concat(all_results, ignore_index=True)

for col in ["train_start", "train_end", "test_start", "test_end"]:
    final_df[col] = pd.to_datetime(final_df[col]).dt.tz_localize(None)

summary_df = (
    final_df
    .groupby("strategy", as_index=False)
    .agg(
        mean_equity_pct_change=("equity_pct_change", "mean"),
        mean_final_value=("final_value", "mean"),
        mean_sharpe=("sharpe", "mean"),
        mean_sortino=("sortino", "mean"),
        mean_max_drawdown_pct=("max_drawdown_%", "mean"),
        mean_win_rate_pct=("win_rate_%", "mean"),
        mean_trades=("trades", "mean"),
        symbols_count=("symbol", "nunique"),
    )
    .sort_values("mean_equity_pct_change", ascending=False)
)

bh_mean = summary_df.loc[
    summary_df["strategy"] == "B&H",
    "mean_equity_pct_change",
]

bh_mean = float(bh_mean.iloc[0]) if len(bh_mean) else 0.0

summary_df["diff_vs_BH_pct"] = (
    summary_df["mean_equity_pct_change"] - bh_mean
)

benchmark_df = summary_df[
    [
        "strategy",
        "mean_equity_pct_change",
        "diff_vs_BH_pct",
        "mean_sharpe",
        "mean_max_drawdown_pct",
        "mean_win_rate_pct",
        "mean_trades",
        "symbols_count",
    ]
].copy()

csv_path = RESULT_DIR / "final_backtest_results.csv"
excel_path = RESULT_DIR / "final_backtest_results.xlsx"

final_df.to_csv(csv_path, index=False)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    final_df.to_excel(writer, sheet_name="raw_results", index=False)
    summary_df.to_excel(writer, sheet_name="summary_by_strategy", index=False)
    benchmark_df.to_excel(writer, sheet_name="benchmark_comparison", index=False)

print("\nDONE")
print(f"CSV:   {csv_path}")
print(f"Excel: {excel_path}")
print(f"Charts: {PNG_DIR}")
print(f"Models: {MODEL_DIR}")