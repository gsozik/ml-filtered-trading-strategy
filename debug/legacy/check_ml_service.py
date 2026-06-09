import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline
from strategy import BuyAndHoldStrategy, RobustTrendStrategy
from backtest import VectorBTBacktester
from ml import WalkForwardTradingOptimizedFilter


DATA_PATH = PROJECT_ROOT / "storage" / "LTC_USDT_4h_2024_2025.csv"

MODEL_NAME = "catboost"

TRAIN_WINDOW = 2000
TEST_WINDOW = 200
VALIDATION_SIZE = 0.25
N_TRIALS = 20

WINDOW_MODE = "expanding"
SCORE_METRIC = "return_minus_drawdown"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    return df.sort_index()


def make_row(name: str, result: dict, ml_metrics: dict | None = None) -> dict:
    m = result["metrics"]

    row = {
        "name": name,
        "final_value": m.get("final_value"),
        "pnl": m.get("total_pnl"),
        "return_%": m.get("total_return_pct"),
        "max_dd_%": m.get("max_drawdown_pct"),
        "sharpe": m.get("sharpe_ratio"),
        "sortino": m.get("sortino_ratio"),
        "calmar": m.get("calmar_ratio"),
        "win_rate_%": m.get("win_rate_pct"),
        "profit_factor": m.get("profit_factor"),
        "trades": m.get("total_trades"),
        "allowed_long": None,
        "allowed_short": None,
    }

    if ml_metrics is not None:
        row["allowed_long"] = ml_metrics.get("allowed_long")
        row["allowed_short"] = ml_metrics.get("allowed_short")

    return row


def print_table(rows: list[dict]) -> None:
    table = pd.DataFrame(rows)

    numeric_cols = table.select_dtypes(include="number").columns
    table[numeric_cols] = table[numeric_cols].round(4)

    print("\nFINAL COMPARISON")
    print("=" * 100)
    print(table.to_string(index=False))


def main():
    df = load_data(DATA_PATH)

    print("\nTIMEFRAME CHECK")
    print("=" * 100)
    print(df.index.to_series().diff().value_counts().head())

    df_features = TechnicalAnalysisPipeline().transform(df)

    ta_strategy = RobustTrendStrategy()
    bh_strategy = BuyAndHoldStrategy()

    backtester = VectorBTBacktester(
        init_cash=10_000,
        fees=0.001,
        slippage=0.0008,
        freq="4h",
        result_dir="backtest_result",
        shift_orders=True,
    )

    ml_filter = WalkForwardTradingOptimizedFilter(
        model_name=MODEL_NAME,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        validation_size=VALIDATION_SIZE,
        window_mode=WINDOW_MODE,
        n_trials=N_TRIALS,
        score_metric=SCORE_METRIC,
        verbose=True,
    )

    df_ml = ml_filter.transform(
        df=df_features,
        strategy=ta_strategy,
        backtester=backtester,
    )

    test_df = df_ml[df_ml["is_ml_predicted"] == 1].copy()

    print("\nDATA INFO")
    print("=" * 100)
    print(f"Full rows: {len(df_features)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Full start: {df_features.index.min()}")
    print(f"Full end:   {df_features.index.max()}")
    print(f"Test start: {test_df.index.min()}")
    print(f"Test end:   {test_df.index.max()}")

    print("\nML FILTER INFO")
    print("=" * 100)
    print(ml_filter.get_metrics())

    print("\nML FILTER DISTRIBUTION")
    print("=" * 100)
    print(test_df["ml_filter"].value_counts().sort_index().to_string())

    windows = ml_filter.get_windows()

    print("\nWALK-FORWARD WINDOWS")
    print("=" * 100)
    print(windows[[
        "window_id",
        "train_rows",
        "test_rows",
        "train_samples",
        "allowed_long",
        "allowed_short",
        "skipped",
        "best_score",
    ]].to_string(index=False))

    orders = ta_strategy.generate_orders(test_df)

    print("\nORDERS DEBUG")
    print("=" * 100)
    print("TA long entries:", int(orders.entries.sum()))
    print("TA short entries:", int(orders.short_entries.sum()))
    print("ML allowed long entries:", int((orders.entries & (test_df["ml_filter"] == 1)).sum()))
    print("ML allowed short entries:", int((orders.short_entries & (test_df["ml_filter"] == -1)).sum()))

    rows = []

    bh_result = backtester.run(
        df=test_df,
        strategy=bh_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="trading_opt_bh",
    )

    rows.append(make_row("Buy&Hold", bh_result))

    ta_result = backtester.run(
        df=test_df,
        strategy=ta_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="trading_opt_ta",
    )

    rows.append(make_row("TA Strategy", ta_result))

    ml_result = backtester.run(
        df=test_df,
        strategy=ta_strategy,
        use_ml_filter=True,
        ml_filter_mode="confirm",
        save_plots=True,
        plot_name=f"trading_opt_{MODEL_NAME}",
    )

    rows.append(
        make_row(
            name=f"TA + {MODEL_NAME} trading optimized",
            result=ml_result,
            ml_metrics=ml_filter.get_metrics(),
        )
    )

    print_table(rows)

    output_dir = PROJECT_ROOT / "backtest_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output_dir / "summary_trading_optimized_filter.csv",
        index=False,
    )

    windows.to_csv(
        output_dir / "trading_optimized_windows.csv",
        index=False,
    )

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()