import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline
from ml import WalkForwardMLFilter
from strategy import BuyAndHoldStrategy, RobustTrendStrategy
from backtest import VectorBTBacktester


DATA_PATH = PROJECT_ROOT / "storage" / "ETH_USDT_4h_2024.csv"

TRAIN_WINDOW = 1000
TEST_WINDOW = 100
WINDOW_MODE = "expanding"

MODEL_NAMES = [
    "random_forest",
    "logistic_regression",
    "catboost",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    return df.sort_index()


def create_ml_filter(model_name: str) -> WalkForwardMLFilter:
    return WalkForwardMLFilter(
        model_name=model_name,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        horizon = 1,
        long_threshold=0.003,
        short_threshold=-0.003,
        window_mode=WINDOW_MODE,
    )


def print_orders_debug(strategy, df: pd.DataFrame, title: str) -> None:
    orders = strategy.generate_orders(df)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Long entries:  {int(orders.entries.sum())}")
    print(f"Long exits:    {int(orders.exits.sum())}")
    print(f"Short entries: {int(orders.short_entries.sum())}")
    print(f"Short exits:   {int(orders.short_exits.sum())}")


def build_result_row(
    name: str,
    backtest_result: dict,
    ml_metrics: dict | None = None,
) -> dict:
    m = backtest_result["metrics"]

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
        "closed_trades": m.get("closed_trades"),
        "open_trades": m.get("open_trades"),
        "ml_accuracy": None,
        "ml_f1_macro": None,
    }

    if ml_metrics is not None:
        row["ml_accuracy"] = ml_metrics.get("accuracy")
        row["ml_f1_macro"] = ml_metrics.get("f1_macro")

    return row


def print_result_table(result_table: pd.DataFrame) -> None:
    display_cols = [
        "name",
        "final_value",
        "pnl",
        "return_%",
        "max_dd_%",
        "sharpe",
        "sortino",
        "calmar",
        "win_rate_%",
        "profit_factor",
        "trades",
        "ml_accuracy",
        "ml_f1_macro",
    ]

    table = result_table[display_cols].copy()

    numeric_cols = table.select_dtypes(include="number").columns
    table[numeric_cols] = table[numeric_cols].round(4)

    print("\nFINAL BACKTEST COMPARISON")
    print("=" * 100)
    print(table.to_string(index=False))


def print_trade_tail(name: str, result: dict, n: int = 5) -> None:
    trades = result["trades"]

    print(f"\nLAST TRADES: {name}")
    print("-" * (13 + len(name)))

    if trades.empty:
        print("No trades")
        return

    cols = [
        "Entry Timestamp",
        "Exit Timestamp",
        "Avg Entry Price",
        "Avg Exit Price",
        "PnL",
        "Return",
        "Direction",
        "Status",
    ]

    existing_cols = [col for col in cols if col in trades.columns]
    print(trades[existing_cols].tail(n).to_string(index=False))


def main():
    df = load_data(DATA_PATH)

    df_features = TechnicalAnalysisPipeline().transform(df)

    ta_strategy = RobustTrendStrategy()
    bh_strategy = BuyAndHoldStrategy()

    backtester = VectorBTBacktester(
        init_cash=10_000,
        fees=0.001,
        slippage=0.0008,
        freq="4h",
        result_dir="backtest_result",
    )

    # Reference ML нужен только для определения общего out-of-sample участка.
    reference_filter = create_ml_filter("random_forest")
    df_reference = reference_filter.transform(df_features)

    test_df = df_reference[df_reference["is_ml_predicted"] == 1].copy()

    print("\nDATA INFO")
    print("=" * 100)
    print(f"Full data rows: {len(df_features)}")
    print(f"Test data rows: {len(test_df)}")
    print(f"Test start: {test_df.index.min()}")
    print(f"Test end:   {test_df.index.max()}")
    print(f"Train window: {TRAIN_WINDOW}")
    print(f"Test window:  {TEST_WINDOW}")
    print(f"Window mode:  {WINDOW_MODE}")

    print("\nWALK-FORWARD WINDOWS")
    print("=" * 100)
    windows = reference_filter.get_windows()
    print(windows.head().to_string(index=False))
    print("...")
    print(windows.tail().to_string(index=False))

    print_orders_debug(
        ta_strategy,
        df_features,
        "RAW TA STRATEGY ORDERS ON FULL DATA",
    )

    print_orders_debug(
        ta_strategy,
        test_df,
        "RAW TA STRATEGY ORDERS ON TEST DATA",
    )

    rows = []
    saved_results = {}

    # 1. Buy&Hold на том же out-of-sample участке
    bh_result = backtester.run(
        df=test_df,
        strategy=bh_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="buy_and_hold",
    )

    rows.append(
        build_result_row(
            name="Buy&Hold",
            backtest_result=bh_result,
        )
    )

    saved_results["Buy&Hold"] = bh_result

    # 2. Базовая TA-стратегия без ML
    base_result = backtester.run(
        df=test_df,
        strategy=ta_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name=f"{ta_strategy.name}_no_ml",
    )

    rows.append(
        build_result_row(
            name="TA Strategy",
            backtest_result=base_result,
        )
    )

    saved_results["TA Strategy"] = base_result

    # 3. TA-стратегия + ML-фильтр для каждой модели
    for model_name in MODEL_NAMES:
        print(f"\nRunning ML model: {model_name}")

        ml_filter = create_ml_filter(model_name)
        df_ml = ml_filter.transform(df_features)

        test_df_ml = df_ml[df_ml["is_ml_predicted"] == 1].copy()

        ml_result = backtester.run(
            df=test_df_ml,
            strategy=ta_strategy,
            use_ml_filter=True,
            ml_filter_mode="confirm",
            save_plots=True,
            plot_name=f"{ta_strategy.name}_{model_name}_confirm",
        )

        result_name = f"TA + {model_name}"

        rows.append(
            build_result_row(
                name=result_name,
                backtest_result=ml_result,
                ml_metrics=ml_filter.get_metrics(),
            )
        )

        saved_results[result_name] = ml_result

        print("ML metrics:")
        print(ml_filter.get_metrics())

        print("ML filter distribution on test:")
        print(
            test_df_ml["ml_filter"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    result_table = pd.DataFrame(rows)
    print_result_table(result_table)

    for name, result in saved_results.items():
        print_trade_tail(name, result, n=5)

    output_dir = PROJECT_ROOT / "backtest_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_backtest_results.csv"
    windows_path = output_dir / "walk_forward_windows.csv"

    result_table.to_csv(summary_path, index=False)
    reference_filter.get_windows().to_csv(windows_path, index=False)

    print(f"\nSaved summary table to: {summary_path}")
    print(f"Saved walk-forward windows to: {windows_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()