import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline
from ml import WalkForwardTradeQualityFilter
from strategy import BuyAndHoldStrategy, RobustTrendStrategy
from backtest import VectorBTBacktester


DATA_PATH = PROJECT_ROOT / "storage" / "TON_USDT_4h_2022-2025.csv"

TRAIN_WINDOW = 1000
TEST_WINDOW = 100
WINDOW_MODE = "expanding"

PROFIT_HORIZON = 6
MIN_PROFIT = 0.003

MODEL_NAMES = [
    "random_forest",
    "logistic_regression",
    "catboost",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    return df.sort_index()


def create_ml_filter(model_name: str) -> WalkForwardTradeQualityFilter:
    return WalkForwardTradeQualityFilter(
        model_name=model_name,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        profit_horizon=PROFIT_HORIZON,
        min_profit=MIN_PROFIT,
        window_mode=WINDOW_MODE,
    )


def print_data_info(df_features: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\nDATA INFO")
    print("=" * 100)
    print(f"Full data rows: {len(df_features)}")
    print(f"Test data rows: {len(test_df)}")
    print(f"Full start: {df_features.index.min()}")
    print(f"Full end:   {df_features.index.max()}")
    print(f"Test start: {test_df.index.min()}")
    print(f"Test end:   {test_df.index.max()}")
    print(f"Train window: {TRAIN_WINDOW}")
    print(f"Test window:  {TEST_WINDOW}")
    print(f"Window mode:  {WINDOW_MODE}")
    print(f"Profit horizon: {PROFIT_HORIZON}")
    print(f"Min profit: {MIN_PROFIT}")


def print_windows(reference_filter: WalkForwardTradeQualityFilter) -> None:
    print("\nWALK-FORWARD WINDOWS")
    print("=" * 100)

    windows = reference_filter.get_windows()

    if windows.empty:
        print("No windows")
        return

    print(windows.head().to_string(index=False))
    print("...")
    print(windows.tail().to_string(index=False))


def print_orders_debug(strategy, df: pd.DataFrame, title: str) -> None:
    orders = strategy.generate_orders(df)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Long entries:  {int(orders.entries.sum())}")
    print(f"Long exits:    {int(orders.exits.sum())}")
    print(f"Short entries: {int(orders.short_entries.sum())}")
    print(f"Short exits:   {int(orders.short_exits.sum())}")


def print_ml_debug(strategy, df_ml: pd.DataFrame, model_name: str) -> None:
    orders = strategy.generate_orders(df_ml)

    long_before = int(orders.entries.sum())
    short_before = int(orders.short_entries.sum())

    long_allowed = int((orders.entries & (df_ml["ml_filter"] == 1)).sum())
    short_allowed = int((orders.short_entries & (df_ml["ml_filter"] == -1)).sum())

    print(f"\nML FILTER DEBUG: {model_name}")
    print("-" * (17 + len(model_name)))
    print(f"TA long entries before ML:   {long_before}")
    print(f"TA short entries before ML:  {short_before}")
    print(f"ML allowed long entries:     {long_allowed}")
    print(f"ML allowed short entries:    {short_allowed}")
    print("ML filter distribution:")
    print(df_ml["ml_filter"].value_counts().sort_index().to_string())


def build_result_row(
    name: str,
    backtest_result: dict,
    ml_metrics: dict | None = None,
) -> dict:
    metrics = backtest_result["metrics"]

    row = {
        "name": name,
        "final_value": metrics.get("final_value"),
        "pnl": metrics.get("total_pnl"),
        "return_%": metrics.get("total_return_pct"),
        "max_dd_%": metrics.get("max_drawdown_pct"),
        "sharpe": metrics.get("sharpe_ratio"),
        "sortino": metrics.get("sortino_ratio"),
        "calmar": metrics.get("calmar_ratio"),
        "win_rate_%": metrics.get("win_rate_pct"),
        "profit_factor": metrics.get("profit_factor"),
        "trades": metrics.get("total_trades"),
        "closed_trades": metrics.get("closed_trades"),
        "open_trades": metrics.get("open_trades"),
        "ml_accuracy": None,
        "ml_precision": None,
        "ml_recall": None,
        "ml_f1": None,
        "ml_train_samples": None,
    }

    if ml_metrics is not None:
        row["ml_accuracy"] = ml_metrics.get("accuracy")
        row["ml_precision"] = ml_metrics.get("precision")
        row["ml_recall"] = ml_metrics.get("recall")
        row["ml_f1"] = ml_metrics.get("f1")
        row["ml_train_samples"] = ml_metrics.get("predicted_trade_samples")

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
        "closed_trades",
        "open_trades",
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
        shift_orders=True,
    )

    # Reference-фильтр нужен только для получения общего out-of-sample периода.
    reference_filter = create_ml_filter("random_forest")
    df_reference = reference_filter.transform(
        df=df_features,
        strategy=ta_strategy,
    )

    test_df = df_reference[df_reference["is_ml_predicted"] == 1].copy()

    print_data_info(df_features=df_features, test_df=test_df)
    print_windows(reference_filter)

    print_orders_debug(
        strategy=ta_strategy,
        df=df_features,
        title="RAW TA STRATEGY ORDERS ON FULL DATA",
    )

    print_orders_debug(
        strategy=ta_strategy,
        df=test_df,
        title="RAW TA STRATEGY ORDERS ON TEST DATA",
    )

    rows = []
    saved_results = {}

    # 1. Buy&Hold на том же out-of-sample периоде.
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

    # 2. Базовая TA-стратегия без ML.
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

    # 3. TA-стратегия + ML trade-quality фильтр.
    for model_name in MODEL_NAMES:
        print(f"\nRunning ML trade-quality model: {model_name}")

        ml_filter = create_ml_filter(model_name)

        df_ml = ml_filter.transform(
            df=df_features,
            strategy=ta_strategy,
        )

        test_df_ml = df_ml[df_ml["is_ml_predicted"] == 1].copy()

        print("ML metrics:")
        print(ml_filter.get_metrics())

        print_ml_debug(
            strategy=ta_strategy,
            df_ml=test_df_ml,
            model_name=model_name,
        )

        # Для trade-quality фильтра используем confirm:
        # ml_filter = 1 разрешает long entry,
        # ml_filter = -1 разрешает short entry,
        # ml_filter = 0 запрещает вход.
        ml_result = backtester.run(
            df=test_df_ml,
            strategy=ta_strategy,
            use_ml_filter=True,
            ml_filter_mode="confirm",
            save_plots=True,
            plot_name=f"{ta_strategy.name}_{model_name}_trade_quality",
        )

        result_name = f"TA + {model_name} trade-quality"

        rows.append(
            build_result_row(
                name=result_name,
                backtest_result=ml_result,
                ml_metrics=ml_filter.get_metrics(),
            )
        )
        saved_results[result_name] = ml_result

    result_table = pd.DataFrame(rows)

    print_result_table(result_table)

    for name, result in saved_results.items():
        print_trade_tail(name=name, result=result, n=5)

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