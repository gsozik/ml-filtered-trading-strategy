import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline
from ml import WalkForwardRegimeFilter
from strategy import BuyAndHoldStrategy, RobustTrendStrategy
from backtest import VectorBTBacktester


DATA_PATH = PROJECT_ROOT / "storage" / "SOL_USDT_4h_2024.csv"

TRAIN_WINDOW = 1500
TEST_WINDOW = 150
WINDOW_MODE = "expanding"
REGIME_LOOKBACK = 18

MODEL_NAMES = [
    "catboost",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    return df.sort_index()


def create_regime_filter(model_name: str) -> WalkForwardRegimeFilter:
    return WalkForwardRegimeFilter(
        model_name=model_name,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        regime_lookback=REGIME_LOOKBACK,
        window_mode=WINDOW_MODE,
        allow_short=True,

        use_optuna=True,
        optuna_trials=20,
        optuna_validation_size=0.25,
        optuna_metric="f1_macro",
    )


def build_result_row(name: str, result: dict, ml_metrics: dict | None = None) -> dict:
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
    cols = [
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

    table = result_table[cols].copy()
    numeric_cols = table.select_dtypes(include="number").columns
    table[numeric_cols] = table[numeric_cols].round(4)

    print("\nFINAL REGIME FILTER COMPARISON")
    print("=" * 100)
    print(table.to_string(index=False))


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

    print(f"\nML DEBUG: {model_name}")
    print("-" * (10 + len(model_name)))
    print("TA long entries:", int(orders.entries.sum()))
    print("TA short entries:", int(orders.short_entries.sum()))
    print("Allowed long entries:", int((orders.entries & (df_ml["ml_filter"] == 1)).sum()))
    print("Allowed short entries:", int((orders.short_entries & (df_ml["ml_filter"] == -1)).sum()))
    print("ml_filter distribution:")
    print(df_ml["ml_filter"].value_counts().sort_index().to_string())


def main():
    df = load_data(DATA_PATH)

    print("\nTIMEFRAME CHECK")
    print("=" * 100)
    print(df.index.to_series().diff().value_counts().head(10))

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

    reference_filter = create_regime_filter("catboost")
    df_reference = reference_filter.transform(df_features)

    test_df = df_reference[df_reference["is_ml_predicted"] == 1].copy()

    print("\nDATA INFO")
    print("=" * 100)
    print(f"Full rows: {len(df_features)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Full start: {df_features.index.min()}")
    print(f"Full end:   {df_features.index.max()}")
    print(f"Test start: {test_df.index.min()}")
    print(f"Test end:   {test_df.index.max()}")
    print(f"Train window: {TRAIN_WINDOW}")
    print(f"Test window: {TEST_WINDOW}")
    print(f"Regime lookback: {REGIME_LOOKBACK}")
    print(f"Window mode: {WINDOW_MODE}")

    print("\nWALK-FORWARD WINDOWS")
    print("=" * 100)
    windows = reference_filter.get_windows()
    print(windows.head().to_string(index=False))
    print("...")
    print(windows.tail().to_string(index=False))

    print_orders_debug(
        strategy=ta_strategy,
        df=df_features,
        title="RAW TA ORDERS ON FULL DATA",
    )

    print_orders_debug(
        strategy=ta_strategy,
        df=test_df,
        title="RAW TA ORDERS ON TEST DATA",
    )

    rows = []
    saved_results = {}

    bh_result = backtester.run(
        df=test_df,
        strategy=bh_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="regime_buy_and_hold",
    )

    rows.append(build_result_row("Buy&Hold", bh_result))
    saved_results["Buy&Hold"] = bh_result

    ta_result = backtester.run(
        df=test_df,
        strategy=ta_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="regime_ta_strategy",
    )

    rows.append(build_result_row("TA Strategy", ta_result))
    saved_results["TA Strategy"] = ta_result

    for model_name in MODEL_NAMES:
        print(f"\nRunning regime model: {model_name}")

        regime_filter = create_regime_filter(model_name)
        df_ml = regime_filter.transform(df_features)

        test_df_ml = df_ml[df_ml["is_ml_predicted"] == 1].copy()

        print("ML metrics:")
        print(regime_filter.get_metrics())

        print_ml_debug(
            strategy=ta_strategy,
            df_ml=test_df_ml,
            model_name=model_name,
        )

        ml_result = backtester.run(
            df=test_df_ml,
            strategy=ta_strategy,
            use_ml_filter=True,
            ml_filter_mode="block_opposite",
            save_plots=True,
            plot_name=f"regime_{ta_strategy.name}_{model_name}",
        )

        result_name = f"TA + {model_name} regime"

        rows.append(
            build_result_row(
                name=result_name,
                result=ml_result,
                ml_metrics=regime_filter.get_metrics(),
            )
        )

        saved_results[result_name] = ml_result

    result_table = pd.DataFrame(rows)

    print_result_table(result_table)

    output_dir = PROJECT_ROOT / "backtest_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_table.to_csv(output_dir / "summary_regime_filter_results.csv", index=False)
    reference_filter.get_windows().to_csv(output_dir / "regime_walk_forward_windows.csv", index=False)

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()