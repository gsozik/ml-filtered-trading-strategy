import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ta import TechnicalAnalysisPipeline
from strategy import BuyAndHoldStrategy, RobustTrendStrategy
from backtest import VectorBTBacktester
from ml import WalkForwardLSTMPriceFilter


DATA_PATH = PROJECT_ROOT / "storage" / "BTC_USDT_4h_2020_2022.csv"

TRAIN_WINDOW = 1000
TEST_WINDOW = 50
WINDOW_MODE = "expanding"

SEQUENCE_LENGTH = 24
HORIZON = 1
RETURN_THRESHOLD = 0.0

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

ALLOW_LONG = True
ALLOW_SHORT = True


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    return df.sort_index()


def make_row(name: str, result: dict) -> dict:
    m = result["metrics"]

    return {
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
    }


def print_table(rows: list[dict]) -> None:
    table = pd.DataFrame(rows)

    numeric_cols = table.select_dtypes(include="number").columns
    table[numeric_cols] = table[numeric_cols].round(4)

    print("\nFINAL LSTM PRICE FILTER COMPARISON")
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

    lstm_filter = WalkForwardLSTMPriceFilter(
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        return_threshold=RETURN_THRESHOLD,
        window_mode=WINDOW_MODE,
        allow_long=ALLOW_LONG,
        allow_short=ALLOW_SHORT,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        random_state=42,
        verbose=True,
    )

    df_ml = lstm_filter.transform(df_features)

    test_df = df_ml[df_ml["is_ml_predicted"] == 1].copy()

    print("\nDATA INFO")
    print("=" * 100)
    print(f"Full rows: {len(df_features)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Full start: {df_features.index.min()}")
    print(f"Full end:   {df_features.index.max()}")
    print(f"Test start: {test_df.index.min()}")
    print(f"Test end:   {test_df.index.max()}")

    print("\nLSTM PRICE FILTER INFO")
    print("=" * 100)
    print(lstm_filter.get_metrics())

    print("\nLSTM PRICE FILTER DISTRIBUTION")
    print("=" * 100)
    print(test_df["ml_filter"].value_counts().sort_index().to_string())

    print("\nPREDICTED RETURN DESCRIPTION")
    print("=" * 100)
    print(test_df["predicted_return"].describe().to_string())

    windows = lstm_filter.get_windows()

    print("\nWALK-FORWARD WINDOWS")
    print("=" * 100)
    print(
        windows[
            [
                "window_id",
                "train_rows",
                "test_rows",
                "train_samples",
                "long_predictions",
                "short_predictions",
                "skipped",
            ]
        ].to_string(index=False)
    )

    orders = ta_strategy.generate_orders(test_df)

    print("\nORDERS DEBUG")
    print("=" * 100)
    print("TA long entries:", int(orders.entries.sum()))
    print("TA short entries:", int(orders.short_entries.sum()))
    print(
        "LSTM confirmed long entries:",
        int((orders.entries & (test_df["ml_filter"] == 1)).sum()),
    )
    print(
        "LSTM confirmed short entries:",
        int((orders.short_entries & (test_df["ml_filter"] == -1)).sum()),
    )

    rows = []

    bh_result = backtester.run(
        df=test_df,
        strategy=bh_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="lstm_price_bh",
    )
    rows.append(make_row("Buy&Hold", bh_result))

    ta_result = backtester.run(
        df=test_df,
        strategy=ta_strategy,
        use_ml_filter=False,
        save_plots=True,
        plot_name="lstm_price_ta",
    )
    rows.append(make_row("TA Strategy", ta_result))

    lstm_result = backtester.run(
        df=test_df,
        strategy=ta_strategy,
        use_ml_filter=True,
        ml_filter_mode="confirm",
        save_plots=True,
        plot_name="lstm_price_ta_filter",
    )
    rows.append(make_row("TA + LSTM price filter", lstm_result))

    print_table(rows)

    output_dir = PROJECT_ROOT / "backtest_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output_dir / "summary_lstm_price_filter.csv",
        index=False,
    )

    windows.to_csv(
        output_dir / "lstm_price_filter_windows.csv",
        index=False,
    )

    test_df[
        [
            "close",
            "predicted_close",
            "predicted_return",
            "ml_filter",
        ]
    ].to_csv(
        output_dir / "lstm_price_predictions.csv",
        index=True,
    )

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()