from pathlib import Path

import pandas as pd
import vectorbt as vbt

from strategy.base import BaseStrategy, StrategyOrders
from backtest.legacy.metrics import extract_metrics


class VectorBTBacktester:
    def __init__(
        self,
        init_cash: float = 10_000,
        fees: float = 0.001,
        slippage: float = 0.0008,
        freq: str = "4h",
        result_dir: str = "backtest_result",
        shift_orders: bool = True,
    ):
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.shift_orders = shift_orders

        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        start_date: str | None = None,
        end_date: str | None = None,
        use_ml_filter: bool = False,
        ml_filter_mode: str = "confirm",
        save_plots: bool = False,
        plot_name: str | None = None,
    ) -> dict:
        df = df.copy().sort_index()

        if start_date is not None:
            df = df.loc[df.index >= pd.Timestamp(start_date)]

        if end_date is not None:
            df = df.loc[df.index <= pd.Timestamp(end_date)]

        self._validate_ohlcv(df)

        orders = strategy.generate_orders(df)

        if use_ml_filter:
            self._validate_ml_filter(df)
            orders = self._apply_ml_filter_to_orders(
                orders=orders,
                ml_filter=df["ml_filter"],
                mode=ml_filter_mode,
            )

        raw_orders = orders

        if self.shift_orders:
            orders = self._shift_orders(orders)

        portfolio = vbt.Portfolio.from_signals(
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            entries=orders.entries,
            exits=orders.exits,
            short_entries=orders.short_entries,
            short_exits=orders.short_exits,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
        )

        metrics = extract_metrics(portfolio)

        if plot_name is None:
            plot_name = strategy.name
            if use_ml_filter:
                plot_name += f"_ml_{ml_filter_mode}"

        if save_plots:
            self.save_plots(
                portfolio=portfolio,
                df=df,
                orders=orders,
                plot_name=plot_name,
            )

        return {
            "portfolio": portfolio,
            "metrics": metrics,
            "trades": portfolio.trades.records_readable,
            "df": df,
            "orders": orders,
            "raw_orders": raw_orders,
        }

    @staticmethod
    def _apply_ml_filter_to_orders(
        orders: StrategyOrders,
        ml_filter: pd.Series,
        mode: str = "confirm",
    ) -> StrategyOrders:
        if mode == "confirm":
            entries = orders.entries & (ml_filter == 1)
            short_entries = orders.short_entries & (ml_filter == -1)

        elif mode == "block_opposite":
            entries = orders.entries & (ml_filter != -1)
            short_entries = orders.short_entries & (ml_filter != 1)

        else:
            raise ValueError(
                "ml_filter_mode must be 'confirm' or 'block_opposite'"
            )

        return StrategyOrders(
            entries=entries.astype(bool),
            exits=orders.exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=orders.short_exits.astype(bool),
        )

    @staticmethod
    def _shift_orders(orders: StrategyOrders) -> StrategyOrders:
        return StrategyOrders(
            entries=orders.entries.astype(bool).shift(1, fill_value=False),
            exits=orders.exits.astype(bool).shift(1, fill_value=False),
            short_entries=orders.short_entries.astype(bool).shift(1, fill_value=False),
            short_exits=orders.short_exits.astype(bool).shift(1, fill_value=False),
        )

    def save_plots(
        self,
        portfolio,
        df: pd.DataFrame,
        orders: StrategyOrders,
        plot_name: str,
    ) -> None:
        safe_name = plot_name.replace(" ", "_").replace("/", "_")

        portfolio.plot().write_html(
            self.result_dir / f"{safe_name}_portfolio.html"
        )

        portfolio.value().vbt.plot(
            title=f"{plot_name} | Portfolio Value"
        ).write_html(
            self.result_dir / f"{safe_name}_equity.html"
        )

        portfolio.drawdown().vbt.plot(
            title=f"{plot_name} | Drawdown"
        ).write_html(
            self.result_dir / f"{safe_name}_drawdown.html"
        )

        price_fig = df["close"].vbt.plot(
            title=f"{plot_name} | Price and Orders"
        )

        price_fig.add_scatter(
            x=df.index[orders.entries],
            y=df.loc[orders.entries, "close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10),
            name="Long Entry",
        )

        price_fig.add_scatter(
            x=df.index[orders.exits],
            y=df.loc[orders.exits, "close"],
            mode="markers",
            marker=dict(symbol="x", size=9),
            name="Long Exit",
        )

        price_fig.add_scatter(
            x=df.index[orders.short_entries],
            y=df.loc[orders.short_entries, "close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10),
            name="Short Entry",
        )

        price_fig.add_scatter(
            x=df.index[orders.short_exits],
            y=df.loc[orders.short_exits, "close"],
            mode="markers",
            marker=dict(symbol="x", size=9),
            name="Short Exit",
        )

        price_fig.write_html(
            self.result_dir / f"{safe_name}_price_orders.html"
        )

        portfolio.trades.records_readable.to_csv(
            self.result_dir / f"{safe_name}_trades.csv",
            index=False,
        )

        pd.Series(extract_metrics(portfolio)).to_csv(
            self.result_dir / f"{safe_name}_metrics.csv"
        )

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Backtest DataFrame is empty after date filtering.")

        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing OHLC columns for backtest: {missing}")

    @staticmethod
    def _validate_ml_filter(df: pd.DataFrame) -> None:
        if "ml_filter" not in df.columns:
            raise ValueError("use_ml_filter=True, but column 'ml_filter' not found.")

        allowed = {-1, 0, 1}
        values = set(df["ml_filter"].dropna().unique())

        if not values.issubset(allowed):
            raise ValueError(f"ml_filter must contain only -1, 0, 1. Found: {values}")
        