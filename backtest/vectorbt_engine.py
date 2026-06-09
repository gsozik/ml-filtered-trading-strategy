from pathlib import Path

import pandas as pd
import vectorbt as vbt

from backtest.metrics import extract_metrics
from strategy.base import BaseStrategy, StrategyOrders
from strategy.buy_and_hold import BuyAndHoldStrategy


class VectorBTBacktester:
    """
    Чистый backtester.

    ml_filter:
        1  = ML говорит long
        0  = ML говорит stay
        -1 = ML говорит short

    entry_mode:
        strict:
            TA long  + ML long  -> long
            TA short + ML short -> short

        soft:
            TA long  + ML long/stay  -> long
            TA short + ML short/stay -> short

    Выходы всегда остаются от обычной стратегии без ML.
    """

    def __init__(
        self,
        init_cash: float = 10_000,
        fees: float = 0.001,
        slippage: float = 0.0008,
        freq: str = "4h",
        shift_orders: bool = True,
        logging: bool = False,
        result_dir: str = "backtest_result",
    ):
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        self.shift_orders = shift_orders
        self.logging = logging

        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        name: str | None = None,
        ml_filter: pd.Series | None = None,
        entry_mode: str = "strict",
    ) -> dict:
        df = df.copy().sort_index()

        orders = strategy.generate_orders(df)

        if ml_filter is not None:
            orders = self._apply_ml_filter(
                orders=orders,
                ml_filter=ml_filter.reindex(df.index).fillna(0),
                entry_mode=entry_mode,
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

        strategy_name = name or strategy.name
        metrics = extract_metrics(portfolio)

        if self.logging:
            self._save_logs(
                portfolio=portfolio,
                df=df,
                orders=orders,
                name=strategy_name,
            )

        return {
            "name": strategy_name,
            "portfolio": portfolio,
            "metrics": metrics,
            "trades": portfolio.trades.records_readable,
            "orders": orders,
            "raw_orders": raw_orders,
            "df": df,
        }

    def run_comparison(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        strategy_name: str,
        include_buy_and_hold: bool = True,
        include_base_strategy: bool = True,
        ml_filters: dict[str, pd.Series] | None = None,
        entry_mode: str = "strict",
    ) -> pd.DataFrame:
        results = []

        if include_buy_and_hold:
            bh = BuyAndHoldStrategy()
            bh_result = self.run_strategy(
                df=df,
                strategy=bh,
                name="B&H",
                ml_filter=None,
            )
            results.append(self._make_row("B&H", bh_result))

        if include_base_strategy:
            base_result = self.run_strategy(
                df=df,
                strategy=strategy,
                name=strategy_name,
                ml_filter=None,
            )
            results.append(self._make_row(strategy_name, base_result))

        if ml_filters:
            for model_name, filter_series in ml_filters.items():
                ml_result = self.run_strategy(
                    df=df,
                    strategy=strategy,
                    name=f"{strategy_name}&ML_{model_name}",
                    ml_filter=filter_series,
                    entry_mode=entry_mode,
                )
                results.append(
                    self._make_row(f"{strategy_name}&ML_{model_name}", ml_result)
                )

        table = pd.DataFrame(results)

        numeric_cols = table.select_dtypes(include="number").columns
        table[numeric_cols] = table[numeric_cols].round(4)

        return table

    @staticmethod
    def _apply_ml_filter(
        orders: StrategyOrders,
        ml_filter: pd.Series,
        entry_mode: str,
    ) -> StrategyOrders:
        if entry_mode == "strict":
            entries = orders.entries & (ml_filter == 1)
            short_entries = orders.short_entries & (ml_filter == -1)

        elif entry_mode == "soft":
            entries = orders.entries & (ml_filter >= 0)
            short_entries = orders.short_entries & (ml_filter <= 0)

        else:
            raise ValueError("entry_mode must be 'strict' or 'soft'")

        return StrategyOrders(
            entries=entries.astype(bool),
            exits=orders.exits.astype(bool),
            short_entries=short_entries.astype(bool),
            short_exits=orders.short_exits.astype(bool),
        )

    @staticmethod
    def _shift_orders(orders: StrategyOrders) -> StrategyOrders:
        return StrategyOrders(
            entries=orders.entries.shift(1, fill_value=False).astype(bool),
            exits=orders.exits.shift(1, fill_value=False).astype(bool),
            short_entries=orders.short_entries.shift(1, fill_value=False).astype(bool),
            short_exits=orders.short_exits.shift(1, fill_value=False).astype(bool),
        )

    @staticmethod
    def _make_row(name: str, result: dict) -> dict:
        m = result["metrics"]

        return {
            "strategy": name,
            "init_cash": m["init_cash"],
            "final_value": m["final_value"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "max_drawdown_%": m["max_drawdown_%"],
            "equity_pct_change": m["equity_pct_change"],
            "win_rate_%": m["win_rate_%"],
            "trades": m["trades"],
        }

    def _save_logs(
        self,
        portfolio,
        df: pd.DataFrame,
        orders: StrategyOrders,
        name: str,
    ) -> None:
        safe_name = name.replace(" ", "_").replace("/", "_")

        portfolio.trades.records_readable.to_csv(
            self.result_dir / f"{safe_name}_trades.csv",
            index=False,
        )

        pd.Series(extract_metrics(portfolio)).to_csv(
            self.result_dir / f"{safe_name}_metrics.csv",
        )

        portfolio.value().to_csv(
            self.result_dir / f"{safe_name}_equity.csv",
        )

        portfolio.plot().write_html(
            self.result_dir / f"{safe_name}_portfolio.html"
        )

        price_fig = df["close"].vbt.plot(
            title=f"{name} | Price and Orders"
        )

        price_fig.add_scatter(
            x=df.index[orders.entries],
            y=df.loc[orders.entries, "close"],
            mode="markers",
            name="Long Entry",
        )

        price_fig.add_scatter(
            x=df.index[orders.exits],
            y=df.loc[orders.exits, "close"],
            mode="markers",
            name="Long Exit",
        )

        price_fig.add_scatter(
            x=df.index[orders.short_entries],
            y=df.loc[orders.short_entries, "close"],
            mode="markers",
            name="Short Entry",
        )

        price_fig.add_scatter(
            x=df.index[orders.short_exits],
            y=df.loc[orders.short_exits, "close"],
            mode="markers",
            name="Short Exit",
        )

        price_fig.write_html(
            self.result_dir / f"{safe_name}_orders.html"
        )