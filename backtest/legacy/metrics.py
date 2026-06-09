import pandas as pd


def extract_metrics(portfolio) -> dict:
    stats = portfolio.stats()
    trades = portfolio.trades.records_readable

    init_cash = float(portfolio.init_cash)
    final_value = float(portfolio.value().iloc[-1])
    total_pnl = final_value - init_cash
    total_return_pct = total_pnl / init_cash * 100

    closed_trades = trades[trades["Status"] == "Closed"] if not trades.empty else trades
    closed_pnl = float(closed_trades["PnL"].sum()) if not closed_trades.empty else 0.0

    open_trades = trades[trades["Status"] == "Open"] if not trades.empty else trades
    open_pnl = float(open_trades["PnL"].sum()) if not open_trades.empty else 0.0

    def get_stat(name: str, default=0.0):
        return stats[name] if name in stats.index else default

    return {
        "init_cash": init_cash,
        "final_value": final_value,
        "total_pnl": total_pnl,
        "closed_pnl": closed_pnl,
        "open_pnl": open_pnl,
        "total_return_pct": float(total_return_pct),
        "benchmark_return_pct": float(get_stat("Benchmark Return [%]", 0.0)),
        "max_drawdown_pct": float(get_stat("Max Drawdown [%]", 0.0)),
        "sharpe_ratio": float(get_stat("Sharpe Ratio", 0.0)),
        "sortino_ratio": float(get_stat("Sortino Ratio", 0.0)),
        "calmar_ratio": float(get_stat("Calmar Ratio", 0.0)),
        "win_rate_pct": float(get_stat("Win Rate [%]", 0.0)),
        "profit_factor": float(get_stat("Profit Factor", 0.0)),
        "total_trades": int(get_stat("Total Trades", 0)),
        "closed_trades": int(len(closed_trades)),
        "open_trades": int(len(open_trades)),
    }


def make_metrics_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)