import numpy as np


def _safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default

    if np.isnan(value) or np.isinf(value):
        return default

    return value


def extract_metrics(portfolio) -> dict:
    stats = portfolio.stats()
    trades = portfolio.trades.records_readable

    init_cash = _safe_float(portfolio.init_cash)
    final_value = _safe_float(portfolio.value().iloc[-1])
    equity_pct_change = (final_value / init_cash - 1) * 100 if init_cash else 0.0

    def stat(name: str, default=0.0):
        return _safe_float(stats[name], default) if name in stats.index else default

    return {
        "init_cash": init_cash,
        "final_value": final_value,
        "sharpe": stat("Sharpe Ratio"),
        "sortino": stat("Sortino Ratio"),
        "max_drawdown_%": stat("Max Drawdown [%]"),
        "equity_pct_change": equity_pct_change,
        "win_rate_%": stat("Win Rate [%]"),
        "trades": int(stat("Total Trades")),
    }