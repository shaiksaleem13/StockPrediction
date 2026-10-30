"""Financial Performance Metrics Engine for EAATS.

Calculates Total ROI %, Annualized Return %, Win Rate %, Annualized Sharpe Ratio,
Maximum Drawdown %, Trade Count, and Average Return per Trade.
All formulas strictly adhere to quantitative finance standards.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

import config


def calculate_sharpe_ratio(
    daily_returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = config.RISK_FREE_RATE,
    annualization_factor: int = config.ANNUALIZATION_FACTOR,
) -> float:
    """Calculates the annualized Sharpe ratio."""
    returns = np.array(daily_returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2 or np.all(returns == 0):
        return 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess_returns = returns - daily_rf

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess <= 1e-8:
        return 0.0

    sharpe = float((mean_excess / std_excess) * np.sqrt(annualization_factor))
    return round(sharpe, 3)


def calculate_max_drawdown(equity_curve: Union[np.ndarray, pd.Series]) -> Tuple[float, np.ndarray]:
    """Calculates the maximum peak-to-trough percentage drawdown."""
    equity = np.array(equity_curve)
    if len(equity) == 0:
        return 0.0, np.array([])

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.where(running_max == 0, 1e-6, running_max) * 100.0
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    return round(abs(max_dd), 2), drawdowns


def calculate_portfolio_metrics(
    equity_curve: Union[np.ndarray, pd.Series],
    daily_returns: Union[np.ndarray, pd.Series],
    closed_trades: List[Dict[str, Union[float, str]]],
    initial_capital: float = config.INITIAL_CAPITAL,
    trading_days: int = None,
) -> Dict[str, Union[float, int]]:
    """Generates comprehensive trading performance metrics summary."""
    equity = np.array(equity_curve)
    if len(equity) == 0:
        return {
            "Total_ROI_pct": 0.0,
            "Annualized_Return_pct": 0.0,
            "Sharpe_Ratio": 0.0,
            "Max_Drawdown_pct": 0.0,
            "Win_Rate_pct": 0.0,
            "Total_Trades": 0,
            "Profitable_Trades": 0,
            "Avg_Trade_Return_pct": 0.0,
            "Final_Equity": initial_capital,
        }

    final_equity = float(equity[-1])
    total_roi = ((final_equity - initial_capital) / initial_capital) * 100.0

    days = trading_days if trading_days is not None else len(equity)
    years = max(days / config.ANNUALIZATION_FACTOR, 1.0 / config.ANNUALIZATION_FACTOR)
    annualized_return = (((final_equity / initial_capital) ** (1.0 / years)) - 1.0) * 100.0

    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd, _ = calculate_max_drawdown(equity)

    # Trade Statistics
    total_trades = len(closed_trades)
    if total_trades > 0:
        trade_returns = [t.get("return_pct", 0.0) for t in closed_trades]
        profitable_trades = sum(1 for r in trade_returns if r > 0.0)
        win_rate = (profitable_trades / total_trades) * 100.0
        avg_trade_ret = float(np.mean(trade_returns))
    else:
        profitable_trades = 0
        win_rate = 0.0
        avg_trade_ret = 0.0

    return {
        "Total_ROI_pct": round(total_roi, 2),
        "Annualized_Return_pct": round(annualized_return, 2),
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown_pct": round(max_dd, 2),
        "Win_Rate_pct": round(win_rate, 2),
        "Total_Trades": total_trades,
        "Profitable_Trades": profitable_trades,
        "Avg_Trade_Return_pct": round(avg_trade_ret, 2),
        "Final_Equity": round(final_equity, 2),
    }
