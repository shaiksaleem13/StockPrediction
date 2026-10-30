"""Event-Driven Walk-Forward Backtester for EAATS.

Executes trades at Next-Day Open price to strictly eliminate look-ahead bias.
Simulates 5 strategies:
1. Buy & Hold Benchmark
2. Technical-Only (SMA/EMA Crossover + RSI)
3. EAATS Pure (No Emotion / Fixed 50% Position Size)
4. EAATS Full (Emotion-Aware + Adaptive Risk Engine)
5. Reinforcement Learning Q-Agent

Includes Historical Crisis Stress Testing (COVID-19 2020 & 2022 Bear Market)
and trade ledger export.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import config
from src.adaptive_risk import determine_adaptive_risk_profile
from src.metrics import calculate_max_drawdown, calculate_portfolio_metrics
from src.rl_agent import TabularQTradingAgent, discretize_state
from src.trading_strategy import evaluate_trading_decision
from src.utils import logger


class BacktestEngine:
    """Simulates trading strategies chronologically day by day."""

    def __init__(
        self,
        df_test: pd.DataFrame,
        predicted_prices: np.ndarray,
        sentiment_series: pd.Series,
        initial_capital: float = config.INITIAL_CAPITAL,
        transaction_fee: float = config.TRANSACTION_FEE,
        confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.df = df_test.copy()
        self.predicted_prices = np.array(predicted_prices)
        self.sentiment_series = pd.Series(sentiment_series, index=self.df.index)
        self.initial_capital = initial_capital
        self.fee_rate = transaction_fee
        self.confidence_threshold = confidence_threshold
        self.n_days = len(self.df)

    def run_buy_and_hold(self) -> Dict[str, Any]:
        """Simulates buying on Day 1 Open and holding until the end."""
        first_open = float(self.df["Open"].iloc[0])
        shares = (self.initial_capital * (1.0 - self.fee_rate)) / first_open

        equity_curve = []
        for i in range(self.n_days):
            curr_close = float(self.df["Close"].iloc[i])
            equity_curve.append(shares * curr_close)

        equity = np.array(equity_curve)
        daily_returns = np.diff(equity) / equity[:-1]
        daily_returns = np.insert(daily_returns, 0, 0.0)

        closed_trades = [
            {
                "entry_date": str(self.df.index[0].date()),
                "exit_date": str(self.df.index[-1].date()),
                "entry_price": round(first_open, 2),
                "exit_price": round(float(self.df["Close"].iloc[-1]), 2),
                "return_pct": round(((float(self.df["Close"].iloc[-1]) - first_open) / first_open) * 100.0, 2),
                "type": "LONG",
            }
        ]

        metrics = calculate_portfolio_metrics(equity, daily_returns, closed_trades, self.initial_capital)
        _, dd_curve = calculate_max_drawdown(equity)

        return {
            "name": "Buy & Hold Benchmark",
            "equity_curve": equity,
            "drawdown_curve": dd_curve,
            "daily_returns": daily_returns,
            "metrics": metrics,
            "trades": closed_trades,
            "signals": [],
        }

    def run_technical_only(self) -> Dict[str, Any]:
        """Simulates trading based strictly on quantitative technical signals."""
        cash = self.initial_capital
        shares = 0.0
        position = 0  # 0: flat, 1: long
        entry_price = 0.0
        entry_date = None
        closed_trades = []
        trade_signals = []
        equity_curve = []

        for i in range(self.n_days):
            curr_close = float(self.df["Close"].iloc[i])
            row = self.df.iloc[i]

            # Signal generated at Close of day i
            from src.trading_strategy import calculate_technical_signal

            tech_score, _ = calculate_technical_signal(row)
            signal = "BUY" if tech_score >= 0.25 else ("SELL" if tech_score <= -0.25 else "HOLD")

            # Execution occurs on Day i+1 Open
            if i + 1 < self.n_days:
                exec_price = float(self.df["Open"].iloc[i + 1])
                exec_date = self.df.index[i + 1]

                if signal == "BUY" and position == 0:
                    position = 1
                    entry_price = exec_price
                    entry_date = exec_date
                    cost = cash * 0.50  # 50% position
                    fee = cost * self.fee_rate
                    shares = (cost - fee) / exec_price
                    cash -= cost
                    trade_signals.append({"date": exec_date, "type": "BUY", "price": exec_price})
                elif signal == "SELL" and position == 1:
                    revenue = shares * exec_price
                    fee = revenue * self.fee_rate
                    net_revenue = revenue - fee
                    cash += net_revenue
                    ret_pct = ((exec_price - entry_price) / entry_price) * 100.0
                    closed_trades.append(
                        {
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(exec_date.date()),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exec_price, 2),
                            "return_pct": round(ret_pct, 2),
                            "type": "LONG",
                        }
                    )
                    trade_signals.append({"date": exec_date, "type": "SELL", "price": exec_price})
                    position = 0
                    shares = 0.0

            total_equity = cash + (shares * curr_close)
            equity_curve.append(total_equity)

        equity = np.array(equity_curve)
        daily_returns = np.diff(equity) / np.where(equity[:-1] == 0, 1e-6, equity[:-1])
        daily_returns = np.insert(daily_returns, 0, 0.0)
        metrics = calculate_portfolio_metrics(equity, daily_returns, closed_trades, self.initial_capital)
        _, dd_curve = calculate_max_drawdown(equity)

        return {
            "name": "Technical Indicators Only",
            "equity_curve": equity,
            "drawdown_curve": dd_curve,
            "daily_returns": daily_returns,
            "metrics": metrics,
            "trades": closed_trades,
            "signals": trade_signals,
        }

    def run_eaats_pure(self) -> Dict[str, Any]:
        """Simulates EAATS ensemble without emotional adaptation (fixed 50% allocation, zero emotion score)."""
        cash = self.initial_capital
        shares = 0.0
        position = 0
        entry_price = 0.0
        entry_date = None
        closed_trades = []
        trade_signals = []
        equity_curve = []

        for i in range(self.n_days):
            curr_close = float(self.df["Close"].iloc[i])
            pred_p = float(self.predicted_prices[i]) if i < len(self.predicted_prices) else curr_close
            row = self.df.iloc[i]

            # Fixed 0.0 emotion (unaware)
            eval_res = evaluate_trading_decision(
                predicted_price=pred_p,
                current_price=curr_close,
                sentiment_score=0.0,
                latest_market_row=row,
                confidence_threshold=self.confidence_threshold,
            )
            decision = eval_res["final_decision"]

            # Next-Day Open Execution
            if i + 1 < self.n_days:
                exec_price = float(self.df["Open"].iloc[i + 1])
                exec_date = self.df.index[i + 1]

                if decision == "BUY" and position == 0:
                    position = 1
                    entry_price = exec_price
                    entry_date = exec_date
                    cost = cash * 0.50
                    fee = cost * self.fee_rate
                    shares = (cost - fee) / exec_price
                    cash -= cost
                    trade_signals.append({"date": exec_date, "type": "BUY", "price": exec_price})
                elif decision == "SELL" and position == 1:
                    revenue = shares * exec_price
                    fee = revenue * self.fee_rate
                    cash += revenue - fee
                    ret_pct = ((exec_price - entry_price) / entry_price) * 100.0
                    closed_trades.append(
                        {
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(exec_date.date()),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exec_price, 2),
                            "return_pct": round(ret_pct, 2),
                            "type": "LONG",
                        }
                    )
                    trade_signals.append({"date": exec_date, "type": "SELL", "price": exec_price})
                    position = 0
                    shares = 0.0

            total_equity = cash + (shares * curr_close)
            equity_curve.append(total_equity)

        equity = np.array(equity_curve)
        daily_returns = np.diff(equity) / np.where(equity[:-1] == 0, 1e-6, equity[:-1])
        daily_returns = np.insert(daily_returns, 0, 0.0)
        metrics = calculate_portfolio_metrics(equity, daily_returns, closed_trades, self.initial_capital)
        _, dd_curve = calculate_max_drawdown(equity)

        return {
            "name": "EAATS Pure (No Emotion)",
            "equity_curve": equity,
            "drawdown_curve": dd_curve,
            "daily_returns": daily_returns,
            "metrics": metrics,
            "trades": closed_trades,
            "signals": trade_signals,
        }

    def run_eaats_full(self) -> Dict[str, Any]:
        """Simulates full EAATS: Emotion-Aware Weighted Ensemble + Adaptive Risk Engine."""
        cash = self.initial_capital
        shares = 0.0
        position = 0
        entry_price = 0.0
        entry_date = None
        closed_trades = []
        trade_signals = []
        equity_curve = []

        from src.sentiment.emotion_engine import classify_emotion_state

        for i in range(self.n_days):
            curr_close = float(self.df["Close"].iloc[i])
            pred_p = float(self.predicted_prices[i]) if i < len(self.predicted_prices) else curr_close
            sent_val = float(self.sentiment_series.iloc[i]) if i < len(self.sentiment_series) else 0.0
            row = self.df.iloc[i]

            # 1. Classify Emotion & Determine Adaptive Risk Profile
            em_label, _ = classify_emotion_state(sent_val)
            # Rolling window for emotional volatility
            recent_sent = self.sentiment_series.iloc[max(0, i - 14) : i + 1].values
            em_vol = float(np.std(recent_sent)) if len(recent_sent) > 1 else 0.0

            risk_profile = determine_adaptive_risk_profile(em_label, em_vol, self.df.iloc[: i + 1])

            # 2. Evaluate Decision with threshold boost
            eval_res = evaluate_trading_decision(
                predicted_price=pred_p,
                current_price=curr_close,
                sentiment_score=sent_val,
                latest_market_row=row,
                confidence_threshold=self.confidence_threshold,
                threshold_boost=risk_profile["threshold_boost"],
            )
            decision = eval_res["final_decision"]

            # Override: block BUYs if panic volatility is extreme
            if decision == "BUY" and risk_profile["block_buys"]:
                decision = "HOLD"

            # 3. Next-Day Open Execution & Stop-Loss Check
            if i + 1 < self.n_days:
                exec_price = float(self.df["Open"].iloc[i + 1])
                exec_date = self.df.index[i + 1]

                # Trailing Stop-Loss Trigger
                if position == 1 and ((exec_price - entry_price) / entry_price) <= -risk_profile["stop_loss"]:
                    revenue = shares * exec_price
                    fee = revenue * self.fee_rate
                    cash += revenue - fee
                    ret_pct = ((exec_price - entry_price) / entry_price) * 100.0
                    closed_trades.append(
                        {
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(exec_date.date()),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exec_price, 2),
                            "return_pct": round(ret_pct, 2),
                            "type": "STOP_LOSS",
                        }
                    )
                    trade_signals.append({"date": exec_date, "type": "STOP_LOSS", "price": exec_price})
                    position = 0
                    shares = 0.0

                elif decision == "BUY" and position == 0:
                    position = 1
                    entry_price = exec_price
                    entry_date = exec_date
                    alloc = risk_profile["position_size"]
                    cost = cash * alloc
                    fee = cost * self.fee_rate
                    shares = (cost - fee) / exec_price
                    cash -= cost
                    trade_signals.append({"date": exec_date, "type": "BUY", "price": exec_price})

                elif decision == "SELL" and position == 1:
                    revenue = shares * exec_price
                    fee = revenue * self.fee_rate
                    cash += revenue - fee
                    ret_pct = ((exec_price - entry_price) / entry_price) * 100.0
                    closed_trades.append(
                        {
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(exec_date.date()),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exec_price, 2),
                            "return_pct": round(ret_pct, 2),
                            "type": "LONG",
                        }
                    )
                    trade_signals.append({"date": exec_date, "type": "SELL", "price": exec_price})
                    position = 0
                    shares = 0.0

            total_equity = cash + (shares * curr_close)
            equity_curve.append(total_equity)

        equity = np.array(equity_curve)
        daily_returns = np.diff(equity) / np.where(equity[:-1] == 0, 1e-6, equity[:-1])
        daily_returns = np.insert(daily_returns, 0, 0.0)
        metrics = calculate_portfolio_metrics(equity, daily_returns, closed_trades, self.initial_capital)
        _, dd_curve = calculate_max_drawdown(equity)

        return {
            "name": "EAATS Full (Emotion-Aware)",
            "equity_curve": equity,
            "drawdown_curve": dd_curve,
            "daily_returns": daily_returns,
            "metrics": metrics,
            "trades": closed_trades,
            "signals": trade_signals,
        }

    def run_rl_agent(self, agent: TabularQTradingAgent) -> Dict[str, Any]:
        """Evaluates trained Tabular Q-Learning Agent on test data."""
        cash = self.initial_capital
        shares = 0.0
        position = 0
        entry_price = 0.0
        entry_date = None
        closed_trades = []
        trade_signals = []
        equity_curve = []

        from src.sentiment.emotion_engine import classify_emotion_state

        for i in range(self.n_days):
            curr_close = float(self.df["Close"].iloc[i])
            pred_p = float(self.predicted_prices[i]) if i < len(self.predicted_prices) else curr_close
            sent_val = float(self.sentiment_series.iloc[i]) if i < len(self.sentiment_series) else 0.0
            row = self.df.iloc[i]
            em_label, _ = classify_emotion_state(sent_val)

            eval_res = evaluate_trading_decision(pred_p, curr_close, sent_val, row)
            score = eval_res["final_score"]
            rsi = float(row.get("RSI_14", 50.0))

            state = discretize_state(score, rsi, em_label, position)
            action = agent.choose_action(state, explore=False)  # Pure exploitation

            # Next-Day Open Execution: 0=Buy, 1=Hold, 2=Sell
            if i + 1 < self.n_days:
                exec_price = float(self.df["Open"].iloc[i + 1])
                exec_date = self.df.index[i + 1]

                if action == 0 and position == 0:
                    position = 1
                    entry_price = exec_price
                    entry_date = exec_date
                    cost = cash * 0.50
                    fee = cost * self.fee_rate
                    shares = (cost - fee) / exec_price
                    cash -= cost
                    trade_signals.append({"date": exec_date, "type": "BUY", "price": exec_price})
                elif action == 2 and position == 1:
                    revenue = shares * exec_price
                    fee = revenue * self.fee_rate
                    cash += revenue - fee
                    ret_pct = ((exec_price - entry_price) / entry_price) * 100.0
                    closed_trades.append(
                        {
                            "entry_date": str(entry_date.date()),
                            "exit_date": str(exec_date.date()),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exec_price, 2),
                            "return_pct": round(ret_pct, 2),
                            "type": "LONG",
                        }
                    )
                    trade_signals.append({"date": exec_date, "type": "SELL", "price": exec_price})
                    position = 0
                    shares = 0.0

            total_equity = cash + (shares * curr_close)
            equity_curve.append(total_equity)

        equity = np.array(equity_curve)
        daily_returns = np.diff(equity) / np.where(equity[:-1] == 0, 1e-6, equity[:-1])
        daily_returns = np.insert(daily_returns, 0, 0.0)
        metrics = calculate_portfolio_metrics(equity, daily_returns, closed_trades, self.initial_capital)
        _, dd_curve = calculate_max_drawdown(equity)

        return {
            "name": "Reinforcement Learning Agent",
            "equity_curve": equity,
            "drawdown_curve": dd_curve,
            "daily_returns": daily_returns,
            "metrics": metrics,
            "trades": closed_trades,
            "signals": trade_signals,
        }

    def run_all_strategies(self, agent: Optional[TabularQTradingAgent] = None) -> Dict[str, Dict[str, Any]]:
        """Runs all 5 comparative strategies across the test period."""
        logger.info(f"Running comparative walk-forward backtest over {self.n_days} test days...")
        results = {
            "buy_and_hold": self.run_buy_and_hold(),
            "technicals": self.run_technical_only(),
            "eaats_pure": self.run_eaats_pure(),
            "eaats_full": self.run_eaats_full(),
        }

        if agent is not None:
            results["rl_agent"] = self.run_rl_agent(agent)

        return results


def run_crisis_stress_test(
    df_full: pd.DataFrame,
    start_date: str,
    end_date: str,
    crisis_name: str,
) -> Dict[str, Any]:
    """Runs a dedicated stress test during a historical market crisis."""
    mask = (df_full.index >= pd.to_datetime(start_date)) & (df_full.index <= pd.to_datetime(end_date))
    df_crisis = df_full.loc[mask]

    if len(df_crisis) < 20:
        return {"error": f"Insufficient data for {crisis_name} ({len(df_crisis)} rows found)."}

    from src.sentiment.emotion_engine import compute_historical_sentiment_proxy

    proxy = compute_historical_sentiment_proxy(df_crisis)
    # Using Close prices as a baseline prediction
    engine = BacktestEngine(df_crisis, df_crisis["Close"].values, proxy)

    res_bh = engine.run_buy_and_hold()
    res_eaats = engine.run_eaats_full()

    return {
        "crisis_name": crisis_name,
        "date_range": f"{start_date} to {end_date}",
        "days": len(df_crisis),
        "buy_and_hold": res_bh["metrics"],
        "eaats_full": res_eaats["metrics"],
        "drawdown_comparison": {
            "buy_and_hold_max_dd": res_bh["metrics"]["Max_Drawdown_pct"],
            "eaats_max_dd": res_eaats["metrics"]["Max_Drawdown_pct"],
            "protection_advantage": round(
                res_bh["metrics"]["Max_Drawdown_pct"] - res_eaats["metrics"]["Max_Drawdown_pct"], 2
            ),
        },
    }
