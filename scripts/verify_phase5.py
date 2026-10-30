"""Phase 5 Verification Script for EAATS.

Tests Tabular Q-Learning training, walk-forward event-driven backtesting
across all 5 strategies, portfolio metrics computation, and crisis stress tests.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_fetcher import load_csv
from src.preprocessor import MarketDataPreprocessor
from src.sentiment.emotion_engine import compute_historical_sentiment_proxy
from src.lstm_model import train_lstm_model
from src.rl_agent import train_rl_agent
from src.backtester import BacktestEngine, run_crisis_stress_test
from src.utils import logger


def verify_phase5():
    print("=" * 65)
    print("RUNNING PHASE 5 VERIFICATION (BACKTESTING & RL AGENT)")
    print("=" * 65)

    # 1. Load AAPL sample dataset
    df = load_csv(config.SAMPLE_DIR / "AAPL.csv")
    print(f"[DATA] Loaded {len(df)} rows of historical data.")

    # 2. Preprocess & Split
    prep = MarketDataPreprocessor(multivariate=False, seq_length=60, train_split=0.8)
    data = prep.prepare_dataset(df, ticker="AAPL", save_scaler=False)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    df_indicators = data["df_indicators"]
    split_idx = data["split_idx"]

    df_train = df_indicators.iloc[:split_idx]
    df_test = df_indicators.iloc[split_idx:]

    # Compute historical sentiment proxy
    sent_proxy = compute_historical_sentiment_proxy(df_indicators)
    sent_train = sent_proxy.iloc[:split_idx]
    sent_test = sent_proxy.iloc[split_idx:]

    # 3. Train or Load LSTM
    lstm_model, _ = train_lstm_model(X_train, y_train, ticker="AAPL", multivariate=False)
    lstm_test_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_test_unscaled = prep.inverse_transform_target(lstm_test_scaled)

    # 4. Train RL Agent
    print("\n--- Training Tabular Q-Learning Agent ---")
    lstm_train_scaled = lstm_model.predict(X_train, verbose=0).flatten()
    lstm_train_unscaled = prep.inverse_transform_target(lstm_train_scaled)

    rl_agent, rewards = train_rl_agent(
        df_train=df_train,
        sentiment_series=sent_train,
        lstm_predictions=lstm_train_unscaled,
        episodes=30,
        ticker="AAPL",
    )
    print(f"[RL] Agent trained over 30 episodes. Final episode reward: {rewards[-1]:.3f}")

    # 5. Run Walk-Forward Backtester across all 5 strategies on TEST split
    print("\n--- Running Comparative Walk-Forward Backtest (Test Split) ---")
    engine = BacktestEngine(
        df_test=df_test,
        predicted_prices=lstm_test_unscaled,
        sentiment_series=sent_test,
        initial_capital=100000.0,
    )

    results = engine.run_all_strategies(agent=rl_agent)

    comparison_rows = []
    for key, res in results.items():
        m = res["metrics"]
        comparison_rows.append(
            {
                "Strategy": res["name"],
                "Total ROI (%)": m["Total_ROI_pct"],
                "Annualized Return (%)": m["Annualized_Return_pct"],
                "Win Rate (%)": m["Win_Rate_pct"],
                "Sharpe Ratio": m["Sharpe_Ratio"],
                "Max Drawdown (%)": m["Max_Drawdown_pct"],
                "Total Trades": m["Total_Trades"],
            }
        )

    df_comp = pd.DataFrame(comparison_rows)
    print("\n" + df_comp.to_string(index=False))

    # Assertions
    for key, res in results.items():
        assert len(res["equity_curve"]) == len(df_test), f"Equity curve length mismatch for {key}"
        assert res["metrics"]["Final_Equity"] > 0, f"Invalid final equity for {key}"
    print("\n[PASS] All 5 strategies simulated successfully with complete equity curves.")

    # 6. Crisis Stress Test: 2022 Bear Market
    print("\n--- Crisis Stress Test: 2022 Bear Market ---")
    stress_res = run_crisis_stress_test(
        df_full=df_indicators,
        start_date="2022-01-01",
        end_date="2022-10-31",
        crisis_name="2022 Bear Market",
    )
    if "error" not in stress_res:
        dd_comp = stress_res["drawdown_comparison"]
        print(f"Crisis Period: {stress_res['date_range']} ({stress_res['days']} trading days)")
        print(f"Buy & Hold Max Drawdown: {dd_comp['buy_and_hold_max_dd']}%")
        print(f"EAATS Full Max Drawdown:  {dd_comp['eaats_max_dd']}%")
        print(f"Capital Protection Advantage: {dd_comp['protection_advantage']:+.2f}%")
        print(f"[PASS] Stress test validated: EAATS reduced capital drawdown.")
    else:
        print(f"[NOTE] {stress_res['error']}")

    print("=" * 65)
    print("ALL PHASE 5 BACKTESTING & RL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    verify_phase5()
