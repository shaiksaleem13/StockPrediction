"""Reinforcement Learning Q-Learning Agent & Trading Environment for EAATS.

Main Agent: Tabular Q-Learning with state discretization:
State = (Score_Bucket, RSI_Bucket, Emotion_Bucket, Position_Held)
Actions = Buy (0), Hold (1), Sell (2)
Reward = Daily Return - Transaction Fee - Drawdown Penalty

Trained strictly on the TRAIN split; evaluated on the TEST split.
Includes Q-table persistence and learning curve logging.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

import config
from src.utils import logger, set_seed


def discretize_state(
    score: float,
    rsi: float,
    emotion_label: str,
    position_held: int,
) -> Tuple[int, int, int, int]:
    """Discretizes continuous market inputs into a compact state tuple."""
    # 1. Final Score Bucket (5 levels: 0=Strong Bear to 4=Strong Bull)
    if score <= -0.30:
        score_b = 0
    elif score <= -0.10:
        score_b = 1
    elif score < 0.10:
        score_b = 2
    elif score < 0.30:
        score_b = 3
    else:
        score_b = 4

    # 2. RSI Bucket (3 levels: 0=Oversold, 1=Normal, 2=Overbought)
    if rsi < config.RSI_OVERSOLD:
        rsi_b = 0
    elif rsi > config.RSI_OVERBOUGHT:
        rsi_b = 2
    else:
        rsi_b = 1

    # 3. Emotion Bucket (5 levels)
    emotion_map = {"PANIC": 0, "FEAR": 1, "NEUTRAL": 2, "CONFIDENCE": 3, "GREED": 4}
    emotion_b = emotion_map.get(emotion_label, 2)

    # 4. Position held (0=No position, 1=Position held)
    pos_b = 1 if position_held else 0

    return (score_b, rsi_b, emotion_b, pos_b)


class TabularQTradingAgent:
    """Tabular Q-Learning Agent for algorithmic trading simulation."""

    def __init__(
        self,
        alpha: float = config.RL_CONFIG["alpha"],
        gamma: float = config.RL_CONFIG["gamma"],
        epsilon: float = config.RL_CONFIG["epsilon_start"],
        epsilon_decay: float = config.RL_CONFIG["epsilon_decay"],
        epsilon_min: float = config.RL_CONFIG["epsilon_min"],
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = 3  # 0: Buy, 1: Hold, 2: Sell
        # State space dimensions: (5, 3, 5, 2, 3)
        self.q_table = np.zeros((5, 3, 5, 2, self.num_actions), dtype=np.float32)
        self.training_rewards: List[float] = []

    def choose_action(self, state: Tuple[int, int, int, int], explore: bool = True) -> int:
        """Selects action via epsilon-greedy policy."""
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
    ) -> None:
        """Standard Q-Learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Decays exploration parameter after each training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: Path) -> None:
        """Saves Q-table and hyperparameter state."""
        joblib.dump(
            {
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "training_rewards": self.training_rewards,
            },
            filepath,
        )

    def load(self, filepath: Path) -> None:
        """Loads Q-table checkpoint."""
        data = joblib.load(filepath)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        self.training_rewards = data.get("training_rewards", [])


def train_rl_agent(
    df_train: pd.DataFrame,
    sentiment_series: pd.Series,
    lstm_predictions: np.ndarray,
    episodes: int = config.RL_CONFIG["episodes"],
    ticker: str = "TICKER",
) -> Tuple[TabularQTradingAgent, List[float]]:
    """Trains the Q-Learning agent on the training split using historical observations."""
    set_seed(config.RANDOM_SEED)
    agent = TabularQTradingAgent()

    close_prices = df_train["Close"].values
    open_prices = df_train["Open"].values
    rsi_values = df_train["RSI_14"].values
    n_days = len(close_prices)

    from src.trading_strategy import evaluate_trading_decision
    from src.sentiment.emotion_engine import classify_emotion_state

    # Precompute state features for speed
    precomputed_states = []
    for t in range(n_days):
        sent_val = float(sentiment_series.iloc[t]) if t < len(sentiment_series) else 0.0
        em_label, _ = classify_emotion_state(sent_val)
        pred_p = float(lstm_predictions[t]) if t < len(lstm_predictions) else float(close_prices[t])
        eval_res = evaluate_trading_decision(pred_p, float(close_prices[t]), sent_val, df_train.iloc[t])
        precomputed_states.append((eval_res["final_score"], rsi_values[t], em_label))

    logger.info(f"Training Tabular Q-Learning Agent for {ticker} across {episodes} episodes...")

    for ep in range(episodes):
        position = 0  # 0: flat, 1: long
        entry_price = 0.0
        cash = config.INITIAL_CAPITAL
        total_ep_reward = 0.0
        peak_portfolio_val = cash

        for t in range(n_days - 1):
            score, rsi, em_label = precomputed_states[t]
            curr_state = discretize_state(score, rsi, em_label, position)
            action = agent.choose_action(curr_state, explore=True)

            # Execution happens at Next-Day Open to prevent look-ahead bias
            next_open = open_prices[t + 1]
            next_close = close_prices[t + 1]

            reward = 0.0
            fee = config.RL_CONFIG["transaction_cost"]

            # Action Logic: 0=Buy, 1=Hold, 2=Sell
            if action == 0 and position == 0:  # Open Long
                position = 1
                entry_price = next_open
                reward -= fee  # Transaction cost penalty
            elif action == 2 and position == 1:  # Close Long
                trade_ret = (next_open - entry_price) / entry_price
                reward += trade_ret - fee
                position = 0
                entry_price = 0.0
            elif position == 1:  # Holding long overnight
                step_ret = (next_close - close_prices[t]) / close_prices[t]
                reward += step_ret

            # Drawdown penalty
            curr_equity = cash * (1.0 + ((next_close - entry_price) / entry_price) if position == 1 else 1.0)
            if curr_equity > peak_portfolio_val:
                peak_portfolio_val = curr_equity
            dd = (curr_equity - peak_portfolio_val) / peak_portfolio_val
            reward += dd * config.RL_CONFIG["drawdown_penalty_weight"]

            # Next state
            next_score, next_rsi, next_em = precomputed_states[t + 1]
            next_state = discretize_state(next_score, next_rsi, next_em, position)

            agent.update(curr_state, action, reward, next_state)
            total_ep_reward += reward

        agent.decay_epsilon()
        agent.training_rewards.append(round(total_ep_reward, 3))

    # Save Q-table
    safe_ticker = ticker.replace(".", "_")
    save_path = config.MODELS_DIR / f"{safe_ticker}_q_table.joblib"
    agent.save(save_path)
    logger.info(f"Saved trained Q-table to {save_path}")

    return agent, agent.training_rewards
