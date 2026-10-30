"""Centralized Configuration & Hyperparameter Repository for EAATS.

All parameters, indicator periods, model architectures, emotion thresholds,
strategy weights, adaptive risk multipliers, and backtesting settings are defined here.
No magic numbers exist in the execution code.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. PATHS & DIRECTORIES
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, SAMPLE_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. MARKET DATA CONFIGURATION
# ==============================================================================
DEFAULT_TICKER = "AAPL"
POPULAR_TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS"]
DEFAULT_PERIOD = "2y"
SUPPORTED_PERIODS = ["6mo", "1y", "2y", "5y"]
DEFAULT_INTERVAL = "1d"
CACHE_TTL_SECONDS = 900  # 15 minutes
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Fallback APIs
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

# ==============================================================================
# 3. FEATURE ENGINEERING & INDICATORS
# ==============================================================================
SMA_PERIOD = 20
EMA_SHORT_PERIOD = 12
EMA_LONG_PERIOD = 26
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
MACD_SIGNAL_PERIOD = 9
INDICATOR_WARMUP_ROWS = 26

# Sequence & Train/Test Split
SEQUENCE_LENGTH = 60  # 60-day sliding window lookback
TRAIN_SPLIT_RATIO = 0.80  # Chronological 80/20 train/test split
SCALE_RANGE = (0, 1)

FEATURE_COLS_UNIVARIATE = ["Close"]
FEATURE_COLS_MULTIVARIATE = [
    "Close",
    "Volume",
    "SMA_20",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Pct_Change",
]

# ==============================================================================
# 4. DEEP LEARNING MODEL HYPERPARAMETERS
# ==============================================================================
RANDOM_SEED = 42

# LSTM Architecture
LSTM_CONFIG = {
    "units_l1": 128,
    "return_sequences_l1": True,
    "dropout_l1": 0.2,
    "units_l2": 64,
    "return_sequences_l2": False,
    "dropout_l2": 0.2,
    "dense_units": 32,
    "activation": "relu",
    "learning_rate": 0.001,
    "epochs": 25,
    "batch_size": 32,
    "validation_split": 0.1,
    "early_stopping_patience": 5,
}

# Transformer Encoder Architecture
TRANSFORMER_CONFIG = {
    "num_heads": 4,
    "key_dim": 32,
    "ffn_units": 64,
    "dropout_rate": 0.1,
    "dense_units": 32,
    "activation": "relu",
    "learning_rate": 0.001,
    "epochs": 25,
    "batch_size": 32,
    "validation_split": 0.1,
    "early_stopping_patience": 5,
}

# ==============================================================================
# 5. MULTI-SOURCE EMOTION DETECTION
# ==============================================================================
MAX_NEWS_HEADLINES = 20
MAX_REDDIT_POSTS = 30
MAX_STOCKTWITS_POSTS = 30
MAX_YOUTUBE_VIDEOS = 3
YOUTUBE_CHUNK_CHARS = 500

FINBERT_MODEL_NAME = "ProsusAI/finbert"

VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05

SOURCE_WEIGHTS = {
    "news": 0.40,
    "social": 0.30,
    "forums": 0.20,
    "video": 0.10,
}

# Emotion State Categories
EMOTION_THRESHOLDS = {
    "PANIC": -0.50,
    "FEAR": -0.15,
    "NEUTRAL_LOW": -0.15,
    "NEUTRAL_HIGH": 0.15,
    "CONFIDENCE": 0.50,
    "GREED": 0.50,
}

# ==============================================================================
# 6. TRADING STRATEGY (WEIGHTED VOTING)
# ==============================================================================
STRATEGY_WEIGHTS = {
    "lstm": 0.40,
    "sentiment": 0.30,
    "technical": 0.30,
}

LSTM_PRICE_CHANGE_THRESHOLD = 0.50  # >= 0.5% predicted move triggers signal
DECISION_BOUNDARY = 0.15  # > +0.15 BUY, < -0.15 SELL, else HOLD
DEFAULT_CONFIDENCE_THRESHOLD = 30.0  # signals below 30% are suppressed to HOLD

# ==============================================================================
# 7. ADAPTIVE EMOTION-BASED RISK ENGINE
# ==============================================================================
RISK_PROFILES = {
    "PANIC": {
        "position_size": 0.25,
        "stop_loss": 0.02,
        "threshold_boost": 20.0,
        "block_buys_on_high_vol": True,
        "explanation": "High market panic detected: Capital allocation reduced to 25%, strict 2% stop-loss activated, and entry barrier raised.",
    },
    "FEAR": {
        "position_size": 0.25,
        "stop_loss": 0.02,
        "threshold_boost": 10.0,
        "block_buys_on_high_vol": False,
        "explanation": "Elevated market fear: Defensive positioning with 25% allocation and tight 2% stop-loss.",
    },
    "NEUTRAL": {
        "position_size": 0.50,
        "stop_loss": 0.03,
        "threshold_boost": 0.0,
        "block_buys_on_high_vol": False,
        "explanation": "Balanced market mood: Standard 50% allocation with baseline 3% stop-loss.",
    },
    "CONFIDENCE": {
        "position_size": 0.75,
        "stop_loss": 0.04,
        "threshold_boost": 0.0,
        "block_buys_on_high_vol": False,
        "explanation": "Strong market confidence: Expansionary 75% allocation with 4% trailing stop-loss.",
    },
    "GREED": {
        "position_size": 0.50,
        "stop_loss": 0.025,
        "threshold_boost": 5.0,
        "block_buys_on_high_vol": False,
        "explanation": "Market euphoria / greed: Contrarian cap at 50% position size and tightened 2.5% stop-loss to guard against reversals.",
    },
}

# ==============================================================================
# 8. REINFORCEMENT LEARNING (Q-LEARNING & DDPG)
# ==============================================================================
RL_CONFIG = {
    "alpha": 0.1,  # Learning rate
    "gamma": 0.95,  # Discount factor
    "epsilon_start": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "episodes": 40,
    "drawdown_penalty_weight": 0.5,
    "transaction_cost": 0.001,
}
ENABLE_DDPG_STRETCH = False

# ==============================================================================
# 9. BACKTESTING & SIMULATION
# ==============================================================================
INITIAL_CAPITAL = 100000.0
TRANSACTION_FEE = 0.001  # 0.1% per trade
RISK_FREE_RATE = 0.0  # Annualised baseline
ANNUALIZATION_FACTOR = 252  # Trading days per year

CRISIS_PERIODS = {
    "COVID_CRASH": ("2020-02-01", "2020-04-30"),
    "BEAR_MARKET_2022": ("2022-01-01", "2022-10-31"),
}
