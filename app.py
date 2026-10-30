"""EAATS - Emotion-Aware Adaptive Trading System.

High-Performance Quantitative Financial Dashboard.
Combines Deep Learning Price Forecasting (Stacked LSTM & Transformer),
Multi-Source Natural Language Emotion Analysis (FinBERT & VADER),
Welles Wilder Quantitative Technical Indicators,
Adaptive Emotion-Based Risk Management, and Event-Driven Backtesting.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import config
from src.adaptive_risk import determine_adaptive_risk_profile
from src.backtester import BacktestEngine, run_crisis_stress_test
from src.data_fetcher import fetch_stock_data, load_csv
from src.lstm_model import (
    calculate_metrics,
    predict_5day_experimental,
    predict_next_day,
    train_lstm_model,
)
from src.preprocessor import MarketDataPreprocessor, add_technical_indicators
from src.rl_agent import TabularQTradingAgent
from src.sentiment.emotion_engine import (
    analyze_market_emotion,
    classify_emotion_state,
    compute_historical_sentiment_proxy,
)
from src.trading_strategy import evaluate_trading_decision
from src.transformer_model import compute_model_comparison_table, train_transformer_model
from src.utils import logger

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & ULTRA-PREMIUM THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="EAATS // Quant Intelligence Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0F172A 0%, #080D1A 60%, #030712 100%);
        color: #F8FAFC;
    }
    
    /* Sleek Custom Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #080D1A; }
    ::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #38BDF8; }

    /* Top Executive Header Bar */
    .terminal-header {
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }
    .terminal-title {
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .terminal-tagline {
        color: #94A3B8;
        font-size: 13px;
        font-weight: 500;
        margin-top: 2px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-live {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Glassmorphic Metric Cards */
    .metric-card-glass {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-card-glass:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .metric-label-modern {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value-modern {
        color: #F8FAFC;
        font-size: 26px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
        letter-spacing: -0.5px;
    }
    .metric-delta-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
    }
    .delta-up { background: rgba(16, 185, 129, 0.2); color: #34D399; }
    .delta-down { background: rgba(239, 68, 68, 0.2); color: #F87171; }

    /* Hero Recommendation Badges */
    .hero-badge-box {
        text-align: center;
        padding: 32px 20px;
        border-radius: 16px;
        margin: 16px 0 24px 0;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.8) 0%, rgba(3, 7, 18, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }
    .badge-signal-buy {
        display: inline-block;
        padding: 16px 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.45);
        border: 1px solid #34D399;
    }
    .badge-signal-sell {
        display: inline-block;
        padding: 16px 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.45);
        border: 1px solid #F87171;
    }
    .badge-signal-hold {
        display: inline-block;
        padding: 16px 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        color: #FFFFFF;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.45);
        border: 1px solid #FBBF24;
    }
    
    /* Suppressed Signal Alert */
    .suppressed-banner {
        background: rgba(180, 83, 9, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 18px;
        color: #FDE68A;
        font-size: 14px;
        font-weight: 500;
        margin: 16px 0;
    }

    /* Emotion Spectrum Badge */
    .emotion-chip {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    .emotion-panic { background: #7F1D1D; color: #FCA5A5; border: 1px solid #EF4444; }
    .emotion-fear { background: #991B1B; color: #FECACA; border: 1px solid #F87171; }
    .emotion-neutral { background: #334155; color: #E2E8F0; border: 1px solid #64748B; }
    .emotion-confidence { background: #065F46; color: #A7F3D0; border: 1px solid #34D399; }
    .emotion-greed { background: #78350F; color: #FDE68A; border: 1px solid #FBBF24; }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 14px;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. SIDEBAR CONTROLS & MODEL TOGGLES
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="font-size: 20px; font-weight: 800; color: #F8FAFC;">⚙️ QUANT ENGINE</div>
            <div style="font-size: 12px; color: #38BDF8; font-weight: 600;">EAATS CONFIGURATION HUB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ticker_choice = st.selectbox(
        "Active Asset Ticker",
        options=["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "Custom Ticker"],
        index=0,
        help="Select a US or Indian equity, or input any global symbol.",
    )
    if ticker_choice == "Custom Ticker":
        ticker = st.text_input("Enter Ticker Symbol", value="NVDA").upper().strip()
    else:
        ticker = ticker_choice

    period = st.selectbox(
        "Historical Timeframe",
        options=config.SUPPORTED_PERIODS,
        index=2,  # Default 2y
        help="Historical lookback window for indicator and model training.",
    )

    uploaded_file = st.file_uploader(
        "Upload Custom Historical CSV",
        type=["csv"],
        help="Columns required: Date, Open, High, Low, Close, Volume",
    )

    st.markdown("---")
    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #94A3B8; margin-bottom: 8px;'>AI MODELS & SIGNALS</div>", unsafe_allow_html=True)
    use_multivariate = st.toggle("Multivariate DL Features", value=False, help="Incorporate Volume, SMA, EMA, RSI, and MACD into DL inputs.")
    use_transformer = st.toggle("Include Transformer Model", value=True, help="Train parallel 4-head Multi-Head Attention network.")
    enable_social_sentiment = st.toggle("Scrape Social & Forum Feeds", value=True, help="Scrape StockTwits and Reddit financial discussions.")
    enable_video_sentiment = st.toggle("Analyze YouTube Commentary", value=True, help="Process finance video transcripts via NLP.")

    st.markdown("---")
    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #94A3B8; margin-bottom: 8px;'>EXECUTION & RISK</div>", unsafe_allow_html=True)
    confidence_slider = st.slider(
        "Confidence Suppression Barrier (%)",
        min_value=10.0,
        max_value=60.0,
        value=config.DEFAULT_CONFIDENCE_THRESHOLD,
        step=5.0,
        help="Signals with lower confidence are downgraded to HOLD to avoid noisy chops.",
    )
    initial_capital = st.number_input(
        "Simulation Starting Capital ($)",
        min_value=1000.0,
        value=config.INITIAL_CAPITAL,
        step=5000.0,
    )

    st.markdown("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        refresh_btn = st.button("🔄 Sync Feeds", use_container_width=True)
    with col_b2:
        retrain_btn = st.button("🧠 Retrain AI", use_container_width=True)

    # ── Download pre-fetched data as CSV ──────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='font-size: 13px; font-weight: 700; color: #94A3B8; margin-bottom: 8px;'>"
        "📥 DOWNLOAD SAMPLE DATA</div>",
        unsafe_allow_html=True,
    )
    st.caption("Download one of the pre-bundled datasets to use as a custom CSV upload.")

    _download_ticker = st.selectbox(
        "Choose dataset",
        options=["AAPL", "TSLA", "RELIANCE.NS"],
        key="dl_ticker",
    )

    _sample_path = PROJECT_ROOT / "data" / "sample" / f"{_download_ticker}.csv"
    if _sample_path.exists():
        _csv_bytes = _sample_path.read_bytes()
        st.download_button(
            label=f"⬇️ Download {_download_ticker}.csv",
            data=_csv_bytes,
            file_name=f"{_download_ticker}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning(f"Sample file not found locally. Run `scripts/download_samples.py` first.")

    st.caption("⚡ Market data via yfinance (near-real-time / delayed 15m).")

# ==============================================================================
# 3. BACKEND PIPELINE EXECUTION
# ==============================================================================
@st.cache_data(ttl=config.CACHE_TTL_SECONDS)
def get_cached_stock_data(ticker_symbol: str, hist_period: str, force: bool = False) -> pd.DataFrame:
    return fetch_stock_data(ticker_symbol, period=hist_period, force_refresh=force)


try:
    if uploaded_file is not None:
        df_raw = load_csv(uploaded_file)
        active_ticker = "UPLOADED_CSV"
        st.sidebar.success(f"✓ Ingested CSV: {len(df_raw)} trading bars.")
    else:
        active_ticker = ticker
        df_raw = get_cached_stock_data(active_ticker, period, force=refresh_btn)
except Exception as e:
    logger.error(f"Error loading stock data: {e}")
    st.error(f"⚠️ Error loading market data for '{ticker}': {e}. Please check the symbol or upload a historical CSV.")
    st.stop()

# Feature Engineering
preprocessor = MarketDataPreprocessor(
    multivariate=use_multivariate,
    seq_length=config.SEQUENCE_LENGTH,
    train_split=config.TRAIN_SPLIT_RATIO,
)
try:
    prep_data = preprocessor.prepare_dataset(df_raw, ticker=active_ticker, save_scaler=True)
except Exception as prep_err:
    st.error(f"Data preprocessing failed: {prep_err}")
    st.stop()

df_indicators = prep_data["df_indicators"]
X_train, y_train = prep_data["X_train"], prep_data["y_train"]
X_test, y_test = prep_data["X_test"], prep_data["y_test"]
split_idx = prep_data["split_idx"]
test_dates = prep_data["test_dates"]
target_scaler = prep_data["target_scaler"]

df_train = df_indicators.iloc[:split_idx]
df_test = df_indicators.iloc[split_idx:]
y_test_unscaled = preprocessor.inverse_transform_target(y_test)
prev_test_prices = preprocessor.inverse_transform_target(X_test[:, -1, 0])

# Deep Learning Inference
with st.spinner("Executing neural inference engines..."):
    lstm_model, _ = train_lstm_model(
        X_train,
        y_train,
        ticker=active_ticker,
        multivariate=use_multivariate,
        force_retrain=retrain_btn,
    )
    lstm_test_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_test_unscaled = preprocessor.inverse_transform_target(lstm_test_scaled)
    lstm_metrics = calculate_metrics(y_test_unscaled, lstm_test_unscaled, prev_test_prices)

    trans_model = None
    trans_test_unscaled = None
    trans_metrics = None
    if use_transformer:
        trans_model, _ = train_transformer_model(
            X_train,
            y_train,
            ticker=active_ticker,
            multivariate=use_multivariate,
            force_retrain=retrain_btn,
        )
        trans_test_scaled = trans_model.predict(X_test, verbose=0).flatten()
        trans_test_unscaled = preprocessor.inverse_transform_target(trans_test_scaled)
        trans_metrics = calculate_metrics(y_test_unscaled, trans_test_unscaled, prev_test_prices)

# Next-Day Target Price
last_60_scaled = X_test[-1]
next_day_predicted_close = predict_next_day(lstm_model, last_60_scaled, target_scaler)
last_actual_close = float(df_indicators["Close"].iloc[-1])

# Multi-Source Emotion Engine
with st.spinner("Quantifying market emotional state across news and social media..."):
    try:
        emotion_results = analyze_market_emotion(
            active_ticker,
            enable_social=enable_social_sentiment,
            enable_video=enable_video_sentiment,
        )
    except Exception as em_err:
        logger.warning(f"Live emotion scraping error: {em_err}; using neutral default.")
        emotion_results = {
            "composite_score": 0.0,
            "emotion_label": "NEUTRAL",
            "explanation": "Live sentiment feed unavailable; using neutral baseline.",
            "emotional_volatility": 0.0,
            "details": {"source_summaries": {}},
            "all_items": [],
            "counts": {"news": 0, "social": 0, "forums": 0, "video": 0},
        }

composite_sentiment = emotion_results["composite_score"]
emotion_label = emotion_results["emotion_label"]
emotional_vol = emotion_results["emotional_volatility"]

# Historical Sentiment Proxy
historical_sentiment_series = compute_historical_sentiment_proxy(df_indicators)
sent_test_series = historical_sentiment_series.iloc[split_idx:]

# Decision Engine & Adaptive Risk
latest_row = df_indicators.iloc[-1]
risk_profile = determine_adaptive_risk_profile(emotion_label, emotional_vol, df_indicators)

decision_info = evaluate_trading_decision(
    predicted_price=next_day_predicted_close,
    current_price=last_actual_close,
    sentiment_score=composite_sentiment,
    latest_market_row=latest_row,
    confidence_threshold=confidence_slider,
    threshold_boost=risk_profile["threshold_boost"],
)

if decision_info["final_decision"] == "BUY" and risk_profile["block_buys"]:
    decision_info["final_decision"] = "HOLD"
    decision_info["is_suppressed"] = True
    decision_info["suppression_reason"] = "BUY order blocked due to severe market emotional volatility."

# ==============================================================================
# 4. TOP EXECUTIVE TERMINAL HEADER
# ==============================================================================
st.markdown(
    f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">EAATS // QUANT INTELLIGENCE TERMINAL</div>
            <div class="terminal-tagline">Real-Time Stock Market Prediction & Emotion-Aware Algorithmic Trading Simulation</div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div class="status-pill status-live">
                <div class="status-dot"></div>
                SYSTEM LIVE
            </div>
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); padding: 5px 12px; border-radius: 20px; font-size: 13px; font-weight: 700; color: #38BDF8;">
                {active_ticker}
            </div>
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); padding: 5px 12px; border-radius: 20px; font-size: 13px; font-weight: 700; color: #F59E0B;">
                MOOD: {emotion_label}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 5. TAB INTERFACE
# ==============================================================================
tabs = st.tabs(
    [
        "📈 Market Action",
        "🔮 AI Price Forecast",
        "🧠 Emotion & Mood",
        "🎯 Signal & Risk Hub",
        "💼 Quant Backtester",
        "📚 Academic Whitepaper",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: MARKET ACTION (TRADINGVIEW CALIBER CHARTING)
# ------------------------------------------------------------------------------
with tabs[0]:
    # Top KPI Ribbon
    pct_day = float(df_indicators["Pct_Change"].iloc[-1])
    high_52 = float(df_raw["High"].max())
    low_52 = float(df_raw["Low"].min())
    cur_rsi = float(df_indicators["RSI_14"].iloc[-1])
    delta_class = "delta-up" if pct_day >= 0 else "delta-down"
    delta_sign = "+" if pct_day >= 0 else ""

    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">Last Traded Price</div>
                <div class="metric-value-modern">${last_actual_close:.2f}</div>
                <div class="metric-delta-pill {delta_class}">{delta_sign}{pct_day:.2f}% (24h)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k2:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">52-Week Range</div>
                <div class="metric-value-modern" style="font-size: 20px;">${low_52:.2f} - ${high_52:.2f}</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">Historical spread</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k3:
        rsi_color = "#34D399" if cur_rsi < 30 else ("#F87171" if cur_rsi > 70 else "#94A3B8")
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">Wilder's RSI (14)</div>
                <div class="metric-value-modern" style="color: {rsi_color};">{cur_rsi:.1f}</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">{"Oversold (<30)" if cur_rsi < 30 else ("Overbought (>70)" if cur_rsi > 70 else "Neutral Zone")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k4:
        ema_diff = float(df_indicators["EMA_12"].iloc[-1] - df_indicators["EMA_26"].iloc[-1])
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">MACD Spread</div>
                <div class="metric-value-modern">{ema_diff:+.3f}</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">EMA12 - EMA26</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k5:
        emotion_css = f"emotion-{emotion_label.lower()}"
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">Market Emotion</div>
                <div style="margin-top: 6px;"><span class="emotion-chip {emotion_css}">{emotion_label}</span></div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 6px;">Volatility: {emotional_vol:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Multi-Subplot Chart
    fig_dash = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=(
            f"<b>{active_ticker} Candlesticks & Adaptive Moving Averages</b>",
            "<b>Wilder's RSI (14-Period Momentum)</b>",
            "<b>MACD Histogram & Signal Oscillator</b>",
        ),
    )

    # 1. Candlestick & Moving Averages
    fig_dash.add_trace(
        go.Candlestick(
            x=df_indicators.index,
            open=df_indicators["Open"],
            high=df_indicators["High"],
            low=df_indicators["Low"],
            close=df_indicators["Close"],
            name="OHLC",
            increasing_line_color="#00F5A0",
            decreasing_line_color="#FF2A55",
            increasing_fillcolor="rgba(0, 245, 160, 0.8)",
            decreasing_fillcolor="rgba(255, 42, 85, 0.8)",
        ),
        row=1,
        col=1,
    )
    fig_dash.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators["SMA_20"], name="SMA 20", line=dict(color="#38BDF8", width=1.5)),
        row=1,
        col=1,
    )
    fig_dash.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators["EMA_12"], name="EMA 12", line=dict(color="#FBBF24", width=1.5)),
        row=1,
        col=1,
    )
    fig_dash.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators["EMA_26"], name="EMA 26", line=dict(color="#C084FC", width=1.5)),
        row=1,
        col=1,
    )

    # 2. RSI with Threshold Bands
    fig_dash.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators["RSI_14"], name="RSI", line=dict(color="#EC4899", width=1.75)),
        row=2,
        col=1,
    )
    fig_dash.add_hline(y=70, line_dash="dash", line_color="#EF4444", line_width=1, row=2, col=1)
    fig_dash.add_hline(y=30, line_dash="dash", line_color="#10B981", line_width=1, row=2, col=1)

    # 3. MACD Subplot
    fig_dash.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators["MACD"], name="MACD", line=dict(color="#06B6D4", width=1.5)),
        row=3,
        col=1,
    )
    fig_dash.add_trace(
        go.Scatter(x=df_indicators.index, y=df_indicators["MACD_Signal"], name="Signal", line=dict(color="#FB923C", width=1.5)),
        row=3,
        col=1,
    )
    colors_hist = ["#00F5A0" if v >= 0 else "#FF2A55" for v in df_indicators["MACD_Hist"]]
    fig_dash.add_trace(
        go.Bar(x=df_indicators.index, y=df_indicators["MACD_Hist"], name="Histogram", marker_color=colors_hist),
        row=3,
        col=1,
    )

    fig_dash.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(11, 15, 25, 0.7)",
        paper_bgcolor="rgba(11, 15, 25, 0.7)",
        height=820,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_dash.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
    fig_dash.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
    st.plotly_chart(fig_dash, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: AI PRICE FORECAST (DEEP LEARNING SUITE)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### 🔮 Deep Learning Multi-Model Price Forecasting")

    # Metrics HUD
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">LSTM Point Error (RMSE)</div>
                <div class="metric-value-modern">${lstm_metrics['RMSE']:.2f}</div>
                <div style="font-size: 11px; color: #34D399; margin-top: 4px;">Root Mean Squared Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m2:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">Mean Absolute Error</div>
                <div class="metric-value-modern">${lstm_metrics['MAE']:.2f}</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">Average Dollar Deviation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m3:
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">Percentage Accuracy (MAPE)</div>
                <div class="metric-value-modern">{lstm_metrics['MAPE']:.2f}%</div>
                <div style="font-size: 11px; color: #38BDF8; margin-top: 4px;">Mean Absolute % Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m4:
        dir_color = "#34D399" if lstm_metrics['Directional_Accuracy'] >= 50 else "#F87171"
        st.markdown(
            f"""
            <div class="metric-card-glass">
                <div class="metric-label-modern">Directional Accuracy</div>
                <div class="metric-value-modern" style="color: {dir_color};">{lstm_metrics['Directional_Accuracy']:.1f}%</div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">Market Move Sign Hit Rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Interactive Forecast vs Actual Chart
    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test_unscaled,
            name="Actual Close ($)",
            line=dict(color="#F8FAFC", width=2.5),
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=test_dates,
            y=lstm_test_unscaled,
            name="Stacked LSTM Forecast ($)",
            line=dict(color="#10B981", width=2.2),
        )
    )
    if trans_test_unscaled is not None:
        fig_pred.add_trace(
            go.Scatter(
                x=test_dates,
                y=trans_test_unscaled,
                name="Transformer Encoder ($)",
                line=dict(color="#38BDF8", width=1.8, dash="dot"),
            )
        )

    fig_pred.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(11, 15, 25, 0.7)",
        paper_bgcolor="rgba(11, 15, 25, 0.7)",
        title="<b>Out-of-Sample Walk-Forward Neural Validation (Unseen 20% Split)</b>",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_pred.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
    fig_pred.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
    st.plotly_chart(fig_pred, use_container_width=True)

    # Side-by-Side Model Comparison Table
    if trans_test_unscaled is not None:
        st.markdown("#### 🏆 Model Benchmark Leaderboard")
        comp_df = compute_model_comparison_table(
            y_test_unscaled,
            lstm_test_unscaled,
            trans_test_unscaled,
            prev_test_prices,
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("ℹ️ The Naive Baseline ($y_{t+1} = y_t$) achieves low point-error in random walk assets, but yields 0.0% directional accuracy.")

    # Target Price & Experimental Horizon
    st.markdown("---")
    col_fc1, col_fc2 = st.columns([1, 2])
    with col_fc1:
        st.markdown("#### Next-Day Close Target")
        move_pct = ((next_day_predicted_close - last_actual_close) / last_actual_close) * 100.0
        delta_css = "delta-up" if move_pct >= 0 else "delta-down"
        delta_sgn = "+" if move_pct >= 0 else ""
        st.markdown(
            f"""
            <div class="metric-card-glass" style="border: 1px solid rgba(56, 189, 248, 0.4);">
                <div class="metric-label-modern">Next Trading Session Target</div>
                <div class="metric-value-modern" style="color: #38BDF8;">${next_day_predicted_close:.2f}</div>
                <div class="metric-delta-pill {delta_css}">{delta_sgn}{move_pct:.2f}% expected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_fc2:
        st.markdown("#### 5-Day Recursive Forecast Trajectory (Experimental)")
        five_day = predict_5day_experimental(lstm_model, last_60_scaled, target_scaler)
        days_ahead = [f"Session +{i+1}" for i in range(5)]
        df_5day = pd.DataFrame({"Horizon": days_ahead, "Forecast Target ($)": [f"${p:.2f}" for p in five_day]})
        st.dataframe(df_5day.T, use_container_width=True)
        st.caption("⚠️ Multi-step recursive forecasting compounds error over consecutive steps; provided for trend estimation only.")

# ------------------------------------------------------------------------------
# TAB 3: EMOTION & MOOD (SENTIMENT INTELLIGENCE)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### 🧠 Natural Language Emotion Detection Engine")

    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        # Gauge Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=composite_sentiment,
                title={"text": f"<b>Market Mood: {emotion_label}</b>", "font": {"size": 20, "color": "#F8FAFC"}},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [-1.0, 1.0], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": "#38BDF8", "thickness": 0.25},
                    "steps": [
                        {"range": [-1.0, -0.5], "color": "#7F1D1D"},
                        {"range": [-0.5, -0.15], "color": "#991B1B"},
                        {"range": [-0.15, 0.15], "color": "#334155"},
                        {"range": [0.15, 0.5], "color": "#065F46"},
                        {"range": [0.5, 1.0], "color": "#047857"},
                    ],
                    "threshold": {
                        "line": {"color": "#F59E0B", "width": 4},
                        "thickness": 0.75,
                        "value": composite_sentiment,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(11, 15, 25, 0.7)",
            paper_bgcolor="rgba(11, 15, 25, 0.7)",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.markdown(
            f"""
            <div class="metric-card-glass" style="margin-top: 10px;">
                <div style="font-size: 16px; font-weight: 700; color: #F8FAFC;">Active Psychological Regime</div>
                <div style="margin: 10px 0;"><span class="emotion-chip emotion-{emotion_label.lower()}">{emotion_label}</span></div>
                <div style="font-size: 13px; color: #CBD5E1; line-height: 1.5;">{emotion_results["explanation"]}</div>
                <div style="margin-top: 14px; font-size: 13px; color: #94A3B8;">
                    <b>Emotional Volatility ($\sigma$):</b> <span style="color:#F59E0B; font-family:'JetBrains Mono';">{emotional_vol:.3f}</span>
                    <br>
                    <b>Scraped Coverage:</b> News: {emotion_results['counts']['news']} | StockTwits: {emotion_results['counts']['social']} | Reddit: {emotion_results['counts']['forums']} | YouTube: {emotion_results['counts']['video']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Scraped Feed Stream
    st.markdown("#### 📰 Multi-Source Sentiment Intelligence Feed")
    items = emotion_results.get("all_items", [])
    if items:
        df_feed = pd.DataFrame(items)
        show_cols = [c for c in ["source", "text", "score", "sentiment_label", "url"] if c in df_feed.columns]
        st.dataframe(df_feed[show_cols].head(30), use_container_width=True, hide_index=True)
    else:
        st.info("No live posts scraped for this ticker. Using historical sentiment proxy.")

# ------------------------------------------------------------------------------
# TAB 4: SIGNAL & RISK HUB (HERO TERMINAL)
# ------------------------------------------------------------------------------
with tabs[3]:
    rec_decision = decision_info["final_decision"]
    conf = decision_info["confidence"]
    badge_cls = "badge-signal-buy" if rec_decision == "BUY" else ("badge-signal-sell" if rec_decision == "SELL" else "badge-signal-hold")

    st.markdown(
        f"""
        <div class="hero-badge-box">
            <div style="font-size: 14px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">
                ALGORITHMIC TRADING DECISION
            </div>
            <div class="{badge_cls}">{rec_decision}</div>
            <div style="margin-top: 14px; font-size: 18px; color: #F8FAFC; font-weight: 600;">
                Signal Confidence: <span style="color: #38BDF8; font-family: 'JetBrains Mono'; font-size: 22px;">{conf:.1f}%</span>
                <span style="color: #64748B; font-size: 14px;">(Required Barrier: {decision_info['effective_threshold']:.1f}%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if decision_info["is_suppressed"]:
        st.markdown(
            f"""
            <div class="suppressed-banner">
                ⚠️ <b>FALSE SIGNAL SUPPRESSED:</b> {decision_info['suppression_reason']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_sig1, col_sig2 = st.columns([1, 1])
    with col_sig1:
        st.markdown("#### ⚖️ 3-Signal Weighted Voting Matrix")
        st.dataframe(decision_info["breakdown_table"], use_container_width=True, hide_index=True)
        st.caption("Final Score = (0.40 × LSTM) + (0.30 × Sentiment) + (0.30 × Technicals)")

    with col_sig2:
        st.markdown("#### 🛡️ Adaptive Emotion-Driven Risk Controls")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(
                f"""
                <div class="metric-card-glass">
                    <div class="metric-label-modern">Position Sizing</div>
                    <div class="metric-value-modern" style="color: #38BDF8;">{risk_profile['position_size_pct']}</div>
                    <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">Dynamic Capital Allocation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_r2:
            st.markdown(
                f"""
                <div class="metric-card-glass">
                    <div class="metric-label-modern">Trailing Stop-Loss</div>
                    <div class="metric-value-modern" style="color: #F87171;">{risk_profile['stop_loss_pct']}</div>
                    <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">Capital Protection Limit</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); padding: 12px 16px; border-radius: 8px; margin-top: 12px;">
                <b style="color: #F8FAFC;">Active Rationale:</b> <span style="color: #CBD5E1;">{risk_profile['explanation']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------------------
# TAB 5: QUANT BACKTESTER (STRATEGY ARENA)
# ------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### 💼 Event-Driven Walk-Forward Backtesting Arena")
    st.caption("Strictly executed at Next-Day Open price at 0.1% transaction cost (zero look-ahead bias).")

    backtest_engine = BacktestEngine(
        df_test=df_test,
        predicted_prices=lstm_test_unscaled,
        sentiment_series=sent_test_series,
        initial_capital=initial_capital,
        confidence_threshold=confidence_slider,
    )

    # RL Agent
    rl_agent = TabularQTradingAgent()
    q_table_path = config.MODELS_DIR / f"{active_ticker.replace('.', '_')}_q_table.joblib"
    if q_table_path.exists():
        rl_agent.load(q_table_path)
    else:
        rl_agent = None

    all_sim_results = backtest_engine.run_all_strategies(agent=rl_agent)

    # Strategy Leaderboard
    sim_rows = []
    for key, s in all_sim_results.items():
        m = s["metrics"]
        sim_rows.append(
            {
                "Strategy": s["name"],
                "Total ROI (%)": m["Total_ROI_pct"],
                "Annualized Return (%)": m["Annualized_Return_pct"],
                "Win Rate (%)": m["Win_Rate_pct"],
                "Sharpe Ratio": m["Sharpe_Ratio"],
                "Max Drawdown (%)": m["Max_Drawdown_pct"],
                "Total Trades": m["Total_Trades"],
                "Final Equity ($)": f"${m['Final_Equity']:,.2f}",
            }
        )
    df_sim_table = pd.DataFrame(sim_rows).sort_values("Sharpe Ratio", ascending=False)
    st.dataframe(df_sim_table, use_container_width=True, hide_index=True)

    # Plotly Equity Curves
    fig_equity = go.Figure()
    palette = {
        "buy_and_hold": "#94A3B8",
        "technicals": "#F59E0B",
        "eaats_pure": "#3B82F6",
        "eaats_full": "#10B981",
        "rl_agent": "#EC4899",
    }
    for k, s in all_sim_results.items():
        fig_equity.add_trace(
            go.Scatter(
                x=df_test.index,
                y=s["equity_curve"],
                name=s["name"],
                line=dict(color=palette.get(k, "#FFFFFF"), width=2.2),
            )
        )
    fig_equity.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(11, 15, 25, 0.7)",
        paper_bgcolor="rgba(11, 15, 25, 0.7)",
        title="<b>Cumulative Equity Curves ($100k Initial Capital)</b>",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date",
        yaxis_title="Portfolio Balance ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_equity.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
    fig_equity.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)")
    st.plotly_chart(fig_equity, use_container_width=True)

    # Crash Stress Test
    st.markdown("---")
    st.markdown("#### 🛡️ Crisis Stress Testing (Capital Preservation)")
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        st.markdown("##### 2020 COVID-19 Crash (Feb - Apr 2020)")
        res_cov = run_crisis_stress_test(df_indicators, "2020-02-01", "2020-04-30", "COVID Crash")
        if "error" not in res_cov:
            dd_c = res_cov["drawdown_comparison"]
            st.markdown(
                f"""
                <div class="metric-card-glass">
                    <div>Buy & Hold Drawdown: <b style="color:#F87171;">{dd_c['buy_and_hold_max_dd']}%</b></div>
                    <div>EAATS Full Drawdown: <b style="color:#34D399;">{dd_c['eaats_max_dd']}%</b></div>
                    <div style="color:#38BDF8; font-weight:700; margin-top:6px;">Protection Advantage: +{dd_c['protection_advantage']:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(res_cov["error"])

    with col_st2:
        st.markdown("##### 2022 Bear Market (Jan - Oct 2022)")
        res_br = run_crisis_stress_test(df_indicators, "2022-01-01", "2022-10-31", "2022 Bear Market")
        if "error" not in res_br:
            dd_b = res_br["drawdown_comparison"]
            st.markdown(
                f"""
                <div class="metric-card-glass">
                    <div>Buy & Hold Drawdown: <b style="color:#F87171;">{dd_b['buy_and_hold_max_dd']}%</b></div>
                    <div>EAATS Full Drawdown: <b style="color:#34D399;">{dd_b['eaats_max_dd']}%</b></div>
                    <div style="color:#38BDF8; font-weight:700; margin-top:6px;">Protection Advantage: +{dd_b['protection_advantage']:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(res_br["error"])

    # Closed Trades Ledger
    st.markdown("---")
    st.markdown("#### 📜 Executed Trade Ledger (EAATS Full)")
    trades_full = all_sim_results["eaats_full"]["trades"]
    if trades_full:
        df_tr = pd.DataFrame(trades_full)
        st.dataframe(df_tr, use_container_width=True, hide_index=True)
        csv_data = df_tr.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Trade Ledger CSV", data=csv_data, file_name=f"{active_ticker}_trades.csv")
    else:
        st.info("No positions closed in this test split.")

# ------------------------------------------------------------------------------
# TAB 6: ACADEMIC WHITEPAPER & METHODOLOGY
# ------------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### 📚 Academic Defense & System Whitepaper")
    st.markdown(
        """
        #### 1. Theoretical Motivation & The Emotional Quant Curve
        Financial markets represent dynamic complex systems driven simultaneously by fundamental quantitative ratios
        and collective investor psychology. While conventional quantitative strategies rely upon rigid mathematical
        indicators that break down during structural shifts, **EAATS** incorporates market emotion as an active risk parameter.

        #### 2. Academic Literature Citations
        - **Jiang et al. (2017)**: *A Deep Reinforcement Learning Framework for Financial Portfolio Management.*
        - **Fischer & Krauss (2018)**: *Deep learning with long short-term memory networks for financial market direction forecasting.*
        - **Zhang et al. (2020)**: *Integrating market sentiment and financial time series for stock trend prediction.*
        - **Hutto & Gilbert (2014)**: *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.*

        ---
        #### ⚠️ Educational Research Disclaimer
        **This platform is an academic engineering simulation prototype designed exclusively for educational and scientific research.**
        The system does not account for real-time exchange liquidity, bid-ask spreads, order execution slippage, or regulatory constraints.
        **Nothing in this application constitutes financial or investment advice.**
        """
    )
