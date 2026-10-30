# EAATS Final-Year B.Tech Viva & Defense Preparation Guide

**Project Title:** EAATS - Emotion-Aware Adaptive Trading System  
**Full Academic Title:** Real-Time Stock Market Prediction & Algorithmic Trading Simulation  
**Domain:** Artificial Intelligence in Computational Finance  
**Presented By:**  
- Puligadda Charishma  
- Bommidi Sree Pranathis  
- Shaik Mansoor Saleem  
- Achanti Rupeshwar  

---

## 1. Executive Summary & Problem Addressed

Traditional algorithmic trading models suffer from three structural drawbacks:
1. **Linear & Static Assumptions:** Fixed indicators (such as SMA-20 or static RSI thresholds) fail when market regimes shift.
2. **Emotional Blindness:** Traditional quant engines ignore human psychological drivers—panic selling, speculative greed, and euphoria—which often dominate during market crises, earnings releases, and geopolitical shocks.
3. **Rigid Risk Sizing:** Standard systems use fixed position sizing regardless of whether the market mood is calm or chaotic.

**EAATS Solution:**  
An ensemble algorithmic trading system that fuses:
- **Deep Learning Forecasting** (Stacked LSTM and Transformer Encoders),
- **Multi-Source Sentiment Analysis** (Google News RSS, StockTwits, Reddit, and YouTube Transcripts scored via VADER and FinBERT),
- **Quantitative Technical Analysis** (Wilder's RSI, EMA/SMA crossovers, MACD),
- **Adaptive Emotion Risk Engine** that modulates position size (15% to 75%) and stop-loss limits (2% to 4%) dynamically,
- **Confidence Threshold Filtering** that suppresses false signals, and
- **Event-Driven Walk-Forward Backtester** executing at Next-Day Open (strictly eliminating look-ahead bias).

---

## 2. Technology Justification Matrix

| Technology | Selection Rationale | Alternative Rejected | Why the Alternative Was Rejected |
| :--- | :--- | :--- | :--- |
| **Stacked LSTM** | Specialised recurrent architecture capable of remembering sequential non-linear dependencies over 60-day windows. | ARIMA / Linear Regression | ARIMA assumes linear stationary dynamics and cannot capture complex financial regime shifts. |
| **Transformer Encoder** | Self-Attention mechanism models relationships across all 60 days in parallel without vanishing gradients. | Vanilla RNN | RNN suffers from severe vanishing/exploding gradients over multi-week time horizons. |
| **VADER Sentiment** | Lexicon- and rule-based sentiment tuned specifically for financial slang, capitalization, and short social posts (instant CPU execution). | TextBlob | TextBlob lacks domain lexicon for financial jargon (e.g., "crushed earnings", "bearish", "breakout"). |
| **FinBERT (ProsusAI)** | Pre-trained on 10,000+ corporate financial news reports; understands nuanced financial context ($P_{pos} - P_{neg}$). | Generic BERT / RoBERTa | Generic language models treat financial words like "liability" or "debt" with negative bias even in neutral filings. |
| **Wilder's RSI(14)** | Uses Welles Wilder's modified moving average; industry standard for momentum overbought/oversold boundaries. | Standard Moving Average RSI | Simple moving average RSI is too volatile and prone to whipsaw false breakouts. |
| **Tabular Q-Learning** | Model-free reinforcement learning that learns optimal action policies (Buy/Sell/Hold) through state exploration. | Deep Q-Network (DQN) | Deep Q-Networks on daily time-series suffer from catastrophic forgetting and non-convergence without massive data. |
| **Streamlit + Plotly** | Interactive web dashboard with zoom, hover, and responsive UI components written cleanly in Python. | Flask / Django + Matplotlib | Flask requires heavy HTML/JS boilerplate; Matplotlib only produces static non-interactive raster images. |

---

## 3. End-to-End Data Flow

```
Step 1: Ingestion (data_fetcher.py)
   ├── Fetch 2-5 years of OHLCV daily bars from Yahoo Finance / CSV
   └── Strip timezones, drop missing values, enforce chronological ordering

Step 2: Feature Engineering (preprocessor.py)
   ├── Compute SMA(20), EMA(12), EMA(26), MACD(12, 26, 9), Wilder's RSI(14), and % Change
   ├── Split data chronologically into 80% Train and 20% Test (NO SHUFFLING)
   ├── Fit MinMaxScaler(0, 1) strictly on 80% Train split (NO DATA LEAKAGE)
   └── Construct 60-day sliding window sequences: X=(N, 60, features), y=(N, 1)

Step 3: Deep Learning Inference (lstm_model.py & transformer_model.py)
   ├── Stacked LSTM: Input(60, f) -> LSTM(128) -> Drop(0.2) -> LSTM(64) -> Drop(0.2) -> Dense(32) -> Dense(1)
   └── Inverse-transform scaled predictions to real dollar values ($)

Step 4: Emotion Engine (src/sentiment/)
   ├── Scrape Google News RSS (FinBERT), Reddit/StockTwits (VADER), and YouTube Transcripts (FinBERT)
   ├── Dynamic weight re-normalization: News 0.40, Social 0.30, Forums 0.20, Video 0.10
   ├── Compute composite score [-1, +1] and emotional volatility (std dev)
   └── Map to Emotion State: PANIC, FEAR, NEUTRAL, CONFIDENCE, GREED

Step 5: Weighted Voting & Signal Suppression (trading_strategy.py)
   ├── Signal 1 (LSTM, 40%): Predicted move >= +0.5% (+1), <= -0.5% (-1), else 0
   ├── Signal 2 (Sentiment, 30%): Score > +0.05 (+1), < -0.05 (-1), else 0
   ├── Signal 3 (Technicals, 30%): EMA>SMA (+0.5), Price>SMA (+0.25), RSI (+0.5/-0.5), MACD (+0.25/-0.25)
   ├── Composite Score = 0.4*LSTM + 0.3*Sent + 0.3*Tech
   ├── Decision Boundary: > +0.15 BUY, < -0.15 SELL, else HOLD
   └── Suppression: If Confidence < Threshold (30%), downgrade to HOLD ("Low Confidence")

Step 6: Adaptive Risk Management (adaptive_risk.py)
   ├── PANIC: Position 25% (15% on high vol), Stop-Loss 2%, Block BUYs on extreme vol
   ├── FEAR: Position 25%, Stop-Loss 2%
   ├── NEUTRAL: Position 50%, Stop-Loss 3%
   ├── CONFIDENCE: Position 75%, Stop-Loss 4%
   └── GREED: Position 50% (contrarian cap), Stop-Loss 2.5%

Step 7: Backtesting & Execution (backtester.py)
   └── Execute signal on Next-Day Open price at 0.1% transaction fee (ZERO LOOK-AHEAD BIAS)
```

---

## 4. Mapping to Project Presentation Slides

| PPT Slide Topic | Theoretical Concept | Concrete Implementation in Codebase |
| :--- | :--- | :--- |
| **Slide 1: Domain Significance** | Fast-changing financial data; limitations of human traders. | `app.py` real-time data ingestion and Plotly interactive dashboard. |
| **Slide 2: Problem Statement** | Markets driven by emotions; fixed indicators fail; inconsistent human judgment. | `src/sentiment/emotion_engine.py` and `src/adaptive_risk.py`. |
| **Slide 3: Proposed Solution** | Multi-source emotion detection + AI price prediction + adaptive risk engine. | The ensemble in `src/trading_strategy.py` and full pipeline. |
| **Slide 4: Literature Review (Jiang et al. 2017)** | Deep Reinforcement Learning agents for autonomous financial trading. | `src/rl_agent.py` Tabular Q-Learning agent with action space (Buy/Hold/Sell). |
| **Slide 5: Literature Review (Fischer & Krauss 2018)** | LSTM neural networks for directional stock market forecasting. | `src/lstm_model.py` 2-layer stacked LSTM architecture with EarlyStopping. |
| **Slide 6: Literature Review (Zhang et al. 2020)** | Integrating public sentiment (Twitter/news) with historical price data. | `src/sentiment/social_source.py` and `news_source.py`. |
| **Slide 7: Emotional Quant Curve** | Market cycles shift from euphoria/greed at the top to panic at the bottom. | Mapped in `adaptive_risk.py`: Contrarian cap (50%) at GREED; capital preservation at PANIC. |
| **Slide 8: Models & Architecture** | LSTM, Transformer, VADER, FinBERT, Rule-based & RL. | All 5 model families implemented with modular decoupling. |
| **Slide 9: Conclusion & Results** | Emotion-aware trading enhances capital protection during crashes. | Verified by 2022 Bear Market stress test: EAATS reduced drawdown by **+22.39%**. |

---

## 5. 15 Likely Examiner Viva Questions & Model Answers

### Q1: What is the primary novelty of EAATS over standard algorithmic trading bots?
**Answer:** Traditional bots rely solely on price-derived mathematical indicators (like moving average crossovers). EAATS integrates **human market psychology** as an active risk parameter. It scrapes news, social media, and video commentary, quantifies emotional states (PANIC, FEAR, NEUTRAL, CONFIDENCE, GREED), and dynamically shrinks or expands capital allocation and stop-loss limits.

### Q2: Why did you predict next-day price rather than multi-step (e.g., 30-day) forecasting?
**Answer:** In financial time-series forecasting, multi-step recursive forecasting compounds prediction errors exponentially—an error on Day 1 skews Day 2, and by Day 5 the trajectory is statistically uncorrelated with reality. EAATS focuses on high-precision next-day prediction and offers a clearly labeled experimental 5-day horizon for trend estimation only.

### Q3: How did you ensure there is no data leakage in your preprocessing?
**Answer:** We implemented two strict controls:
1. **Chronological Splitting:** We used a non-shuffled 80% train / 20% test split. Shuffling time-series leaks future information into past training instances.
2. **Train-Only Scaler Fitting:** `MinMaxScaler` is fitted *strictly* on the first 80% training observations (`scaler.fit(X_train)`). The test split is transformed using the pre-fitted parameters without ever exposing test prices to the scaler.

### Q4: How do you prevent look-ahead bias during backtesting?
**Answer:** In `src/backtester.py`, all technical indicators, LSTM predictions, and sentiment scores are evaluated using data available at the **Close of day $t$**. The resulting trade order is executed at the **Open price of day $t+1$**. This mirrors real-world trading, where an algorithm computing signals after market close cannot trade at past closing prices.

### Q5: What is the Naive Baseline, and why did it achieve a lower RMSE than the LSTM?
**Answer:** The Naive Baseline predicts tomorrow's price will equal today's price ($y_{t+1} = y_t$). Because stock prices are near martingales (random walks with drift), day-to-day absolute price differences are very small (\$4.49 on AAPL). However, the Naive Baseline has **0.00% Directional Accuracy**—it cannot tell you whether to Buy or Sell. The LSTM predicts the directional slope ($50.41\%$ accuracy), which is what makes algorithmic trading possible.

### Q6: Why did you use both VADER and FinBERT instead of just one?
**Answer:** They serve complementary roles:
- **FinBERT** is a heavy Transformer pre-trained on corporate financial filings and Bloomberg/Reuters headlines. It understands nuanced formal syntax.
- **VADER** is a rule-based lexicon optimized for informal, slang-heavy social media text (Reddit, StockTwits) with punctuation emphasis and emojis ("TSLA to the moon! 🚀🚀"). Running FinBERT on thousands of short tweets would be slow and computationally wasteful.

### Q7: What happens to your system if internet connectivity or the Reddit API fails?
**Answer:** We engineered **graceful degradation** across all modules. If Reddit or YouTube APIs fail, the emotion engine automatically re-normalizes the weights of the remaining active sources so they sum to 1.0. If all live text sources are unreachable, the system defaults the emotion score to Neutral (0.0), logs a visible UI alert, and generates decisions safely using the LSTM and technical indicators.

### Q8: What is the "Confidence Threshold Filter", and why is it important?
**Answer:** In real trading, many signals are weak, noisy, or conflicting (e.g., LSTM says Buy, but Technicals say Sell). The confidence score is computed as $\min(|FinalScore| \times 100, 100)\%$. If confidence is below our configurable threshold (default 30%), the trade is suppressed and downgraded to **HOLD**. This prevents churn and eliminated unprofitable chop in our backtests.

### Q9: Explain the contrarian risk rule for the "GREED" emotion state.
**Answer:** According to the "Emotional Quant Curve", when market retail sentiment reaches extreme euphoria/greed, assets are typically overbought and vulnerable to sudden institutional profit-taking. Rather than expanding exposure at the top, our adaptive risk engine caps position sizing at 50% and tightens the trailing stop-loss to 2.5%, preventing the system from buying market peaks.

### Q10: How does your Q-Learning agent state space represent the market?
**Answer:** To avoid the curse of dimensionality, continuous variables are discretized into a 4-dimensional state tuple:
$$\text{State} = (S_{score}, S_{rsi}, S_{emotion}, S_{position})$$
where $S_{score} \in [0..4]$ (5 buckets), $S_{rsi} \in [0..2]$ (3 buckets), $S_{emotion} \in [0..4]$ (5 buckets), and $S_{position} \in [0, 1]$ (flat or long), yielding 150 compact discrete states.

### Q11: What was the reward function used in the RL agent?
**Answer:** The reward function optimizes portfolio growth while penalizing trading friction and capital drawdowns:
$$\text{Reward}_t = \Delta \text{PortfolioValue}_t - \text{Fee} - \lambda \times \text{Drawdown}_t$$
where $\lambda = 0.5$ penalizes holding positions through steep paper drawdowns.

### Q12: Why did Buy & Hold achieve a higher total ROI than EAATS during the 2021–2026 bull test horizon?
**Answer:** In an uninterrupted bull market, any risk-managed strategy that holds 25%–50% cash will lag 100% fully invested buy-and-hold on absolute return. However, EAATS achieved a **higher annualized Sharpe ratio (1.456 vs 1.223)** and a **much lower maximum drawdown (5.21% vs 13.80%)**. In volatile or bear markets (like 2022), EAATS outperformed Buy & Hold by **+22.39%**.

### Q13: How did you calculate Wilder's RSI?
**Answer:** Instead of simple moving averages, Welles Wilder's formula uses an exponential smoothing factor of $\alpha = \frac{1}{\text{period}}$:
$$\text{AvgGain}_t = \frac{\text{AvgGain}_{t-1} \times 13 + \text{Gain}_t}{14}$$
This gives greater weight to recent momentum while preserving long-term smoothed price memory.

### Q14: What is the "Historical Sentiment Proxy" used in backtesting?
**Answer:** News RSS and Reddit search APIs only return current/recent headlines; historical social text from 2020 is inaccessible without expensive enterprise historical APIs. Rather than fabricating fake historical headlines, we compute a mathematically grounded fear/greed proxy based on rolling 14-day momentum normalized by rolling volatility ($\tanh(\text{Momentum} / \text{Vol})$), and clearly disclose this proxy in the user interface.

### Q15: If an examiner asks whether this system can be deployed directly with real capital tomorrow, what would you say?
**Answer:** "No, this is an academic research simulation. To deploy with live capital, we would need to integrate exchange execution APIs (like Interactive Brokers or Zerodha), model order book market impact and bid-ask slippage, handle trade execution latency, and incorporate regulatory compliance checks."
