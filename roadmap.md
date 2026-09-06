### 1. Market Data Strategy: US Pre-Training vs. Indian Market Target

#### The Transferability Dilemma

Training an algorithmic strategy on US market data to trade Indian Futures & Options (NSE) is **partially valid for feature representation**, but **invalid for execution policy** without domain-specific calibration.

* **What Transfers (Invariant Features):** Statistical properties of price action—such as volatility clustering, momentum decay, mean-reverting regimes, and normalized volume-price relationships—are universal across liquid continuous double-auction markets. Pre-training a deep feature extractor (e.g., Temporal Convolutional Networks or Transformer Encoders) on high-density US data (e.g., S&P 500 / Nasdaq futures) creates robust, general-purpose market state representations.
* **What Does Not Transfer (Market Microstructure & F&O Mechanics):**
* **Cost & Friction Structures:** Indian markets have unique statutory costs (Securities Transaction Tax / STT, GST, Stamp Duty, SEBI turnover fees) that severely impact high-frequency or short-term compounding compared to US fee structures.
* **Derivatives Dynamics:** Options on NSE (e.g., Nifty / Bank Nifty weekly/monthly contracts) exhibit distinct implied volatility surface dynamics, rapid theta decay profiles, and liquidity concentration that differ significantly from US equity options.
* **Session Structure & Gaps:** Indian markets operate on a 6.15-hour trading day with significant overnight gap risks driven by global markets (US / Asia overnight moves).



#### Recommendation

Adopt a **Two-Tier Data Strategy**:

1. **Pre-Training Phase (US Futures Data):** Use multi-year 1-minute OHLCV/Tick data from US indices (NQ, ES) to pre-train state-space encoders on asset-agnostic features (normalized log-returns, relative volume, volatility ratio).
2. **Target Fine-Tuning & Evaluation (NSE Data):** The core RL policy **must** be trained, fine-tuned, and evaluated on authentic 1-minute historical tick/OHLCV data for Indian Index Futures and liquid Options (Nifty / BankNifty). Do not evaluate performance metrics using US data if the execution target is Indian F&O.

---

### 2. Standard System Architecture & Methodology

```
+-----------------------------------------------------------------------------------+
|                               1. DATA PIPELINE                                    |
|   Raw Data -> Stationarity Engine -> Purged Splitting -> Anti-Lookahead Storage   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           2. FEATURE ENGINEERING ENGINE                           |
|      Z-Score Returns, Volatility Scaling, Microstructure Features (Zero-Lookahead)|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        3. HIGH-STRICTNESS ENVIRONMENT                             |
|    Gym API | Slippage Model | Spread Expansion | Statutory Costs (+10% Buffer)   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                          4. MODEL TRAINING & CURRICULUM                           |
|       Risk-Averse RL (PPO/SAC) | Differential Sharpe / Drawdown Penalty Reward    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        5. VALIDATION & TESTING SUITE                              |
|   Combinatorial Purged CV | Walk-Forward Out-of-Sample | Synthetic Stress Testing     |
+-----------------------------------------------------------------------------------+

```

---

### 3. Data Engineering & Anti-Lookahead Architecture

To ensure complete elimination of lookahead bias and data leakage:

#### Point-in-Time Data Pipeline

* **Strict Temporal Indexing:** All feature computations at step $t$ must only use information available up to timestamp $t-\Delta t$, where $\Delta t$ is the minimum execution latency (e.g., 500ms to 1s).
* **Stationarity Transformations:** Do not feed raw prices into the neural network. Transform all continuous variables into stationary series:
* Log-returns: $r_t = \ln(P_t / P_{t-1})$
* Fractional Differentiation: Apply fractional differentiation $d \in (0, 1)$ to preserve memory while achieving stationarity.
* Rolling Z-Scores: Standardize features using expanding window or strictly backward-looking rolling windows:


$$Z_t = \frac{X_t - \mu_{t-k:t}}{\sigma_{t-k:t}}$$



#### Data Splitting Protocol (Eliminating Leakage)

Standard random train-test splits introduce severe temporal leakage in time-series models.

* **Combinatorial Purged Cross-Validation (CPCV):** Split historical data into $N$ blocks. When creating training and testing sets, apply:
1. **Purging:** Remove training samples whose labels overlap in time with testing samples.
2. **Embargoing:** Remove a fixed percentage of training samples immediately following test sets to eliminate autoregressive serial correlation spillover.



---

### 4. Simulation Engine (+10% Strict Friction Calibration)

The environment must penalize agents more aggressively than live market conditions to ensure strategy survival.

#### Friction Parameters vs. Real World

| Parameter | Real World Baseline | Strict Simulation Target (+10% Penalty) |
| --- | --- | --- |
| **Bid-Ask Spread** | Average prevailing spread | $1.10 \times \text{Average Spread}$ |
| **Execution Slippage** | Variable per order size | Fixed base slippage + $1.10 \times \text{Volatile Market Slippage}$ |
| **Statutory Fees & Taxes** | Brokerage + STT + Exchange fees | Full Statutory Fees $\times 1.10$ |
| **Execution Latency** | 100ms - 200ms | Forced 1-bar execution delay or 500ms market lag |
| **Fill Probability** | Immediate fill on market order | Partial fills on limit orders; random 5% order rejection rate |

#### Execution Logic Engine

* **Entry/Exit Execution:** Order placement at step $t$ executes at price $P_{t+1, \text{open}}$ adjusted for spread and slippage penalties.
* **Short Selling & Option Decay:** Include explicit margin requirements, overnight holding penalties, and realistic black-scholes option decay behavior if trading synthetic option options/greeks.

---

### 5. Reward Function Design for Drawdown Control

Standard return-maximization rewards cause agents to take tail risks, leading to severe drawdowns. To enforce the **20x return with $<30\%$ drawdown** constraint at 0x leverage, use risk-adjusted, drawdown-aware reward functions.

#### Mathematical Formulations

1. **Differential Sortino Ratio Reward:**
$$R_t = \frac{r_t - r_f}{\sigma_{\text{downstream}, t} + \epsilon}$$


Where $\sigma_{\text{downstream}}$ penalizes only negative return variance.
2. **Dynamic Drawdown Penalty Integration:**
$$\text{DD}_t = \frac{\max_{0 \le \tau \le t} (V_\tau) - V_t}{\max_{0 \le \tau \le t} (V_\tau)}$$


$$R_{\text{total}, t} = r_t - \lambda \cdot (\text{DD}_t)^2 - \beta \cdot \mathbf{1}_{\{\text{DD}_t > 0.25\}}$$


* $\lambda$: Penalty scaling factor for continuous drawdown.
* $\beta$: Heavy step-penalty triggered when drawdown breaches 25% (providing a safety buffer before the 30% hard limit).



---

### 6. Validation Framework & Success Thresholds

A model is considered valid **only** if it passes all three validation layers without modification of hyper-parameters.

```
+--------------------------------------------------------------------------+
|                     STAGE 1: In-Sample Training                          |
|         Train agent on historical train split (e.g., 2018-2022)          |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                   STAGE 2: Purged Out-Of-Sample Test                     |
|        Evaluate on unseen test split (e.g., 2023-2026) with strict        |
|                    friction settings (+10% penalty)                      |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                  STAGE 3: Synthetic Stress Testing                       |
|   Run 1,000 Monte Carlo Block Bootstraps & Historical Crisis Replays      |
+--------------------------------------------------------------------------+

```

#### Pass Criteria Thresholds (0x Leverage Base Environment)

1. **Net Cumulative Return:** $\ge 20\text{x}$ ($2000\%$) across the full evaluation timeline.
2. **Maximum Peak-to-Trough Drawdown:** $\le 30\%$.
3. **Calmar Ratio:** $\ge 3.0$ ($\text{Annualized Return} / \text{Max Drawdown}$).
4. **Profit Factor:** $\ge 1.8$ ($\text{Gross Profits} / \text{Gross Losses}$).
5. **Monte Carlo Survival Rate:** $> 95\%$ of synthetic bootstrap runs maintain drawdown $< 30\%$.

---

### 7. Implementation Roadmap

1. **Standardize Project Directory:** Structure into clear modules: `data_engine/`, `feature_engineering/`, `simulation_env/`, `models/`, and `evaluation_suite/`.
2. **Data Ingestion & Cleaning:** Ingest authentic 1-minute historical OHLCV data for target instruments (NSE Indices / Futures). Apply forward-fill checks, continuous futures roll adjustments, and stationary transformations.
3. **Build Strict Simulation Environment:** Implement custom OpenAI Gym / Gymnasium environment incorporating statutory fees, bid-ask spreads, and latency delays boosted by +10%.
4. **Baseline Model Training:** Train PPO / SAC agents on standardized stationary state vectors using drawdown-penalized reward functions.
5. **Walk-Forward Out-of-Sample Testing:** Evaluate trained checkpoints across rolling test windows to confirm absence of overfitting and lookahead bias.

---

### Recommended Next Steps

Since you are starting fresh and building this standard suite step-by-step:

1. Should we define the exact mathematical formulation for your **feature state vector** (indicators, stationarity, normalization methods)?
2. Or would you like to structure the **custom Gym/Gymnasium environment specification** for the 10% strict friction simulator first?
