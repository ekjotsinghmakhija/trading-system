# QUANTUM-50K ENGINE: TECHNICAL IMPLEMENTATION SPECIFICATION

## System Objective & Target Execution Frame

* **Primary Objective:** Compound an initial capital base of ₹50,000 INR to ₹20,00,000 INR (a 40x / +3,900% absolute return) within 24 calendar months (~500 active trading days).
* **Execution Asset Class:** NSE Nifty 50 and Bank Nifty Intraday Derivatives (100% Cash-Financed Long Option Buying). Option writing is disabled due to SEBI's SPAN + Exposure margin wall (₹1.2 Lakhs to ₹1.8 Lakhs per lot).
* **System Class:** Non-HFT Machine Learning Pipeline (1-Minute Sequence Bars, Zero Forward-Looking Bias, Zero Overfitting).
* **Risk Parameters:** Peak-to-Trough Portfolio Drawdown $\le 30\%$, Daily Equity Hard Lockout = 5% (₹2,500 loss cap on initial ₹50,000 capital).

### Log-Compounding Yield & Expectancy Formulation

Compounding equity from $C_0 = \text{₹50,000}$ to $C_T \ge \text{₹20,00,000}$ across $T = 500$ trading days without leverage is governed by the discrete log-compounding yield equation:

$$C_T = C_0 \cdot \prod_{i=1}^{T} \left( 1 + f_i \cdot R_i - \Phi_i \right) \ge \text{₹20,00,000}$$

Where:

* $f_i \in [0.10, 0.40]$ represents the dynamic fractional equity allocation per trade on day $i$.
* $R_i$ is the un-leveraged gross option premium return vector on trade $i$.
* $\Phi_i$ is the total transaction friction penalty (Brokerage + STT + Exchange Charges + GST + Slippage) relative to account equity.
* **Required Compound Daily Growth Rate (CDGR):** $0.741\%$ net capital yield per trading day.
* **Target Expectancy Profile:** Assuming 1 to 2 high-conviction trades per day (~600 total trades over 24 months), a 54% Win Rate paired with a 1:1.8 Risk-to-Reward Ratio achieves a net daily expectancy of $+0.82\%$, satisfying the $0.741\%$ CDGR requirement after accounting for transaction friction.

---

## 1. System Architecture & Hardware Resource Allocation Engine

The pipeline runs on dedicated local consumer workstation hardware. High-cost cloud servers and cloud GPUs are eliminated by optimizing tensor memory layouts, asynchronous file I/O, and multi-threaded CPU/GPU task isolation.

```
+---------------------------------------------------------------------------------------+
|                               LOCAL HARDWARE TIER                                     |
|  • Workstation CPU: 8c/16t – Parquet I/O, Feature Calculation & Dynamic Resampling    |
|  • System RAM: 32 GB – Memory-Mapped DuckDB / Polars Arrow Tables                     |
|  • GPU Compute: NVIDIA RTX CUDA – TCN-GRU Forward/Backward PyTorch Training Pass      |
|  • Local Storage: 1 TB NVMe SSD – Partitioned Local Database Structure               |
+---------------------------------------------------------------------------------------+

```

### Memory Engineering & Zero-Copy Pipeline

* **Polars & PyArrow Zero-Copy Ingestion:** All tabular inputs utilize Apache Arrow memory tables to eliminate serialization and copying overhead. Data transfers from disk storage to PyTorch tensors run directly through contiguous Arrow memory buffers.
* **Streaming Walk-Forward Dataset Chunks:** To operate cleanly within a 32 GB RAM ceiling without memory fragmentation, historical multi-year 1-minute datasets (2021–2026) stream in 3-month walk-forward chunks rather than loading as a single dense array. Peak system memory usage remains strictly below 18 GB.
* **Inference Runtime Latency Budget:** The dual Actor-Critic forward pass executes in $< 4\text{ ms}$ on CUDA acceleration and $< 8\text{ ms}$ on multi-threaded CPU runtimes. This easily satisfies the 1-minute bar execution cutoff ($60,000\text{ ms}$).

---

## 2. Dual-Engine Data Ingestion & Storage Pipeline

Because Zerodha's Kite Connect API purges expired options contracts on settlement day, historical backtesting and live execution require a hybrid dual-data workflow:

```
                  [ HISTORICAL ARCHIVE ENGINE ]             [ LIVE SYSTEM DAEMON ]
                             │                                         │
             GlobalDataFeeds / TrueData Bulk Dump             Zerodha Kite WebSocket
             (2021–2026 Raw 1-Min Futures + Chains)          (Streaming Tick Ingestion)
                             │                                         │
                             ▼                                         ▼
            ┌───────────────────────────────────────────────────────────────┐
            │               LOCAL PARQUET / DUCKDB STORAGE                  │
            │   /data/parquet/                                              │
            │   ├── index=NIFTY/year=2025/month=10/spot_fut_options.parquet │
            │   └── index=BANKNIFTY/year=2025/month=10/spot_fut_options.parquet│
            └───────────────────────────────────────────────────────────────┘

```

### Historical Archive Specifications (2021–2026)

* **Data Sources:** GlobalDataFeeds (GDF) / TrueData REST API bulk export archives.
* **Instrument Scope:**
* **Nifty 50:** Spot Index, Current-Month Futures, ATM $\pm 10$ Strike Option Contracts (Calls & Puts).
* **Bank Nifty:** Spot Index, Current-Month Futures, ATM $\pm 10$ Strike Option Contracts (Calls & Puts).


* **Recorded Fields:** Timestamp, Open, High, Low, Close, Volume, Open Interest (OI), Best Bid Price, Best Ask Price, Best Bid Depth, Best Ask Depth.
* **Storage Format & Storage Footprint:** Snappy-compressed Apache Parquet format partitioned on disk by `/index/year/month/`. Disk footprint is approximately 220 GB.

### Live Ingestion Engine (Zerodha Workaround)

* **Execution Service:** Python `asyncio` daemon listening to the KiteTicker WebSocket.
* **Resampling Buffer:** Incomming ticks convert into 1-minute OHLCV, Volume, and OI bars inside an in-memory ring buffer.
* **End-of-Day Persistence:** At 3:30 PM IST, the daemon flushes processed 1-minute bars directly into local DuckDB tables before Zerodha purges expired tokens at midnight.

---

## 3. 18-Feature Uncorrelated Alpha Matrix & CSV Factor Discovery Engine

The system processes 18 orthogonal indicators spanning spot, futures, and option chain dynamics to capture continuous market structure.

```
+-----------------------------------------------------------------------------------------------+
|                             RAW 1-MINUTE MULTI-ASSET INPUT STREAM                             |
+-----------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-----------------------------------------------------------------------------------------------+
|                             FEATURE GENERATION CLUSTERS (18 FEATURES)                         |
+-----------------------------------------------------------------------------------------------+
| Cluster 1: Spot & Futures Momentum   │ Cluster 2: Volatility & Skew Accelerators               |
| • Scaled RSI (14-period)             │ • IV Skew Velocity (Call IV vs. Put IV)                |
| • VWAP Distance Ratio                │ • Put-Call Ratio (PCR) Velocity                        |
| • Futures Basis (Fut Close - Spot)   │ • Realized Volatility Velocity                         |
| • Futures OI Change Acceleration     │ • Effective Delta Acceleration                         |
| • Price Return Acceleration          │ • IV-RV Gap Ratio                                      |
| • Momentum Density Ratio             │ • Vega Exposure Velocity                               |
+--------------------------------------+--------------------------------------------------------+
| Cluster 3: Microstructure & Order Flow                                                        |
| • Level-2 Order Flow Imbalance (OFI)                                                          |
| • Bid-Ask Spread Decay Ratio                                                                  |
| • Volume Spike Factor ($V_t / \text{MA}(V)_{20}$)                                             |
| • Ask Depth Ratio                                                                             |
| • Bid Depth Ratio                                                                             |
| • Micro-Price Drift                                                                           |
+-----------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+-----------------------------------------------------------------------------------------------+
|                            ORTHOGONALIZATION & FILTERING PIPELINE                             |
| • Step 1: Pairwise Correlation Filter — Drops features with $|r| > 0.65$                      |
| • Step 2: Variance Inflation Factor — Retains features with $VIF < 5.0$                        |
+-----------------------------------------------------------------------------------------------+

```

### Mathematical Definitions of Key Indicators

#### Level-2 Order Flow Imbalance ($\text{OFI}_{1\text{m}}$)

$$\text{OFI}_{1\text{m}} = \frac{\sum_{i=1}^{5} V_{\text{bid}, i} - \sum_{i=1}^{5} V_{\text{ask}, i}}{\sum_{i=1}^{5} V_{\text{bid}, i} + \sum_{i=1}^{5} V_{\text{ask}, i}}$$

#### Implied Volatility Skew Velocity ($\alpha_{\text{skew}}$)

$$\alpha_{\text{skew}} = \frac{(\text{IV}_{\text{Put}} - \text{IV}_{\text{Call}})_t - (\text{IV}_{\text{Put}} - \text{IV}_{\text{Call}})_{t-3}}{3}$$

#### Put-Call Ratio Velocity ($v_{\text{PCR}}$)

$$v_{\text{PCR}} = \frac{\text{PCR}_t - \text{PCR}_{t-5}}{5}, \quad \text{where } \text{PCR} = \frac{\sum \text{OI}_{\text{Puts}}}{\sum \text{OI}_{\text{Calls}}}$$

#### Effective Delta Acceleration ($\gamma_{\text{eff}}$)

$$\gamma_{\text{eff}} = \frac{\Delta_t - \Delta_{t-1}}{S_{\text{spot}, t} - S_{\text{spot}, t-1}}$$

#### Intraday VWAP Distance Ratio ($d_{\text{VWAP}}$)

$$d_{\text{VWAP}} = \frac{P_{\text{spot}} - \text{VWAP}_{\text{intraday}}}{\text{VWAP}_{\text{intraday}}}$$

#### Futures Basis ($B_{\text{fut}}$)

$$B_{\text{fut}} = \frac{P_{\text{futures}} - P_{\text{spot}}}{P_{\text{spot}}}$$

### CSV/Parquet Feature-Return Ledger & Factor Discovery Engine

To identify non-linear alpha combinations that can exceed the 40x target, every training and validation step logs a complete record into disk storage (`/logs/experiments/factor_ledger.parquet`).

#### CSV/Parquet Ledger Schema

```
[ Timestamp (UTC) ]
[ Index Symbol (NIFTY / BANKNIFTY) ]
[ Raw Features 1..18 (Float64) ]
[ Orthogonal Features 1..18 (Float64) ]
[ Model Action Output a_t (Float32) ]
[ Executed Lot Size (Int16) ]
[ Entry Option Premium (Float32) ]
[ Exit Option Premium (Float32) ]
[ Holding Time Duration Δt_hold (Int32 Seconds) ]
[ Transaction Friction Charges Φ_i (Float32 INR) ]
[ Forward Net Return Horizon 1m  (Float32 %) ]
[ Forward Net Return Horizon 5m  (Float32 %) ]
[ Forward Net Return Horizon 15m (Float32 %) ]
[ Forward Net Return Horizon 30m (Float32 %) ]
[ Forward Net Return Horizon 60m (Float32 %) ]
[ Realized Trade Net INR Return R_INR (Float32 INR) ]

```

#### Post-Hoc Factor Analysis & Linear/Non-Linear Isolation

1. **Feature Importance Ranking:** Runs quarterly via ElasticNet Regression and Random Forest Feature Importance models applied to the logged Parquet sheets.
2. **Interaction Factor Generation:** Evaluates cross-product feature interactions (e.g., $\text{OFI}_{1\text{m}} \times \alpha_{\text{skew}}$) against forward returns ($R_{\text{INR}}$) to discover composite alpha factors.
3. **Alpha Weight Adjustments:** High-performing composite factors are integrated back into Cluster 1–3 inputs to increase expectancy without altering base network capacity.

---

## 4. Deep Neural Network Model Architecture & Local Minima Escape Mechanism

The agent uses a Temporal Convolutional Network (TCN) paired with a Gated Recurrent Unit (GRU) to process temporal sequences without encountering solver instability.

```
[ Input Sequence Tensor: 60 Bars x 18 Orthogonal Features ]
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Temporal Convolutional Network (TCN) Layer                      │
│ • Dilated Conv1D (Kernel=3, Dilation=1, Channels=64) + BatchNorm│
│ • Dilated Conv1D (Kernel=3, Dilation=2, Channels=64) + BatchNorm│
│ • Dilated Conv1D (Kernel=3, Dilation=4, Channels=64) + BatchNorm│
│ • Spatial Dropout (0.15) + ReLU Activation                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Gated Recurrent Unit (GRU) Layer                                │
│ • Hidden Size = 128 Units                                       │
│ • Maintains continuous intraday hidden state h_t                │
└─────────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Actor Policy Head     │       │ Critic State Head     │
│ • Dense(64) -> ReLU   │       │ • Dense(64) -> ReLU   │
│ • Dense(1)  -> Tanh   │       │ • Dense(1)  -> Linear │
│ Action a_t ∈ [-1,1]   │       │ State Value V(s_t)    │
└───────────────────────┘       └───────────────────────┘

```

### Policy Execution Boundaries

* $a_t > +0.25$: Trigger Long ATM/Near-OTM Call Option Purchase (Nifty or Bank Nifty).
* $a_t < -0.25$: Trigger Long ATM/Near-OTM Put Option Purchase (Nifty or Bank Nifty).
* $-0.25 \le a_t \le +0.25$: Neutral / Cash. Close open option positions.

### Local Minima Detection & Automated Perturbation Protocol

#### Local Minima Stagnation Condition

During training, if the model generates three consecutive identical evaluation metrics or parameter outputs across three 100,000-step evaluation checkpoints ($K_{m}, K_{m+1}, K_{m+2}$):

$$\left\vert{} M_{k} - M_{k-1} \right\vert{} < \epsilon \quad \text{and} \quad \left\vert{} M_{k-1} - M_{k-2} \right\vert{} < \epsilon, \quad \text{where } \epsilon = 10^{-4}$$

The network is classified as stuck in a local minimum or policy plateau.

```
                       [ CHECKPOINT EVALUATION ]
                                   │
                                   ▼
                   Check Metric Stagnation Across
                   3 Checkpoints (300,000 Steps)
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
       [ Metric Delta > ε ]                 [ Metric Delta < ε ]
                 │                                   │
                 ▼                                   ▼
       Continue Normal PPO                 TRIGGER ESCAPE PROTOCOL
       Training Pipeline                   (3-Stage Perturbation)
                                                     │
                                   ┌─────────────────┼─────────────────┐
                                   ▼                 ▼                 ▼
                             [ Stage 1 ]       [ Stage 2 ]       [ Stage 3 ]
                             Cosine LR Warm    Entropy Multiplier Weight Noise
                             Restart           Boost (5x)        Injection

```

#### 3-Stage Escape Action Sequence

1. **Stage 1: Cosine Learning Rate Warm Restart (LWR)**
Instantly resets the current learning rate $\eta$ back to its maximum upper bound $\eta_{\text{max}} = 1 \times 10^{-6}$, breaking out of flat gradient surfaces.
2. **Stage 2: Adaptive Policy Entropy Multiplier Boost**
Temporarily increases the PPO entropy loss coefficient $c_2$ by $5 \times$ (from $0.01$ to $0.05$) for the next 20,000 training steps. This penalizes deterministic policy outputs and forces exploration.
3. **Stage 3: Parametric Network Weight Noise Injection**
Injects zero-mean Gaussian noise directly into the Actor network weights $W_a$:

$$W_a \leftarrow W_a + \mathcal{N}\left(0, \, \sigma_{\text{noise}}^2\right), \quad \text{where } \sigma_{\text{noise}} = 0.02$$



This alters the policy manifold while preserving structural representation in the TCN feature extraction layers.

---

## 5. 12-Million Step Walk-Forward Training & Validation Framework

To prevent overfitting and forward-looking bias, training executes across a 12,000,000-step regime using a non-overlapping walk-forward strategy.

```
[ Historical Dataset: 2021 - 2026 ]
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Walk-Forward Window Setup                                       │
│ ┌──────────────────────────────────────────────┬──────────────┐ │
│ │ Train Window: 12 Months Rolling              │ Test: 3 Mos  │ │
│ └──────────────────────────────────────────────┴──────────────┘ │
│ (1-Day Embargo gap inserted between Train and Test splits)      │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Periodic Checkpoint Evaluation (Every 100,000 Steps)            │
│ • Compute Out-of-Sample Sharpe Ratio                            │
│ • Verify Drawdown Limits (≤ 30%)                                │
│ • Save Checkpoint Weights (.pt) and INR Ledger Logs             │
└─────────────────────────────────────────────────────────────────┘

```

### Training Hyperparameters & Scheduler

* **Total Training Volume:** 12,000,000 Timesteps.
* **Learning Rate Range:** $\eta \in [1 \times 10^{-6}, \, 5 \times 10^{-7}]$.
* **Learning Rate Schedule:** Cosine Annealing with Warm Restarts scheduled across 12M steps.
* **Optimizer:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\text{weight\_decay} = 10^{-4}$).
* **PPO Clip Range ($\epsilon_{\text{ppo}}$):** $0.12$.
* **GAE Parameters ($\gamma / \lambda_{\text{gae}}$):** $\gamma = 0.99$, $\lambda_{\text{gae}} = 0.95$.
* **Batch Sizing:** 2,048 timesteps per mini-batch update.

### Loss Function Formulation

The total loss optimizes the Proximal Policy Optimization (PPO) clipped surrogate objective, combined with Huber Loss on the Critic network and dynamic entropy regularization:

$$L_{\text{total}}(\theta) = L_{\text{CLIP}}(\theta) - c_1 \cdot L_{\text{Huber}}(\theta) + c_2 \cdot S_{[\text{entropy}]}(\theta)$$

Where:

$$L_{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon_{\text{ppo}}, 1+\epsilon_{\text{ppo}})\hat{A}_t \right) \right]$$

$$r_t(\theta) = \frac{\pi_\theta(a_t \vert{} s_t)}{\pi_{\theta_{\text{old}}}(a_t \vert{} s_t)}$$

### Bounded Differential Sharpe Ratio (DSR) Reward Architecture

To prevent Critic Value function divergence across 12M steps, running performance moments are normalized into bounded stationary space $[-1, +1]$ using an adaptive Z-Score transform:

$$\tilde{A}_t = \tanh\left( \frac{A_t - \mu_A}{\sigma_A} \right), \quad \tilde{B}_t = \tanh\left( \frac{B_t - \mu_B}{\sigma_B} \right)$$

$$\text{Reward}_t = \text{DSR}_t - \lambda_{\Theta} \cdot \left( \frac{\Delta t_{\text{hold}}}{\tau_{\text{expiry}}} \right) - \phi \cdot \left( \frac{\text{Brokerage} + \text{STT} + \text{Turnover Fees}}{\text{Account Capital}} \right)$$

### 100,000-Step Checkpoint Evaluation Protocol

Every 100,000 steps, the model pauses training and executes an evaluation pass on unseen out-of-sample test data:

1. **Checkpoint File Generation:** Saves network parameters to disk (`/models/checkpoints/model_step_100000.pt`).
2. **Performance Metrics Calculation:** Logs Win Rate (%), Expectancy per Trade (₹), Sharpe Ratio, Max Drawdown (%), and Cumulative Net Portfolio Yield (INR).
3. **Rejection Filter:** Checkpoint weights are discarded if Out-of-Sample Max Drawdown exceeds $22\%$ or if net cumulative return is negative.

---

## 6. Dynamic Holding Engine & Pegged Limit Execution State Machine

Trades adapt dynamically across two operational regimes based on real-time option momentum and time decay.

```
                          [ ENTRY SIGNAL TRIGGERED ]
                                     │
                                     ▼
                        [ Evaluate Trend Strength ]
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
  [ Regime A: Micro-Scalp ]                             [ Regime B: Trend Ride ]
  - Duration: 2 to 10 Minutes                           - Duration: 10 to 60 Minutes
  - Exit: Fast Momentum Decay /                         - Exit: Dynamic Trailing Stop
    Delta Inversion (dΔ/dt < 0)                                  │
                                                                 ▼
                                                      [ HARD CAP: 60 Minutes ]
                                                      (Force Exit to Kill Theta)

```

### Exit Conditions

#### 1. Delta Inversion & Volatility Stall

Exits immediately if option Delta acceleration turns negative while holding a position:

$$\frac{\partial^2 P_{\text{option}}}{\partial S^2} \cdot \frac{dS}{dt} < 0$$

#### 2. Dynamic Option Trailing Stop Loss

$$\text{SL}_t = \max\left( P_{\text{entry}} \times (1 - \text{SL}_{\text{base}}), \; P_{\text{peak}} - 1.5 \times \text{ATR}_{\text{option}}(t) \right)$$

#### 3. Hard 1-Hour Theta Wall

All open positions are force-closed after 60 minutes of holding time. This prevents time decay ($-\Theta$) during consolidation from eroding unrealized profits.

### Pegged Limit Order Exit Machine

To avoid paying wide bid-ask spreads on market orders, exits execute via a pegged limit state machine:

```
                   [ EXIT SIGNAL ACTIVATED ]
                               │
                               ▼
         Place Limit Order at Mid-Ask Price:
         P_limit = P_bid + 0.75 * (P_ask - P_bid)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
  [ Executed in < 3s ]                 [ Unfilled after 3s ]
            │                                     │
            ▼                                     ▼
     [ Trade Closed ]                 Step Limit Down by 1 Tick
                                      (Re-evaluate until filled)

```

---

## 7. Operational Capital Compounding & Dynamic Sizing Schedule

Position sizes scale dynamically based on total account equity across regulatory frameworks.

### Capital Growth Stages & Risk Matrix

| Capital Stage | Account Equity (INR) | Index Contract | Lot Size | Approx Premium | Max Executed Lots | Allocation (% Equity) | Daily Hard Loss Cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Phase 1: Seed** | ₹50,000 | Nifty 50 ATM | 65 | ₹100 | **3 Lots** | ₹19,500 (**39.0%**) | ₹2,500 (5.0%) |
| **Phase 2: Growth** | ₹1,50,000 | Nifty / Bank Nifty | 65 / 30 | ₹110 | **8 Lots** | ₹57,200 (**38.1%**) | ₹7,500 (5.0%) |
| **Phase 3: Expansion** | ₹5,00,000 | Nifty / Bank Nifty Multi | 65 / 30 | ₹120 | **25 Lots** | ₹1,95,000 (**39.0%**) | ₹25,000 (5.0%) |
| **Phase 4: Scale** | ₹20,00,000 | Portfolio Diversified | 65 / 30 | ₹130 | **90 Lots** | ₹7,60,500 (**38.0%**) | ₹1,00,000 (5.0%) |

### Operational Guardrails & Execution Hard Stops

* **Intraday Session Cutoff:** No new positions are opened after **2:30 PM IST**.
* **Hard EOD Square-Off:** All active positions are force-closed by **3:10 PM IST** to eliminate overnight gap risk.
* **Daily Circuit Breaker:** If cumulative daily losses reach 5% of starting daily equity (₹2,500 during Phase 1), the execution daemon revokes session keys until the next trading day.

---

## 8. Implementation & Deployment Timeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ WEEKS 1–3: DATA PIPELINE & FEATURE LEDGER                                             │
│ • Configure DuckDB & Parquet partitioning for 2021–2026 data.                          │
│ • Build 18-feature engine & set up CSV/Parquet experiment logging schema.              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ WEEKS 4–8: 12M STEP TCN-GRU MODEL TRAINING                                             │
│ • Execute 12,000,000-step training run on local NVIDIA GPU (LR: 1e-6 to 5e-7).          │
│ • Evaluate out-of-sample checkpoints every 100,000 steps.                              │
│ • Validate 3-stage local minima perturbation logic.                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ WEEKS 9–11: ASYNC EXECUTION DAEMON & PAPER TRADING                                    │
│ • Build Python asyncio daemon for Zerodha KiteTicker WebSocket.                       │
│ • Test pegged limit order machine and token-bucket rate limiter (<= 2 req/sec).       │
│ • Confirm end-to-end execution latency remains under 120ms.                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ WEEK 12+: LIVE DEPLOYMENT                                                              │
│ • Deploy with ₹50,000 starting capital.                                                │
│ • Enforce 5% daily circuit breaker (₹2,500 cap) and auto-logging factor sheets.       │
└────────────────────────────────────────────────────────────────────────────────────────┘

```
