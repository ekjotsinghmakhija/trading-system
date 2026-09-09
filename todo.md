## Phase 1: Storage Architecture & Zero-Copy Memory Pipeline (Weeks 1–2)

### Hardware & Environment Initialization

* [ ] **Configure Local Memory Architecture:** Set up Apache Arrow memory tables integrated with Polars to enable zero-copy tensor casting into PyTorch memory blocks without array re-allocation.
* [ ] **CUDA Acceleration Pipeline Verification:** Verify PyTorch CUDA drivers on the workstation GPU to confirm tensor forward/backward inference routines execute in $< 4\text{ ms}$ for a 60-bar sequence window.

### Partitioned Storage Setup

* [ ] **Directory Hierarchy Creation:** Build local disk storage structure under `/data/parquet/` partitioned strictly by `index={NIFTY,BANKNIFTY}/year={YYYY}/month={MM}/`.
* [ ] **DuckDB Schema Definition:** Create a local persistent DuckDB database configured with Snappy compression for query-based historical sampling and end-of-day WebSocket persistence.

---

## Phase 2: Dual Data Ingestion Pipeline (Weeks 1–3)

### Historical Archive Engine (2021–2026 Data)

* [ ] **Bulk Ingestion Adapter:** Build script to ingest historical 1-minute CSV/binary dumps from GlobalDataFeeds/TrueData covering Spot, Current-Month Futures, and ATM $\pm 10$ Strike Calls/Puts.
* [ ] **Data Cleaning & Timestamp Alignment:** Enforce UTC timestamp normalization, fill 1-minute sequence gaps using zero-volume forward fills, and strip non-trading market hours (retain 09:15 to 15:30 IST).
* [ ] **Parquet Partitioning Utility:** Compress raw ticks into Snappy-compressed Parquet archives organized in the `/data/parquet/` directory layout.

### Live Ingestion Async Daemon

* [ ] **Asyncio KiteTicker Client:** Implement a persistent Python `asyncio` WebSocket daemon listening to Zerodha KiteTicker feed for live Spot, Futures, and Option Depth updates.
* [ ] **In-Memory Ring Buffer Resampler:** Build a tick resampler that processes incoming WebSocket ticks into 1-minute OHLCV, Volume, Open Interest (OI), and Level-2 order book depth bars.
* [ ] **End-of-Day Persistence Task:** Schedule an automated daily job at 3:30 PM IST to flush in-memory intraday bars directly to the DuckDB store before contract tokens expire at midnight.

---

## Phase 3: 18-Feature Alpha Matrix & Factor Discovery Ledger (Weeks 2–3)

### Cluster 1: Spot & Futures Momentum Indicators

* [ ] **RSI Calculation:** Implement 14-period RSI scaled to range $[-1, 1]$.
* [ ] **VWAP Distance Ratio ($d_{\text{VWAP}}$):** Compute $d_{\text{VWAP}} = \frac{P_{\text{spot}} - \text{VWAP}_{\text{intraday}}}{\text{VWAP}_{\text{intraday}}}$.
* [ ] **Futures Basis ($B_{\text{fut}}$):** Compute $B_{\text{fut}} = \frac{P_{\text{futures}} - P_{\text{spot}}}{P_{\text{spot}}}$.
* [ ] **Futures OI Change Acceleration:** Calculate second-derivative of 1-minute Futures Open Interest changes.
* [ ] **Price Return Acceleration & Momentum Density:** Implement 1-minute and 3-minute log return velocity and volume-weighted price impulse.

### Cluster 2: Volatility & Skew Accelerators

* [ ] **IV Skew Velocity ($\alpha_{\text{skew}}$):** Implement 3-bar delta of Put IV vs. Call IV:

$$\alpha_{\text{skew}} = \frac{(\text{IV}_{\text{Put}} - \text{IV}_{\text{Call}})_t - (\text{IV}_{\text{Put}} - \text{IV}_{\text{Call}})_{t-3}}{3}$$


* [ ] **PCR Velocity ($v_{\text{PCR}}$):** Compute 5-bar derivative of total Put/Call Open Interest ratio.
* [ ] **Effective Delta Acceleration ($\gamma_{\text{eff}}$):** Implement discrete option Delta acceleration relative to underlying spot movement:

$$\gamma_{\text{eff}} = \frac{\Delta_t - \Delta_{t-1}}{S_{\text{spot}, t} - S_{\text{spot}, t-1}}$$


* [ ] **Realized Volatility Velocity, IV-RV Gap Ratio, and Vega Velocity:** Implement calculations using 15-minute rolling annualized volatility windows.

### Cluster 3: Microstructure & Level-2 Order Flow

* [ ] **Level-2 Order Flow Imbalance ($\text{OFI}_{1\text{m}}$):** Implement 5-depth bid-ask top-of-book imbalance:

$$\text{OFI}_{1\text{m}} = \frac{\sum_{i=1}^{5} V_{\text{bid}, i} - \sum_{i=1}^{5} V_{\text{ask}, i}}{\sum_{i=1}^{5} V_{\text{bid}, i} + \sum_{i=1}^{5} V_{\text{ask}, i}}$$


* [ ] **Microstructure Dynamics:** Implement Bid-Ask Spread Decay Ratio, Volume Spike Factor ($V_t / \text{MA}(V)_{20}$), Ask/Bid Depth Ratios, and Micro-Price Drift.

### Feature Orthogonalization Pipeline

* [ ] **Pairwise Correlation Filter:** Drop features with absolute correlation coefficient $\vert{}r\vert{} > 0.65$.
* [ ] **Variance Inflation Factor (VIF) Filter:** Compute dynamic VIF scores and retain features with $VIF < 5.0$.

### Factor Discovery Experiment Ledger

* [ ] **Parquet Ledger Writer:** Create a logging module pointing to `/logs/experiments/factor_ledger.parquet` to capture 18 features, model actions ($a_t$), friction penalties ($\Phi_i$), and forward returns across 1m, 5m, 15m, 30m, and 60m horizons.
* [ ] **Post-Hoc Feature Analysis Script:** Build a quarterly pipeline using ElasticNet and Random Forest feature importance to identify emergent multi-feature interactions.

---

## Phase 4: Model Architecture & Local Minima Escape Engine (Weeks 4–5)

### Network Architecture Construction

* [ ] **Temporal Convolutional Network (TCN):** Build a 3-layer dilated Conv1D network (Kernel Size = 3, Dilations = [1, 2, 4], Channels = 64) with BatchNorm, Spatial Dropout (0.15), and ReLU activations.
* [ ] **Gated Recurrent Unit (GRU):** Implement a GRU layer with 128 hidden units to maintain state $h_t$ across 60-bar receptive sequences.
* [ ] **Dual Actor-Critic Heads:**
* **Actor:** Dense(64) $\to$ ReLU $\to$ Dense(1) $\to$ `tanh` (outputs action $a_t \in [-1, +1]$).
* **Critic:** Dense(64) $\to$ ReLU $\to$ Dense(1) $\to$ Linear (outputs state value $V(s_t)$).



### Policy Action Threshold Logic

* [ ] Set $a_t > +0.25 \implies$ Buy ATM/Near-OTM Call Options.
* [ ] Set $a_t < -0.25 \implies$ Buy ATM/Near-OTM Put Options.
* [ ] Set $-0.25 \le a_t \le +0.25 \implies$ Flat / Cash (Close active option positions).

### Local Minima Detection & Automated Perturbation

* [ ] **Stagnation Monitoring Metric:** Monitor out-of-sample Sharpe and net metrics across 3 consecutive 100,000-step evaluation checkpoints ($K_m, K_{m+1}, K_{m+2}$). Trigger escape sequence if parameter delta falls below $\epsilon = 10^{-4}$.
* [ ] **Stage 1 Escape Implementation:** Execute Cosine Learning Rate Warm Restart, resetting learning rate $\eta$ back to $\eta_{\text{max}} = 1 \times 10^{-6}$.
* [ ] **Stage 2 Escape Implementation:** Increase PPO entropy loss coefficient $c_2$ by $5 \times$ (from $0.01$ to $0.05$) for 20,000 steps.
* [ ] **Stage 3 Escape Implementation:** Inject Gaussian noise into Actor parameters:

$$W_a \leftarrow W_a + \mathcal{N}(0, \, 0.02^2)$$



---

## Phase 5: 12M Step Walk-Forward Training Engine (Weeks 6–8)

### Walk-Forward Split Routine

* [ ] **Data Splitter:** Set up a rolling 12-month training window paired with a 3-month out-of-sample test window. Insert a mandatory 1-day embargo gap between train/test splits to eliminate leakage.
* [ ] **Streaming Data Loader:** Stream sequence chunks in 3-month increments to ensure system RAM remains below 18 GB during training.

### Optimizer & Loss Function Setups

* [ ] **AdamW Configuration:** Initialize AdamW with $\beta_1 = 0.9, \beta_2 = 0.999$, and weight decay $= 10^{-4}$. Set cosine annealing schedule across 12,000,000 timesteps ($\eta \in [1 \times 10^{-6}, 5 \times 10^{-7}]$).
* [ ] **PPO Loss Engine:** Construct PPO clipped surrogate advantage loss ($\epsilon_{\text{ppo}} = 0.12$, GAE $\gamma = 0.99, \lambda_{\text{gae}} = 0.95$, mini-batch size = 2,048) combined with Huber Loss on the Critic network.
* [ ] **Bounded Differential Sharpe Ratio (DSR) Reward Function:** Implement DSR reward function applying Z-score normalization and dynamic `tanh` transformation to keep running moments within $[-1, +1]$. Factor in theta time-decay penalty and transaction friction charges $\Phi_i$.

### Checkpoint Manager & Rejection Rules

* [ ] **100k Checkpoint Evaluator:** Schedule evaluation pass every 100,000 steps. Save weights to `/models/checkpoints/model_step_100000.pt`.
* [ ] **Automated Model Discard Rule:** Reject checkpoint weights if out-of-sample max drawdown exceeds $22\%$ or net portfolio yield is negative.

---

## Phase 6: Dynamic Exit Engine & Execution State Machine (Weeks 9–10)

### Exit Trigger Conditions

* [ ] **Delta Inversion Detector:** Implement exit trigger firing when option Delta acceleration turns negative:

$$\frac{\partial^2 P_{\text{option}}}{\partial S^2} \cdot \frac{dS}{dt} < 0$$


* [ ] **Dynamic Option Trailing Stop Loss:** Calculate real-time trailing stop:

$$\text{SL}_t = \max\left(P_{\text{entry}} \times (1 - \text{SL}_{\text{base}}), \; P_{\text{peak}} - 1.5 \times \text{ATR}_{\text{option}}(t)\right)$$


* [ ] **Hard 1-Hour Theta Wall:** Implement a 60-minute holding timer forcing immediate order exit to eliminate asymptotic time decay ($-\Theta$).

### Pegged Limit Order State Machine

* [ ] **Mid-Ask Peg Order Builder:** Configure exit order placement at mid-ask price:

$$P_{\text{limit}} = P_{\text{bid}} + 0.75 \times (P_{\text{ask}} - P_{\text{bid}})$$


* [ ] **Step-Down Retry Machine:** Monitor limit order fill status every 3 seconds. If unfilled, step limit price down by 1 tick until execution completes.

---

## Phase 7: Sizing Schedule, Risk Guardrails & Integration (Weeks 10–11)

### Dynamic Capital Compounding Schedule

* [ ] **Phase 1 Allocation Rules (Seed: ₹50,000):** Restrict trading to Nifty 50 ATM Options, maximum **3 Lots** (Lot Size = 65), allocation capped at ₹19,500 (39% of capital).
* [ ] **Phase 2–4 Scaling Mechanics:** Program automatic lot size adjustments as capital grows (Phase 2 @ ₹1,50,000 $\to 8$ lots; Phase 3 @ ₹5,00,000 $\to 25$ lots; Phase 4 @ ₹20,00,000 $\to 90$ lots).

### System Risk Cutoffs & Lockouts

* [ ] **Daily Equity Circuit Breaker:** Implement an intraday loss tracking circuit breaker that revokes broker session keys if daily loss hits 5% (₹2,500 on Phase 1 capital).
* [ ] **Session Cutoff Routines:** Halt new entries after 2:30 PM IST. Initiate EOD position square-off at 3:10 PM IST.

### API Rate Limiter & Latency Benchmarking

* [ ] **Token-Bucket Rate Limiter:** Build a broker request rate limiter ensuring order actions remain strictly below Zerodha API limits ($\le 2$ requests/sec).
* [ ] **Latency Audit:** Verify total pipeline round-trip latency (data ingestion $\to$ model inference $\to$ order routing) remains under $120\text{ ms}$.

---

## Phase 8: Live Deployment Checklist (Week 12+)

* [ ] Run 5 consecutive days of paper trading via Kite Interactive Sandbox to verify state machine reliability under market conditions.
* [ ] Confirm automatic Parquet factor ledger generation and verify logs under `/logs/experiments/factor_ledger.parquet`.
* [ ] Verify daily hard circuit breaker lockout mechanisms using simulated account drawdown triggers.
* [ ] Deploy system live with initial seed capital of ₹50,000 INR.
