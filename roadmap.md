Here is the visual flowchart of the system architecture, followed by the point-wise breakdown of the project status and roadmap.

### System Architecture Flowchart

```text
+-----------------------------------------------------------------------+
|                      1. DATA ACQUISITION & PIPELINE                   |
| Broker API Extraction -> Contract Stitching -> Stationarity Engine    |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                      2. FEATURE ENGINEERING ENGINE                    |
| Rolling Z-Scores -> Volatility Scaling -> Fractional Differencing     |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                      3. HIGH-STRICTNESS SIMULATION                    |
| Gym API Engine -> +10% Friction Penalty (Spread, Slippage, Fees)      |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                         4. POLICY TRAINING                            |
|    PPO/SAC RL Agent -> Drawdown-Penalized Reward Optimization         |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                        5. VALIDATION SUITE                            |
| Combinatorial Purged CV -> Verify Target (20x Return, <30% Drawdown)  |
+-----------------------------------------------------------------------+

```

---

### 1. STATUS: COMPLETED WORK

* The standard system architecture and pipeline methodology are fully defined.


* Mathematical formulations for zero-lookahead feature engineering are established.


* These formulations include stationarity transformations and fractional differentiation.


* The high-strictness simulation environment (Gym API with +10% friction) is fully structured.


* Dynamic drawdown-penalized reward functions are mathematically formulated.


* Anti-leakage validation protocols utilizing Combinatorial Purged Cross-Validation (CPCV) are set.



### 2. STATUS: PENDING WORK

* Authentic data acquisition via broker APIs (Zerodha or Angel Broking) must be executed.


* The extraction loop must be run to pull historical data.


* The rollover rule must be applied to stitch continuous futures contracts.


* Price adjustments must be calculated to finalize the `market_data.csv` file.


* Baseline model training must be executed.


* Walk-forward out-of-sample testing must be performed to verify the strategy against target metrics.



### 3. ROADMAP: MARKET DATA STRATEGY

* The system will focus exclusively on Indian Index Futures and liquid Options (Nifty 50, Bank Nifty).


* Individual stock derivatives are excluded due to massive bid-ask spreads.


* Stock derivatives are also excluded because SEBI physical settlement rules require massive margin buffers that cripple capital efficiency.


* US 1-minute futures data (NQ, ES) will be used to pre-train state-space encoders on asset-agnostic features.


* The core execution policy must be fine-tuned and evaluated strictly on Indian F&O data.



### 4. ROADMAP: DATA ACQUISITION & STITCHING

* A one-month API extraction run via a retail broker will be executed to obtain institutional-grade data. *(Note: As of recent updates, Zerodha's Kite Connect API now includes historical data access with its base ₹500/month subscription, dropping the old ₹2000 add-on fee).*


* The broker's instrument list must be queried to map Nifty 50 Futures contracts from 2018 through August 2026.


* 1-minute OHLCV data will be requested one contract at a time.


* 1-second pauses will be incorporated to respect API rate limits.


* The expiring contract will be dropped on the Tuesday before expiry to prevent artificial price gaps.


* The system will roll over to the next month's contract and calculate the precise price difference at the time of the switch.


* All preceding historical data will be shifted by that difference to create a smooth, mathematically continuous price line.



### 5. ROADMAP: EVALUATION & TARGET METRICS

* The agent is trained using a Differential Sortino Ratio Reward.


* The agent is also trained with a Dynamic Drawdown Penalty Integration.


* A trained model must pass In-Sample Training, Purged Out-Of-Sample Testing, and Synthetic Stress Testing without hyper-parameter modification.


* The Net Cumulative Return must be greater than or equal to 2000% (20x).


* The Maximum Peak-to-Trough Drawdown must be less than or equal to 30%.


* The Calmar Ratio must be greater than or equal to 3.0.


* The Profit Factor must be greater than or equal to 1.8.


* The Monte Carlo Survival Rate must be greater than 95% of bootstrap runs maintaining a drawdown below 30%.
