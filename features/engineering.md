# State Vector Formulation (Zero-Lookahead)

To ensure the model learns market invariants rather than overfitting absolute price levels, the state vector $S_t$ at time $t$ consists strictly of stationary transformations.

## 1. Price Action (Log Returns)
Absolute prices are non-stationary. We use rolling log returns.
$R_{t, k} = \ln(P_t / P_{t-k})$
Calculated for multiple time horizons (e.g., $k \in \{1, 5, 15, 60\}$ minutes).

## 2. Volatility (Parkinson Estimator)
To capture intra-bar variance without relying solely on close prices.
$V_t = \sqrt{\frac{1}{4 \ln(2)} \left( \ln\left(\frac{H_t}{L_t}\right) \right)^2}$
*Smoothed via Exponential Moving Average (EMA).*

## 3. Order Flow / Volume Imbalance Shock
Measures aggressive buying vs. selling pressure, normalized via rolling Z-score to handle volume spikes at market open/close.
$Z(V_t) = \frac{V_t - \mu_{V, t-w:t}}{\sigma_{V, t-w:t}}$
Where $w$ is the rolling window (e.g., 200 periods).

## 4. Fractional Differentiation (Advanced Memory Preservation)
Instead of standard integer differencing ($d=1$), we apply fractional differencing $d \in (0, 1)$ to achieve stationarity (ADF test $p < 0.05$) while preserving maximum memory of the original price series.
$ \Delta^d X_t = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} X_{t-k} $
*Weights truncated when they fall below a significance threshold (e.g., 1e-4).*
