# Data Ingestion & Sanitization Protocol

Financial data is inherently noisy, incomplete, and subject to structural breaks. The ingestion engine must sanitize the raw 1-minute OHLCV data to ensure structural integrity before feature engineering.

## 1. Nan & Gap Handling (Strict Zero-Lookahead)
*   **Rule:** Never backward-fill (`bfill`) missing data. Backward filling leaks future prices into the past.
*   **Protocol:** Use strictly forward-fill (`ffill`) for missing close prices to reflect the last known state. Volume for gap periods must be set to `0`.
*   **Execution:** $P_{t} = P_{t-1}$ if $P_t$ is `NaN`.

## 2. Continuous Futures Adjustment
Derivatives expire (weekly/monthly). Splicing raw contracts creates artificial price jumps (gaps) that neural networks misinterpret as massive returns.
*   **Panama Canal Method (Backward Difference):**
    Adjust historical contracts so the price matches the new contract at the rollover date.
    $P_{adjusted, t} = P_{raw, t} + (P_{new, roll} - P_{old, roll})$
*   *Note: This preserves absolute point differences (crucial for PnL calculations) but can result in negative prices for very old data. For percentage return models, Backward Ratio adjustment is preferred:*
    $P_{adjusted, t} = P_{raw, t} \times \left( \frac{P_{new, roll}}{P_{old, roll}} \right)$

## 3. Outlier Truncation (Spike Filter)
Exchange glitches create anomalous wicks. Cap tick-to-tick returns that exceed $5\sigma$ of the rolling 10,000-period variance to prevent gradient explosions during model training.
