# CPCV & Anti-Leakage Validation Strategy

Standard chronological train/test splits (Train 2018-2022 -> Test 2023) result in a model that only learns one specific market path. Standard K-Fold leaks future data into the past. Combinatorial Purged Cross-Validation (CPCV) solves this.

## 1. The Path Generation
Divide the total dataset into $N$ chronological blocks (e.g., $N=6$).
Choose $k$ blocks to form the testing set (e.g., $k=2$).
This yields $\binom{N}{k}$ unique train/test splits. The model trains on the remaining $N-k$ blocks.

## 2. Purging (Addressing Horizon Overlap)
If our labels (or features) look $h$ periods ahead/behind, the training data immediately preceding a test block contains overlapping information.
*   **Action:** Drop all training samples where the timestamp falls within $[t_{\text{test\_start}} - h, t_{\text{test\_start}}]$.

## 3. Embargoing (Addressing Serial Correlation)
Financial markets exhibit strong autoregressive properties (momentum). Training on data immediately following a test block allows the model to "guess" the test set outcome by looking at the immediate aftermath.
*   **Action:** Implement an embargo window $e$ (e.g., 5 days of 1-minute bars). Drop all training samples within $[t_{\text{test\_end}}, t_{\text{test\_end}} + e]$.

## 4. Evaluation Path
By training on these purged/embargoed combinations, we generate multiple Out-of-Sample (OOS) predictions. Recombining these yields a single continuous OOS equity curve spanning the entire dataset, proving the model is robust to diverse market regimes, not just a single chronological sequence.
