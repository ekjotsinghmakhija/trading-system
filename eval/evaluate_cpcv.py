import numpy as np
import pandas as pd


class CPCVEvaluator:
    """
    Combinatorial Purged Cross-Validation Engine.
    Enforces Purging and Embargoing between train/test splits to eliminate leakage.
    """
    def __init__(self, n_splits: int = 5, purge_window: int = 60, embargo_window: int = 120):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def generate_purged_folds(self, total_samples: int):
        fold_size = total_samples // self.n_splits
        indices = np.arange(total_samples)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else total_samples

            test_idx = indices[test_start:test_end]

            # Apply Purging & Embargoing
            train_mask = np.ones(total_samples, dtype=bool)

            # Purge before and after test set
            purge_start = max(0, test_start - self.purge_window)
            purge_end = min(total_samples, test_end + self.purge_window + self.embargo_window)

            train_mask[purge_start:purge_end] = False
            train_idx = indices[train_mask]

            yield train_idx, test_idx

    def compute_deflated_sharpe_ratio(self, returns: np.ndarray, num_trials: int = 10) -> float:
        """
        Calculates Deflated Sharpe Ratio (DSR) adjusting for backtest trial count.
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        sr_calc = np.mean(returns) / np.std(returns) * np.sqrt(375 * 252)
        skew = pd.Series(returns).skew()
        kurt = pd.Series(returns).kurtosis()

        # Expected maximum Sharpe under null hypothesis
        e_max_sr = (1 - 0.57721566) * np.quantile(np.random.normal(0, 1, 10000), 1 - 1/num_trials) + \
                   0.57721566 * np.quantile(np.random.normal(0, 1, 10000), 1 - 1/(num_trials * np.e))

        denom = np.sqrt(1 - skew * sr_calc + ((kurt - 1) / 4) * (sr_calc ** 2))
        if denom == 0 or np.isnan(denom):
            return 0.0

        dsr = (sr_calc - e_max_sr) * np.sqrt(len(returns) - 1) / denom
        return float(dsr)
