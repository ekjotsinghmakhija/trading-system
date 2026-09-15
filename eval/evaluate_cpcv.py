# eval/evaluate_cpcv.py

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from typing import List, Tuple, Generator


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged Cross-Validation (CPCV) with Embargoing.
    Prevents leakage caused by overlapping triple-barrier target horizons.
    """
    def __init__(self, n_splits: int = 5, n_test_splits: int = 2, pct_embargo: float = 0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.pct_embargo = pct_embargo

    def split(
        self,
        df: pd.DataFrame,
        holding_periods: pd.Series
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(df)
        indices = np.arange(n_samples)
        embargo_offset = int(n_samples * self.pct_embargo)

        # Generate chunk bounds
        chunk_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)
        chunks = [(chunk_bounds[i], chunk_bounds[i + 1]) for i in range(self.n_splits)]

        from itertools import combinations
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))

        for test_chunk_ids in test_combinations:
            test_indices = []
            purge_mask = np.zeros(n_samples, dtype=bool)

            for cid in test_chunk_ids:
                start, end = chunks[cid]
                test_indices.extend(indices[start:end])

                # Purge train samples prior to test start whose labels overlap into test set
                for t in range(max(0, start - 50), start):
                    if t + holding_periods.iloc[t] >= start:
                        purge_mask[t] = True

                # Apply post-test embargo window
                embargo_end = min(n_samples, end + embargo_offset)
                purge_mask[end:embargo_end] = True

            test_idx = np.array(test_indices)
            purge_mask[test_idx] = True
            train_idx = indices[~purge_mask]

            yield train_idx, test_idx


def compute_deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 10,
    benchmark_sr: float = 0.0
) -> float:
    """
    Computes Deflated Sharpe Ratio (DSR) to control for multiple testing over-optimism.
    """
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    n = len(returns)
    sr_hat = (returns.mean() / returns.std()) * np.sqrt(252 * 375)

    sk = skew(returns)
    kt = kurtosis(returns, fisher=True)

    # Estimate expected maximum Sharpe ratio under null hypothesis
    e_max_sr = benchmark_sr + (1 - 0.5772156649) * norm.ppf(1 - 1.0 / n_trials) + 0.5772156649 * norm.ppf(1 - 1.0 / (n_trials * np.e))

    sr_variance = (1 + (0.5 * sr_hat**2) - (sk * sr_hat) + ((kt / 4.0) * sr_hat**2)) / (n - 1)
    dsr_statistic = (sr_hat - e_max_sr) / np.sqrt(max(sr_variance, 1e-8))

    return float(norm.cdf(dsr_statistic))
