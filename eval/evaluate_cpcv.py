import numpy as np
import polars as pl
from itertools import combinations
from typing import List, Tuple, Generator

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV) for time series backtesting.
    Enforces time-based purging and embargoing between train and test splits
    to prevent lookahead bias and autocorrelation leakage.
    """
    def __init__(self, n_splits: int = 5, n_test_splits: int = 2, purge_window: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_window = purge_window  # Number of steps forward (e.g., target horizon = 5m)
        self.embargo_pct = embargo_pct

    def split(self, df: pl.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(df)
        indices = np.arange(n_samples)

        # Partition data into contiguous temporal blocks
        block_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)
        blocks = [indices[block_bounds[i]:block_bounds[i+1]] for i in range(self.n_splits)]

        embargo_offset = int(n_samples * self.embargo_pct)

        # Generate combinations of test blocks
        test_block_combos = list(combinations(range(self.n_splits), self.n_test_splits))

        for test_combo in test_block_combos:
            test_indices_list = [blocks[i] for i in test_combo]
            test_indices = np.concatenate(test_indices_list)

            # Start with all remaining indices as train set candidate
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            # Apply Purging and Embargoing around each test block
            for b_idx in test_combo:
                block_start = blocks[b_idx][0]
                block_end = blocks[b_idx][-1]

                # Purge: remove training samples immediately preceding test block whose evaluation overlaps
                purge_start = max(0, block_start - self.purge_window)
                train_mask[purge_start:block_start] = False

                # Embargo: remove training samples immediately following test block to break serial correlation
                embargo_end = min(n_samples, block_end + 1 + embargo_offset)
                train_mask[block_end + 1:embargo_end] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices
