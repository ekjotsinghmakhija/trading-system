import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict

class FactorLedger:
    """
    High-dimensional state memory ledger.
    Tracks feature states S_t against forward return outcomes R_{t+5}
    and computes hazard regime entry suppression masks.
    """
    def __init__(self, hazard_threshold: float = -0.0015, target_threshold: float = 0.0020, k_neighbors: int = 25):
        self.hazard_threshold = hazard_threshold
        self.target_threshold = target_threshold
        self.k_neighbors = k_neighbors
        self.nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        self.state_matrix: np.ndarray = np.array([])
        self.outcomes: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def build_ledger(self, df: pl.DataFrame, feature_cols: list) -> Dict[str, int]:
        """
        Extracts feature vectors and forward path labels from feature DataFrame.
        """
        clean_df = df.drop_nulls(subset=feature_cols + ["target_5m_return"])
        self.state_matrix = clean_df.select(feature_cols).to_numpy()
        self.outcomes = clean_df.select("target_5m_return").to_numpy().flatten()

        # Labels: +1 (Target), -1 (Hazard), 0 (Neutral)
        labels = np.zeros_like(self.outcomes)
        labels[self.outcomes >= self.target_threshold] = 1
        labels[self.outcomes <= self.hazard_threshold] = -1

        self.nn_model.fit(self.state_matrix)
        self.is_fitted = True

        return {
            "total_states": len(self.outcomes),
            "target_states": int(np.sum(labels == 1)),
            "hazard_states": int(np.sum(labels == -1)),
            "neutral_states": int(np.sum(labels == 0))
        }

    def evaluate_hazard_probability(self, query_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates query state vector(s) against historical ledger.
        Returns (hazard_probability, entry_suppression_mask).
        """
        if not self.is_fitted:
            raise RuntimeError("FactorLedger must be built before evaluating hazard states.")

        if query_state.ndim == 1:
            query_state = query_state.reshape(1, -1)

        distances, indices = self.nn_model.kneighbors(query_state)

        # Compute local hazard ratio among k-nearest neighbors
        neighbor_outcomes = self.outcomes[indices]
        hazard_counts = np.sum(neighbor_outcomes <= self.hazard_threshold, axis=1)
        hazard_probs = hazard_counts / self.k_neighbors

        # Suppress entries if > 35% of nearest historical states resulted in hazard outcomes
        suppression_mask = hazard_probs > 0.35
        return hazard_probs, suppression_mask
