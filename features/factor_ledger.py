import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple, List

class FactorLedger:
    def __init__(
        self,
        hazard_threshold: float = -0.0015,
        target_threshold: float = 0.0020,
        k_neighbors: int = 25
    ):
        self.hazard_threshold = hazard_threshold
        self.target_threshold = target_threshold
        self.k_neighbors = k_neighbors
        self.nn_model = None
        self.state_matrix = None
        self.is_hazard_state = None

    def build_ledger(self, df: pl.DataFrame, feature_cols: List[str]) -> Dict[str, int]:
        # Filter for hazard states
        hazard_df = df.filter(pl.col("target_5m_return") <= self.hazard_threshold)

        # Fall back to 5th percentile if threshold yields no rows
        if len(hazard_df) == 0:
            q_val = df.select(pl.col("target_5m_return").quantile(0.05)).item()
            hazard_df = df.filter(pl.col("target_5m_return") <= q_val)

        if len(hazard_df) == 0:
            hazard_df = df

        self.state_matrix = hazard_df.select(feature_cols).to_numpy().astype(np.float32)
        self.is_hazard_state = np.ones(len(self.state_matrix), dtype=bool)

        # Parallelize KNN fit across all CPU threads
        n_neighbors = min(self.k_neighbors, len(self.state_matrix))
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1)
        self.nn_model.fit(self.state_matrix)

        return {
            "total_states": len(df),
            "indexed_hazard_states": len(self.state_matrix),
            "k_neighbors": n_neighbors
        }

    def evaluate_hazard_probability(self, current_state: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.nn_model is None or self.state_matrix is None or len(self.state_matrix) == 0:
            return 0.0, np.array([False])

        state_input = current_state.reshape(1, -1).astype(np.float32)
        distances, indices = self.nn_model.kneighbors(state_input)

        hazard_neighbors = self.is_hazard_state[indices[0]]
        hazard_prob = float(np.mean(hazard_neighbors))
        is_suppressed = hazard_prob >= 0.6

        return hazard_prob, np.array([is_suppressed])
