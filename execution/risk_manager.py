import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import Dict, Tuple

class RiskManager:
    """
    Risk & Portfolio Allocation Engine.
    Combines Ledoit-Wolf covariance shrinkage, Drawdown-Decaying Fractional Kelly,
    and strict intraday session window filters.
    """
    def __init__(
        self,
        max_drawdown_limit: float = 0.30,
        kelly_fraction: float = 0.5,
        max_single_position: float = 1.0
    ):
        self.max_dd_limit = max_drawdown_limit
        self.kelly_fraction = kelly_fraction
        self.max_single_position = max_single_position
        self.peak_portfolio_value: float = 1.0

    def is_valid_execution_window(self, current_time_str: str) -> bool:
        """
        Enforces execution timing windows (NSE IST):
        - Morning Trend: 09:30 - 11:30 IST
        - Afternoon Ramp: 13:30 - 15:15 IST
        """
        time_obj = pd.to_datetime(current_time_str).time()

        morning_start = pd.to_datetime("09:30").time()
        morning_end = pd.to_datetime("11:30").time()
        afternoon_start = pd.to_datetime("13:30").time()
        afternoon_end = pd.to_datetime("15:15").time()

        in_morning = morning_start <= time_obj <= morning_end
        in_afternoon = afternoon_start <= time_obj <= afternoon_end

        return in_morning or in_afternoon

    def compute_drawdown_decay(self, current_portfolio_value: float) -> float:
        """
        Computes non-linear drawdown penalty decay multiplier:
        Decay = (1 - Current_DD / Max_DD_Limit)^2
        """
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value

        current_dd = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value

        if current_dd >= self.max_dd_limit:
            return 0.0  # Hard stop: breach of drawdown threshold forces 0 allocation

        decay_factor = (1.0 - (current_dd / self.max_dd_limit)) ** 2
        return float(np.clip(decay_factor, 0.0, 1.0))

    def calculate_allocations(
        self,
        expected_returns: np.ndarray,      # Vector of predicted mean returns [mu_nifty, mu_banknifty]
        historical_returns: np.ndarray,    # Matrix of recent returns [T, N_assets]
        current_portfolio_value: float,
        timestamp_str: str
    ) -> Dict[str, float]:
        """
        Calculates risk-bounded position allocation weights for NIFTY and BANK NIFTY.
        """
        # Session timing check
        if not self.is_valid_execution_window(timestamp_str):
            return {"NIFTY": 0.0, "BANKNIFTY": 0.0, "status": "OUTSIDE_EXECUTION_WINDOW"}

        # Calculate Ledoit-Wolf shrunk covariance matrix
        lw = LedoitWolf()
        shrunk_cov = lw.fit(historical_returns).covariance_

        # Unconstrained Markowitz optimal weights: w = Cov^-1 * mu
        try:
            inv_cov = np.linalg.inv(shrunk_cov)
            raw_weights = inv_cov @ expected_returns
        except np.linalg.LinAlgError:
            raw_weights = np.zeros_like(expected_returns)

        # Apply Half-Kelly scaling and Drawdown Decay
        dd_decay = self.compute_drawdown_decay(current_portfolio_value)
        scaled_weights = raw_weights * self.kelly_fraction * dd_decay

        # Enforce single position bounds [-max_single_position, +max_single_position]
        bounded_weights = np.clip(scaled_weights, -self.max_single_position, self.max_single_position)

        return {
            "NIFTY": float(bounded_weights[0]),
            "BANKNIFTY": float(bounded_weights[1]),
            "drawdown_decay": dd_decay,
            "status": "ACTIVE"
        }
