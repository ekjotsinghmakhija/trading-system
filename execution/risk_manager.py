# execution/risk_manager.py

import logging
import numpy as np
import pandas as pd
from datetime import datetime, time
from sklearn.covariance import LedoitWolf
from typing import Dict, Any, Union, Optional

logger = logging.getLogger("RiskManager")


class RiskManager:
    """
    Unified Intraday Risk & Portfolio Allocation Engine.
    Combines Ledoit-Wolf covariance shrinkage, Drawdown-Decaying Fractional Kelly,
    NSE execution session window filters, and strict options theta cutoffs.
    """
    def __init__(
        self,
        max_drawdown_limit: float = 0.30,
        max_daily_loss_pct: float = 0.02,
        kelly_fraction: float = 0.5,
        max_single_position: float = 1.0,
        max_position_lots: int = 4,
        options_theta_cutoff: time = time(14, 45)
    ):
        self.max_dd_limit = max_drawdown_limit
        self.max_daily_loss_pct = max_daily_loss_pct
        self.kelly_fraction = kelly_fraction
        self.max_single_position = max_single_position
        self.max_position_lots = max_position_lots
        self.options_theta_cutoff = options_theta_cutoff

        self.peak_portfolio_value: float = 1.0
        self.current_daily_pnl: float = 0.0

    def is_valid_execution_window(self, current_time: Union[str, datetime, time]) -> bool:
        """
        Enforces execution timing windows (NSE IST):
        - Morning Trend: 09:30 - 11:30 IST
        - Afternoon Ramp: 13:30 - 15:15 IST
        - Hard Options Theta Cutoff: 14:45 IST
        """
        if isinstance(current_time, str):
            time_obj = pd.to_datetime(current_time).time()
        elif isinstance(current_time, datetime):
            time_obj = current_time.time()
        elif isinstance(current_time, time):
            time_obj = current_time
        else:
            raise ValueError(f"Unsupported time format: {type(current_time)}")

        # Hard cutoff for late-day options buys
        if time_obj >= self.options_theta_cutoff:
            return False

        morning_start = time(9, 30)
        morning_end = time(11, 30)
        afternoon_start = time(13, 30)
        afternoon_end = time(15, 15)

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

        if self.peak_portfolio_value <= 0:
            return 1.0

        current_dd = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value

        if current_dd >= self.max_dd_limit:
            logger.warning(f"[HARD STOP] Portfolio breach: DD {current_dd:.2%} >= Limit {self.max_dd_limit:.2%}")
            return 0.0

        decay_factor = (1.0 - (current_dd / self.max_dd_limit)) ** 2
        return float(np.clip(decay_factor, 0.0, 1.0))

    def validate_signal(
        self,
        signal: int,
        portfolio_value: float,
        current_time: Union[str, datetime]
    ) -> bool:
        """
        Validates individual directional signals against session limits & daily drawdown limits.
        """
        if signal == 0:
            return False

        # Session timing check
        if not self.is_valid_execution_window(current_time):
            logger.warning(f"[RISK REJECT] Signal {signal} rejected outside execution window ({current_time})")
            return False

        # Max daily loss check
        max_loss_amount = portfolio_value * self.max_daily_loss_pct
        if self.current_daily_pnl <= -max_loss_amount:
            logger.error(f"[RISK REJECT] Max daily loss breach: PnL = ₹{self.current_daily_pnl:.2f}")
            return False

        return True

    def calculate_allocations(
        self,
        expected_returns: np.ndarray,      # Vector of predicted mean returns [mu_nifty, mu_banknifty]
        historical_returns: np.ndarray,    # Matrix of recent returns [T, N_assets]
        current_portfolio_value: float,
        timestamp_str: str
    ) -> Dict[str, Any]:
        """
        Calculates Markowitz optimal allocations using Ledoit-Wolf shrunk covariance,
        scaled by Half-Kelly leverage and Drawdown Decay.
        """
        if not self.is_valid_execution_window(timestamp_str):
            return {
                "NIFTY": 0.0,
                "BANKNIFTY": 0.0,
                "drawdown_decay": 0.0,
                "status": "OUTSIDE_EXECUTION_WINDOW"
            }

        # Ledoit-Wolf shrunk covariance calculation
        try:
            lw = LedoitWolf()
            shrunk_cov = lw.fit(historical_returns).covariance_
            inv_cov = np.linalg.inv(shrunk_cov)
            raw_weights = inv_cov @ expected_returns
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.error(f"[COV ERROR] Shrinkage failed: {e}. Falling back to zero allocation.")
            raw_weights = np.zeros_like(expected_returns)

        # Fractional Kelly & Drawdown Decay scaling
        dd_decay = self.compute_drawdown_decay(current_portfolio_value)
        scaled_weights = raw_weights * self.kelly_fraction * dd_decay

        # Single position clipping
        bounded_weights = np.clip(scaled_weights, -self.max_single_position, self.max_single_position)

        return {
            "NIFTY": float(bounded_weights[0]) if len(bounded_weights) > 0 else 0.0,
            "BANKNIFTY": float(bounded_weights[1]) if len(bounded_weights) > 1 else 0.0,
            "drawdown_decay": dd_decay,
            "status": "ACTIVE"
        }

    def update_pnl(self, trade_pnl: float) -> None:
        """Accumulates daily realized PnL."""
        self.current_daily_pnl += trade_pnl


# Backward-compatibility alias
IntradayRiskManager = RiskManager
