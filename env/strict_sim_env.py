import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any, List

class StrictFrictionEnv(gym.Env):
    """
    Custom Gym Environment for algorithmic trading with +10% friction strictness
    and drawdown-penalized reward mechanics.
    """
    metadata = {'render_modes': ['human', 'system']}

    def __init__(self, data: pd.DataFrame, features: pd.DataFrame, initial_balance: float = 50000.0):
        super(StrictFrictionEnv, self).__init__()

        # Ensure data alignment
        assert len(data) == len(features), "Raw data and features must align perfectly."

        self.data = data.reset_index(drop=True)
        self.features = features.reset_index(drop=True).values
        self.initial_balance = initial_balance
        self.max_steps = len(self.data) - 1

        # Action Space: 0 (Short), 1 (Flat), 2 (Long)
        self.action_space = spaces.Discrete(3)

        # Observation Space: N-dimensional feature vector
        self.obs_shape = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

        # Friction Constants (Indian Markets + 10%)
        # Brokerage + STT + GST + SEBI approximated as percentage of notional
        self.base_txn_cost_pct = 0.0003  # ~3 bps average per leg
        self.strictness_multiplier = 1.10
        self.sim_txn_cost_pct = self.base_txn_cost_pct * self.strictness_multiplier

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0

        self.current_position = 0 # -1, 0, 1

        return self.features[self.current_step], {}

    def _calculate_slippage(self, volatility: float) -> float:
        """Dynamic slippage scaled by current volatility and strictness multiplier."""
        base_slippage_pct = 0.0001 # 1 bps minimum
        vol_adjusted_slippage = base_slippage_pct + (volatility * 0.05)
        return vol_adjusted_slippage * self.strictness_multiplier

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map action [0, 1, 2] to position [-1, 0, 1]
        target_position = action - 1

        # 1. Fetch current step data
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        volatility = self.features[self.current_step][self.features.columns.get_loc("parkinson_vol")] # Assuming mapped

        # 2. Execution logic & Friction
        transaction_cost = 0.0
        slippage_cost = 0.0

        if target_position != self.current_position:
            # We are transacting
            transaction_cost = self.sim_txn_cost_pct
            slippage_cost = self._calculate_slippage(volatility)
            self.current_position = target_position

        # 3. Calculate Market Return (t to t+1)
        market_return = (next_price - current_price) / current_price

        # 4. Calculate Portfolio Step Return (Zero Leverage)
        step_return = (self.current_position * market_return) - transaction_cost - slippage_cost

        # Update Balance
        self.balance *= (1 + step_return)

        # 5. Drawdown Tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        self.current_drawdown = (self.peak_balance - self.balance) / self.peak_balance

        # 6. Drawdown-Penalized Reward Function
        # Base reward is the step PnL percentage
        reward = step_return

        # Continuous quadratic penalty for being in drawdown
        lambda_penalty = 2.0
        reward -= lambda_penalty * (self.current_drawdown ** 2)

        # 7. Terminal Conditions
        terminated = False
        truncated = False

        # Hard Stop: > 25% Drawdown
        if self.current_drawdown > 0.25:
            reward -= 10.0 # Massive terminal penalty (beta)
            terminated = True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        next_obs = self.features[self.current_step]
        info = {
            "balance": self.balance,
            "drawdown": self.current_drawdown,
            "position": self.current_position
        }

        return next_obs, reward, terminated, truncated, info
