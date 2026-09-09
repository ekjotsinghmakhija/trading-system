import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any, Union


class AdvancedFrictionEnv(gym.Env):
    """
    Advanced Trading Environment featuring continuous position allocation [-1, 1],
    holding duration penalties, dynamic volatility slippage, and drawdown limits.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: Union[pd.DataFrame, pd.Series], features: pd.DataFrame, initial_balance: float = 50000.0):
        super(AdvancedFrictionEnv, self).__init__()

        # Defensive type checks
        if isinstance(data, (list, np.ndarray)):
            self.data = pd.DataFrame(data, columns=['close']).reset_index(drop=True)
        elif isinstance(data, pd.Series):
            self.data = data.to_frame(name='close').reset_index(drop=True)
        else:
            self.data = data.reset_index(drop=True)

        self.feature_colnames = list(features.columns)
        self.parkinson_idx = self.feature_colnames.index("parkinson_vol") if "parkinson_vol" in self.feature_colnames else None

        self.features = features.reset_index(drop=True).values.astype(np.float32)
        self.initial_balance = initial_balance
        self.max_steps = len(self.data) - 1

        # Continuous action space: target leverage from -1.0 (100% Short) to +1.0 (100% Long)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space includes features plus current position leverage
        self.obs_shape = self.features.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

        self.sim_txn_cost_pct = 0.0003 * 1.10  # Base cost + 10% safety buffer

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.current_position = 0.0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        feat_obs = self.features[self.current_step]
        return np.append(feat_obs, self.current_position).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        target_position = float(np.clip(action[0], -1.0, 1.0))

        current_price = float(self.data.iloc[self.current_step]['close'])
        next_price = float(self.data.iloc[self.current_step + 1]['close'])

        position_change = abs(target_position - self.current_position)

        if self.parkinson_idx is not None:
            volatility = float(self.features[self.current_step][self.parkinson_idx])
        else:
            volatility = 0.001

        slippage = (0.0001 + (volatility * 0.05)) * 1.10
        friction_cost = position_change * (self.sim_txn_cost_pct + slippage)

        market_return = (next_price - current_price) / current_price
        gross_return = self.current_position * market_return
        net_step_return = gross_return - friction_cost

        self.current_position = target_position
        self.balance *= (1.0 + net_step_return)

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        self.current_drawdown = (self.peak_balance - self.balance) / self.peak_balance

        # Reward formulation
        reward = net_step_return * 100.0
        if self.current_drawdown > 0.20:
            reward -= (self.current_drawdown - 0.20) * 50.0

        terminated = False
        truncated = False

        if self.current_drawdown >= 0.30:
            reward -= 50.0
            terminated = True

        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            "balance": self.balance,
            "drawdown": self.current_drawdown,
            "position": self.current_position
        }

        return self._get_observation(), reward, terminated, truncated, info
