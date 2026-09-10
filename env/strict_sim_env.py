import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class StrictOptionSimEnv(gym.Env):
    """
    High-Fidelity Intraday Options Trading Simulation Environment.
    Includes strict transaction fees, bid-ask spreads, and action dead zones
    to prevent micro-churning and trade-frequency explosion.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_capital: float = 25_000.0,
        lot_size: int = 25,
        fee_per_trade: float = 20.0,
        slippage_pct: float = 0.0005,
        action_threshold: float = 0.4
    ):
        super(StrictOptionSimEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.fee_per_trade = fee_per_trade
        self.slippage_pct = slippage_pct
        self.action_threshold = action_threshold

        # Action Space: [-1.0, 1.0] -> Continual signal map
        # -1.0 to -0.4: Short / Sell | -0.4 to 0.4: FLAT / HOLD | 0.4 to 1.0: Long / Buy
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation Space
        num_features = len(self.feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.capital = self.initial_capital
        self.position = 0  # -1: Short, 0: Flat, 1: Long
        self.entry_price = 0.0
        self.history_capital = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.history_capital = [self.capital]

        obs = self.df.loc[self.current_step, self.feature_cols].to_numpy(dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        raw_action = float(action[0])

        # 1. Action Dead-Zone Mapping (Prevents Micro-Churning)
        if raw_action > self.action_threshold:
            target_position = 1
        elif raw_action < -self.action_threshold:
            target_position = -1
        else:
            target_position = 0

        current_price = float(self.df.loc[self.current_step, "close"])
        step_pnl = 0.0
        trade_cost = 0.0

        # 2. Position Change & Transaction Costs Execution
        if target_position != self.position:
            # Exit existing position if active
            if self.position != 0:
                exit_price = current_price * (1.0 - self.slippage_pct if self.position == 1 else 1.0 + self.slippage_pct)
                pnl_points = (exit_price - self.entry_price) if self.position == 1 else (self.entry_price - exit_price)
                step_pnl += pnl_points * self.lot_size
                trade_cost += self.fee_per_trade

            # Enter new position if target is non-flat
            if target_position != 0:
                self.entry_price = current_price * (1.0 + self.slippage_pct if target_position == 1 else 1.0 - self.slippage_pct)
                trade_cost += self.fee_per_trade

            self.position = target_position
        elif self.position != 0:
            # Mark-to-market step reward for holding active positions
            next_price = float(self.df.loc[min(self.current_step + 1, self.max_steps), "close"])
            price_diff = next_price - current_price
            step_pnl += (price_diff if self.position == 1 else -price_diff) * self.lot_size

        net_step_pnl = step_pnl - trade_cost
        self.capital += net_step_pnl
        self.history_capital.append(self.capital)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps or self.capital <= (self.initial_capital * 0.2)
        truncated = False

        # Reward formulation: Reward Net PnL - Turnover Penalty
        turnover_penalty = 1.0 if (target_position != self.position and target_position != 0) else 0.0
        reward = net_step_pnl - turnover_penalty

        next_obs = self.df.loc[min(self.current_step, self.max_steps), self.feature_cols].to_numpy(dtype=np.float32)

        info = {
            "step_pnl": step_pnl,
            "trade_cost": trade_cost,
            "capital": self.capital,
            "position": self.position
        }

        return next_obs, reward, terminated, truncated, info
