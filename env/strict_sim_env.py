import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class StrictOptionSimEnv(gym.Env):
    """
    High-Performance Production Gymnasium Environment for Intraday Options.
    Implements continuous target mapping, Indian options market frictions,
    and portfolio return reward shaping.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_capital: float = 50000.0,
        sequence_length: int = 60,
        lot_size: int = 25
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.sequence_length = sequence_length
        self.num_features = len(feature_cols)

        # C-contiguous NumPy arrays for high-speed tensor conversion
        self.features_array = np.ascontiguousarray(self.df[self.feature_cols].to_numpy(dtype=np.float32))
        self.close_prices = np.ascontiguousarray(self.df["close"].to_numpy(dtype=np.float32))
        self.total_rows = len(self.df)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sequence_length, self.num_features), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Indian Options Market Operational Constraints
        self.LOT_SIZE = lot_size
        self.STT_PREMIUM_RATE = 0.000625  # STT rate applied on option sell turnover
        self.BROKERAGE_PER_ORDER = 20.0  # Flat brokerage per leg (₹20)
        self.EXCHANGE_CHARGES = 0.0005     # NSE transaction fees + GST
        self.MAX_HOLDING_MINUTES = 45      # Cap maximum position holding time
        self.ACTION_EXEC_THRESHOLD = 0.10  # Active action boundary to eliminate dead zones

        self.current_step = 0
        self.capital = initial_capital
        self.position = 0  # Position states: 1 (Long), -1 (Short proxy), 0 (Flat)
        self.entry_price = 0.0
        self.entry_step = 0
        self.portfolio_values = []

    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        return self.features_array[start_idx:end_idx]

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = self.sequence_length - 1
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_step = 0
        self.portfolio_values = [self.initial_capital]

        obs = self._get_observation()
        info = {"capital": self.capital, "position": self.position}
        return obs, info

    def _calculate_frictions(self, premium: float, is_exit: bool = False) -> float:
        turnover = premium * self.LOT_SIZE
        stt = turnover * self.STT_PREMIUM_RATE if is_exit else 0.0
        exchange = turnover * self.EXCHANGE_CHARGES
        return self.BROKERAGE_PER_ORDER + stt + exchange

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raw_act = float(action[0])
        current_price = float(self.close_prices[self.current_step])

        # Map continuous actions to target allocation state
        target_pos = 0
        if raw_act > self.ACTION_EXEC_THRESHOLD:
            target_pos = 1
        elif raw_act < -self.ACTION_EXEC_THRESHOLD:
            target_pos = -1

        step_pnl = 0.0
        friction_cost = 0.0
        forced_exit = False

        # 1. Enforce time-decay cap rule
        if self.position != 0 and (self.current_step - self.entry_step) >= self.MAX_HOLDING_MINUTES:
            target_pos = 0
            forced_exit = True

        # 2. Force square-off at end of session data
        if self.current_step >= self.total_rows - 2:
            target_pos = 0

        # Execute position transition
        if target_pos != self.position:
            # Exit active position
            if self.position != 0:
                raw_return = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
                raw_pnl = raw_return * self.LOT_SIZE
                exit_friction = self._calculate_frictions(current_price, is_exit=True)
                friction_cost += exit_friction
                step_pnl += (raw_pnl - exit_friction)
                self.position = 0

            # Enter new position
            if target_pos != 0:
                entry_friction = self._calculate_frictions(current_price, is_exit=False)
                friction_cost += entry_friction
                step_pnl -= entry_friction
                self.position = target_pos
                self.entry_price = current_price
                self.entry_step = self.current_step

        elif self.position != 0:
            # Mark-to-market intra-bar continuous PnL tracking
            prev_price = float(self.close_prices[self.current_step - 1])
            price_diff = current_price - prev_price
            step_pnl += (price_diff * self.LOT_SIZE) if self.position == 1 else (-price_diff * self.LOT_SIZE)

        prev_capital = self.capital
        self.capital += step_pnl
        self.portfolio_values.append(self.capital)

        capital_return = step_pnl / prev_capital if prev_capital > 0 else 0.0

        # Reward formulation: Return percentage minus friction penalty
        reward = capital_return * 100.0
        if friction_cost > 0:
            reward -= (friction_cost / prev_capital) * 10.0

        if forced_exit:
            reward -= 0.05

        self.current_step += 1
        terminated = self.current_step >= self.total_rows - 1
        truncated = self.capital <= (self.initial_capital * 0.50)  # Stop run at 50% max drawdown

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "capital": self.capital,
            "position": self.position,
            "step_pnl": step_pnl,
            "friction_cost": friction_cost,
            "forced_exit": forced_exit
        }

        return obs, reward, terminated, truncated, info
