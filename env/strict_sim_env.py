import numpy as np
import pandas as pd


class StrictOptionSimEnv:
    """
    Realistic trading simulation with fixed position sizing, transaction fees,
    and strict step sequence execution to prevent lookahead bias.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list = None,
        fee_rate: float = 0.0003,        # 0.03% fee
        impact_coef: float = 0.0001,     # Impact cost
        dead_zone: float = 0.05,         # Conviction barrier
        holding_penalty: float = 0.00005,
        initial_capital: float = 50000.0,
        max_position_size: float = 5000.0 # Maximum allocation per trade
    ):
        self.df = df.reset_index(drop=True)

        if feature_cols is None:
            self.feature_cols = [c for c in self.df.columns if c != "close"]
        else:
            self.feature_cols = feature_cols

        self.fee_rate = fee_rate
        self.impact_coef = impact_coef
        self.dead_zone = dead_zone
        self.holding_penalty = holding_penalty
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size

        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.current_position = 0.0
        self.equity = self.initial_capital

        self.history_capital = [self.equity]
        self.history_pnl = [0.0]
        self.history_positions = [0.0]

    @property
    def capital(self) -> float:
        return self.equity

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = np.random.randint(0, max(1, self.max_steps - 5000))
        self.current_position = 0.0
        self.equity = self.initial_capital

        self.history_capital = [self.equity]
        self.history_pnl = [0.0]
        self.history_positions = [0.0]

        return self._get_observation(), {}

    def _get_observation(self):
        safe_step = min(self.current_step, self.max_steps)
        row = self.df.iloc[safe_step]
        obs = row[self.feature_cols].values.astype(np.float32)
        return np.append(obs, np.float32(self.current_position))

    def step(self, raw_action: np.ndarray):
        if self.current_step >= self.max_steps:
            obs = self._get_observation()
            info = {
                "pnl": 0.0, "step_pnl": 0.0, "turnover_cost": 0.0,
                "trade_cost": 0.0, "position": self.current_position,
                "equity": self.equity, "capital": self.equity
            }
            return obs, 0.0, True, False, info

        action_val = float(raw_action[0]) if isinstance(raw_action, (np.ndarray, list)) else float(raw_action)

        # Action mapping
        if abs(action_val) < self.dead_zone:
            target_position = 0.0
        else:
            sign = np.sign(action_val)
            scaled = (abs(action_val) - self.dead_zone) / (1.0 - self.dead_zone)
            target_position = float(sign * np.clip(scaled, 0.0, 1.0))

        # Real step returns calculation
        current_price = self.df.iloc[self.current_step]["close"]
        self.current_step += 1
        next_price = self.df.iloc[min(self.current_step, self.max_steps)]["close"]

        price_diff = next_price - current_price

        # Absolute currency PnL based on position sizing instead of full multiplicative compounding
        trade_allocation = self.current_position * self.max_position_size
        raw_pnl = (trade_allocation / (current_price + 1e-8)) * price_diff

        pos_delta = abs(target_position - self.current_position)
        turnover_cost = (pos_delta * self.fee_rate * self.max_position_size) + (self.impact_coef * (pos_delta ** 2))
        holding_cost = abs(target_position) * self.holding_penalty * self.max_position_size

        net_pnl = raw_pnl - turnover_cost - holding_cost
        self.current_position = target_position
        self.equity += net_pnl

        self.history_capital.append(self.equity)
        self.history_pnl.append(net_pnl)
        self.history_positions.append(self.current_position)

        terminated = bool(self.current_step >= self.max_steps)
        truncated = bool(self.equity < (self.initial_capital * 0.5))

        info = {
            "pnl": net_pnl,
            "step_pnl": net_pnl,
            "turnover_cost": turnover_cost,
            "trade_cost": turnover_cost,
            "position": self.current_position,
            "equity": self.equity,
            "capital": self.equity
        }

        # Scaled step reward for stable gradient updates
        normalized_reward = float(np.clip(net_pnl / 100.0, -10.0, 10.0))

        return self._get_observation(), normalized_reward, terminated, truncated, info
