import numpy as np
import pandas as pd


class StrictOptionSimEnv:
    """
    Simulated trading environment featuring action dead-zones, market impact,
    linear/quadratic transaction costs, and step-by-step metric history tracking.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list = None,
        fee_rate: float = 0.0003,       # 0.03% base fee
        impact_coef: float = 0.0001,    # Quadratic market impact
        dead_zone: float = 0.05,        # Lowered to encourage early trade execution
        holding_penalty: float = 0.00005, # Mild friction
        initial_capital: float = 50000.0  # Set starting portfolio capital to ₹50,000
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

        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.current_position = 0.0
        self.equity = self.initial_capital

        # History Tracking
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
        # Index Guard
        if self.current_step >= self.max_steps:
            obs = self._get_observation()
            info = {
                "pnl": 0.0,
                "step_pnl": 0.0,
                "turnover_cost": 0.0,
                "trade_cost": 0.0,
                "position": self.current_position,
                "equity": self.equity,
                "capital": self.equity
            }
            return obs, 0.0, True, False, info

        action_val = float(raw_action[0]) if isinstance(raw_action, (np.ndarray, list)) else float(raw_action)

        # Dead Zone Action Mapping
        if abs(action_val) < self.dead_zone:
            target_position = 0.0
        else:
            sign = np.sign(action_val)
            scaled = (abs(action_val) - self.dead_zone) / (1.0 - self.dead_zone)
            target_position = float(sign * np.clip(scaled, 0.0, 1.0))

        # Returns and Friction
        current_price = self.df.iloc[self.current_step]["close"]
        self.current_step += 1
        next_price = self.df.iloc[min(self.current_step, self.max_steps)]["close"]

        price_return = (next_price - current_price) / (current_price + 1e-8)
        raw_pnl = self.current_position * price_return

        pos_delta = abs(target_position - self.current_position)
        turnover_cost = (pos_delta * self.fee_rate) + (self.impact_coef * (pos_delta ** 2))
        holding_cost = abs(target_position) * self.holding_penalty

        step_reward = raw_pnl - turnover_cost - holding_cost
        self.current_position = target_position
        self.equity *= (1.0 + step_reward)

        # Record History
        self.history_capital.append(self.equity)
        self.history_pnl.append(raw_pnl)
        self.history_positions.append(self.current_position)

        terminated = bool(self.current_step >= self.max_steps)
        truncated = bool(self.equity < (self.initial_capital * 0.5))

        info = {
            "pnl": raw_pnl,
            "step_pnl": raw_pnl,
            "turnover_cost": turnover_cost,
            "trade_cost": turnover_cost,
            "position": self.current_position,
            "equity": self.equity,
            "capital": self.equity
        }

        return self._get_observation(), float(step_reward * 100.0), terminated, truncated, info
