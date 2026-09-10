import numpy as np
import pandas as pd


class StrictOptionSimEnv:
    """
    Simulated trading environment featuring dead-zone action filters, market impact,
    and linear/quadratic transaction costs to prevent over-trading.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list = None,
        fee_rate: float = 0.0003,       # 0.03% base fee
        impact_coef: float = 0.0001,    # Quadratic market impact coefficient
        dead_zone: float = 0.15,        # Minimum conviction barrier
        holding_penalty: float = 0.0001 # Holding friction
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

        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.current_position = 0.0
        self.equity = 10000.0

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = np.random.randint(0, max(1, self.max_steps - 2000))
        self.current_position = 0.0
        self.equity = 10000.0

        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = row[self.feature_cols].values.astype(np.float32)
        return np.append(obs, np.float32(self.current_position))

    def step(self, raw_action: np.ndarray):
        action_val = float(raw_action[0]) if isinstance(raw_action, (np.ndarray, list)) else float(raw_action)

        # 1. Action Dead Zone Mapping
        if abs(action_val) < self.dead_zone:
            target_position = 0.0
        else:
            sign = np.sign(action_val)
            scaled = (abs(action_val) - self.dead_zone) / (1.0 - self.dead_zone)
            target_position = float(sign * np.clip(scaled, 0.0, 1.0))

        # 2. Price Return & Cost Computations
        current_price = self.df.iloc[self.current_step]["close"]
        self.current_step += 1
        next_price = self.df.iloc[self.current_step]["close"]

        price_return = (next_price - current_price) / (current_price + 1e-8)
        raw_pnl = self.current_position * price_return

        # Transaction Fees + Quadratic Impact Slippage
        pos_delta = abs(target_position - self.current_position)
        turnover_cost = (pos_delta * self.fee_rate) + (self.impact_coef * (pos_delta ** 2))
        holding_cost = abs(target_position) * self.holding_penalty

        step_reward = raw_pnl - turnover_cost - holding_cost
        self.current_position = target_position
        self.equity *= (1.0 + step_reward)

        terminated = bool(self.current_step >= self.max_steps)
        truncated = bool(self.equity < 5000.0)

        info = {
            "pnl": raw_pnl,
            "turnover_cost": turnover_cost,
            "position": self.current_position,
            "equity": self.equity
        }

        return self._get_observation(), float(step_reward * 100.0), terminated, truncated, info
