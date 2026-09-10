import numpy as np
import pandas as pd


class StrictOptionSimEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list = None,
        initial_capital: float = 50000.0,
        cooldown_steps: int = 5,
        max_hold_steps: int = 60
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols if feature_cols else [c for c in self.df.columns if c != "close"]
        self.initial_capital = initial_capital
        self.cooldown_steps = cooldown_steps
        self.max_hold_steps = max_hold_steps

        self.variable_friction_rate = 0.0018
        self.fixed_order_fee = 22.0
        self.max_trade_allocation = 19500.0  # Phase 1 Seed (3 lots)
        self.daily_drawdown_limit = 0.05 * self.initial_capital

        self.reset()

    @property
    def capital(self) -> float:
        return self.equity

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = np.random.randint(0, max(1, len(self.df) - 5000))
        self.max_steps = len(self.df) - 1

        self.current_position = 0.0
        self.equity = self.initial_capital
        self.day_start_equity = self.equity
        self.entry_step = 0
        self.last_trade_step = -self.cooldown_steps

        return self._get_observation(), {}

    def _get_observation(self):
        safe_step = min(self.current_step, self.max_steps)
        row = self.df.iloc[safe_step]
        obs = row[self.feature_cols].values.astype(np.float32)
        hold_ratio = np.float32((self.current_step - self.entry_step) / self.max_hold_steps if self.current_position != 0 else 0.0)
        return np.append(obs, [np.float32(self.current_position), hold_ratio])

    def step(self, raw_action: np.ndarray):
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, False, {"equity": self.equity}

        action_val = float(raw_action[0]) if isinstance(raw_action, (np.ndarray, list)) else float(raw_action)

        proposed_position = 0.0 if abs(action_val) < 0.25 else float(np.sign(action_val) * np.clip((abs(action_val) - 0.25) / 0.75, 0.0, 1.0))

        # Enforce holding cooldown
        pos_delta = abs(proposed_position - self.current_position)
        is_cooldown_active = (self.current_step - self.last_trade_step) < self.cooldown_steps

        if pos_delta > 1e-3 and is_cooldown_active:
            target_position = self.current_position
            pos_delta = 0.0
        else:
            target_position = proposed_position

        # Enforce 60-Bar Theta Wall
        holding_duration = self.current_step - self.entry_step if self.current_position != 0 else 0
        if holding_duration >= self.max_hold_steps and target_position != 0.0:
            target_position = 0.0
            pos_delta = abs(target_position - self.current_position)

        if self.current_position == 0.0 and target_position != 0.0:
            self.entry_step = self.current_step
            self.last_trade_step = self.current_step

        current_price = self.df.iloc[self.current_step]["close"]
        self.current_step += 1
        next_price = self.df.iloc[min(self.current_step, self.max_steps)]["close"]

        price_diff = next_price - current_price
        trade_allocation = self.current_position * self.max_trade_allocation
        raw_pnl = (trade_allocation / (current_price + 1e-8)) * price_diff

        turnover_value = pos_delta * self.max_trade_allocation
        friction_cost = (turnover_value * self.variable_friction_rate) + (self.fixed_order_fee if pos_delta > 1e-3 else 0.0)

        net_pnl = raw_pnl - friction_cost
        self.current_position = target_position
        self.equity += net_pnl

        # Multi-Horizon Forward Return Tracking
        forward_returns = {f"fwd_ret_{h}m": (self.df.iloc[min(self.current_step + h, self.max_steps)]["close"] - current_price) / (current_price + 1e-8) for h in [1, 5, 15, 30, 60]}

        state_record = {
            "timestamp_step": self.current_step,
            "action_raw": action_val,
            "position": self.current_position,
            "net_pnl": net_pnl,
            "equity": self.equity,
            "friction_cost": friction_cost,
            **forward_returns
        }

        daily_loss = self.day_start_equity - self.equity
        terminated = bool(self.current_step >= self.max_steps)
        truncated = bool(daily_loss >= self.daily_drawdown_limit or self.equity < (self.initial_capital * 0.70))

        reward = net_pnl / 100.0
        normalized_reward = float(np.clip(reward, -10.0, 10.0))

        info = {"pnl": net_pnl, "equity": self.equity, "state_record": state_record}
        return self._get_observation(), normalized_reward, terminated, truncated, info
