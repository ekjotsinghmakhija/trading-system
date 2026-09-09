import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class StrictOptionSimEnv(gym.Env):
    """
    Gymnasium environment for Indian Index Options intraday trading.
    - Observation Space: (60, num_features) normalized sequence matrix
    - Action Space: Continuous scalar in [-1.0, +1.0]
        * action >  0.25: Go Long Call (Bullish)
        * action < -0.25: Go Long Put (Bearish)
        * -0.25 <= action <= 0.25: Flat / Close Position
    - Enforces max holding time of 60 minutes, STT/brokerage friction, and intraday square-off.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, feature_cols: list, initial_capital: float = 100000.0, sequence_length: int = 60):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.sequence_length = sequence_length
        self.num_features = len(feature_cols)

        # Environment Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sequence_length, self.num_features), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal Execution Constants
        self.LOT_SIZE = 25  # Nifty option lot size
        self.STT_PREMIUM_RATE = 0.000625  # STT 0.0625% on sell side premium
        self.BROKERAGE_PER_ORDER = 20.0  # ₹20 flat per order
        self.MAX_HOLDING_MINUTES = 60

        # State Variables
        self.current_step = 0
        self.capital = initial_capital
        self.position = 0  # 0: Flat, +1: Long Call, -1: Long Put
        self.entry_price = 0.0
        self.entry_step = 0
        self.returns_history = []

    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        obs = self.df.iloc[start_idx:end_idx][self.feature_cols].values
        return obs.astype(np.float32)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = self.sequence_length - 1
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_step = 0
        self.returns_history = []

        obs = self._get_observation()
        info = {"capital": self.capital, "position": self.position}
        return obs, info

    def _calculate_frictions(self, premium: float, is_exit: bool = False) -> float:
        """Calculates total transaction friction including turnover fees and STT."""
        turnover = premium * self.LOT_SIZE
        stt = turnover * self.STT_PREMIUM_RATE if is_exit else 0.0
        exchange_charges = turnover * 0.0005  # Exchange transaction fee
        return self.BROKERAGE_PER_ORDER + stt + exchange_charges

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        act_val = float(action[0])
        current_price = float(self.df.iloc[self.current_step]["close"])

        # Determine intended target direction
        target_pos = 0
        if act_val > 0.25:
            target_pos = 1
        elif act_val < -0.25:
            target_pos = -1

        step_pnl = 0.0
        forced_exit = False

        # Holding Cap Enforcer (60 Minutes Limit)
        if self.position != 0 and (self.current_step - self.entry_step) >= self.MAX_HOLDING_MINUTES:
            target_pos = 0
            forced_exit = True

        # End-of-Day Enforcer (Square off at 15:15 IST / step near end)
        if self.current_step >= len(self.df) - 2:
            target_pos = 0

        # Execute Order & Position Updates
        if target_pos != self.position:
            # 1. Close Existing Position
            if self.position != 0:
                raw_return = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
                raw_pnl = raw_return * self.LOT_SIZE
                friction = self._calculate_frictions(current_price, is_exit=True)
                step_pnl += (raw_pnl - friction)
                self.position = 0

            # 2. Open New Position
            if target_pos != 0:
                friction = self._calculate_frictions(current_price, is_exit=False)
                step_pnl -= friction
                self.position = target_pos
                self.entry_price = current_price
                self.entry_step = self.current_step

        elif self.position != 0:
            # Mark-to-market step return calculation
            prev_price = float(self.df.iloc[self.current_step - 1]["close"])
            price_diff = current_price - prev_price
            step_pnl += (price_diff * self.LOT_SIZE) if self.position == 1 else (-price_diff * self.LOT_SIZE)

        # Update Portfolio Capital
        self.capital += step_pnl
        step_return = step_pnl / self.capital
        self.returns_history.append(step_return)

        # Reward Calculation: Differential Sharpe Ratio Proxy
        ret_mean = np.mean(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.0
        ret_std = np.std(self.returns_history[-20:]) + 1e-6
        reward = float(step_return / ret_std)

        # Apply Forced Exit Penalty
        if forced_exit:
            reward -= 0.05

        # Progress Time Index
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.capital <= (self.initial_capital * 0.70)  # Stop trading if 30% drawdown reached

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "capital": self.capital,
            "position": self.position,
            "step_pnl": step_pnl,
            "forced_exit": forced_exit
        }

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # Test Environment Integration
    from features.feature_engineer import FeatureEngine

    dates = pd.date_range("2026-09-01 09:15:00", periods=200, freq="1min", tz="UTC")
    dummy_df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(200).cumsum() + 25000,
        "high": np.random.randn(200).cumsum() + 25020,
        "low": np.random.randn(200).cumsum() + 24980,
        "close": np.random.randn(200).cumsum() + 25000,
        "volume": np.random.randint(100, 5000, size=200)
    })

    engine = FeatureEngine(dummy_df)
    matrix = engine.build_feature_matrix()
    feature_cols = [c for c in matrix.columns if c.startswith("feat_")]

    env = StrictOptionSimEnv(matrix, feature_cols)
    obs, info = env.reset()
    print(f"[✓] Environment initialized successfully.")
    print(f"    Observation Shape: {obs.shape}")

    # Run a test step with positive long signal action
    next_obs, reward, term, trunc, info = env.step(np.array([0.8]))
    print(f"    Step Test Output -> Reward: {reward:.4f}, Capital: {info['capital']:.2f}, Position: {info['position']}")
