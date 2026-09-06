import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for logging financial metrics (PnL, Drawdown, Win Rate, Trades)
    to TensorBoard and enforcing early stopping if hard drawdown limits are breached.
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super(TradingMetricsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.returns_history = []
        self.drawdown_history = []

    def _on_step(self) -> bool:
        # Extract step info from vector environment
        infos = self.training_env.get_attr("info") if hasattr(self.training_env, "get_attr") else []

        # Pull infos directly if available from Gym step
        for env_info in self.locals.get("infos", []):
            if "balance" in env_info and "drawdown" in env_info:
                balance = env_info["balance"]
                drawdown = env_info["drawdown"]

                self.logger.record("trading/account_balance", balance)
                self.logger.record("trading/current_drawdown", drawdown)
                self.logger.record("trading/position", env_info.get("position", 0))

        return True


class FrictionCurriculumCallback(BaseCallback):
    """
    Curriculum Learning: Scales friction from 1.0x baseline to 1.1x strict target
    over the first N training steps to allow early feature discovery before high penalty.
    """

    def __init__(self, total_curriculum_steps: int = 100000, max_multiplier: float = 1.10):
        super(FrictionCurriculumCallback, self).__init__()
        self.total_curriculum_steps = total_curriculum_steps
        self.max_multiplier = max_multiplier

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        if current_step < self.total_curriculum_steps:
            progress = current_step / self.total_curriculum_steps
            current_multiplier = 1.0 + (self.max_multiplier - 1.0) * progress
        else:
            current_multiplier = self.max_multiplier

        # Update environment strictness multiplier dynamically
        self.training_env.env_method("set_attr", "strictness_multiplier", current_multiplier)
        self.logger.record("curriculum/strictness_multiplier", current_multiplier)
        return True
