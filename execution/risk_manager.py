from config import TradingConfig


class RiskManager:
    """
    Independent pre-trade compliance guardrail. Intercepts policy
    actions and forces position liquidation if risk thresholds are breached.
    """
    def __init__(self, config: TradingConfig = TradingConfig()):
        self.cfg = config
        self.peak_equity = config.initial_capital
        self.kill_switch_active = False

    def validate_action(self, proposed_action: float, current_equity: float, current_pos: float) -> float:
        # Update Peak Equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Compute Current Drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        # Breach of 30% Drawdown Limit -> Engage Kill-Switch
        if drawdown >= self.cfg.max_drawdown_limit or self.kill_switch_active:
            self.kill_switch_active = True
            return -current_pos  # Force immediate flat exit

        # Position Limit Cap
        target_pos = current_pos + proposed_action
        if abs(target_pos) > self.cfg.max_position_size:
            allowed_action = np.sign(proposed_action) * (self.cfg.max_position_size - abs(current_pos))
            return float(allowed_action)

        return float(proposed_action)
