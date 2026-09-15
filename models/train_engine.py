# models/train_engine.py

import torch
import torch.nn as nn


class DifferentialSharpeLoss(nn.Module):
    """
    Numerically Stable Differentiable Sharpe Loss.
    """
    def __init__(self, drawdown_penalty_weight: float = 0.0, max_allowed_drawdown: float = 0.15, eps: float = 1e-4):
        super(DifferentialSharpeLoss, self).__init__()
        self.penalty_weight = drawdown_penalty_weight
        self.max_dd = max_allowed_drawdown
        self.eps = eps

    def forward(self, signals: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        portfolio_returns = signals.view(-1) * target_returns.view(-1)

        mean_return = torch.mean(portfolio_returns)
        var_return = torch.var(portfolio_returns, unbiased=False)
        std_return = torch.sqrt(var_return + self.eps)

        sharpe_ratio = mean_return / std_return

        if self.penalty_weight > 0.0:
            cum_returns = torch.cumsum(portfolio_returns, dim=0)
            running_max = torch.cummax(cum_returns, dim=0)[0]
            drawdowns = running_max - cum_returns
            max_drawdown = torch.max(drawdowns)
            drawdown_penalty = torch.square(torch.clamp(max_drawdown - self.max_dd, min=0.0))
            return -sharpe_ratio + (self.penalty_weight * drawdown_penalty)

        return -sharpe_ratio
