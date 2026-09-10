import torch
import torch.nn as nn

class DifferentialSharpeLoss(nn.Module):
    """
    Differentiable Portfolio Optimization Loss Function.
    Directly maximizes Sharpe Ratio of strategy returns while penalizing drawdowns exceeding threshold.
    """
    def __init__(self, drawdown_penalty_weight: float = 10.0, max_allowed_drawdown: float = 0.15, eps: float = 1e-6):
        super(DifferentialSharpeLoss, self).__init__()
        self.penalty_weight = drawdown_penalty_weight
        self.max_dd = max_allowed_drawdown
        self.eps = eps

    def forward(self, signals: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        """
        signals: [Batch, 1] continuous outputs in [-1, +1]
        target_returns: [Batch, 1] forward log-returns
        """
        portfolio_returns = signals * target_returns

        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + self.eps
        sharpe_ratio = mean_return / std_return

        # Differentiable Drawdown Penalty Computation
        cum_returns = torch.cumsum(portfolio_returns, dim=0)
        running_max = torch.cummax(cum_returns, dim=0)[0]
        drawdowns = (running_max - cum_returns)
        max_drawdown = torch.max(drawdowns)

        drawdown_penalty = torch.square(torch.clamp(max_drawdown - self.max_dd, min=0.0))

        # Loss minimization objective
        loss = -sharpe_ratio + (self.penalty_weight * drawdown_penalty)
        return loss
