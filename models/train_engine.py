# models/train_engine.py

import torch
import torch.nn as nn


class DifferentialSharpeLoss(nn.Module):
    """
    Numerically Stable Differentiable Sharpe Loss with Signal and Target Standardization.
    Prevents zero-gradient saturation and vanishing loss scaling across micro-batches.
    """
    def __init__(self, eps: float = 1e-6):
        super(DifferentialSharpeLoss, self).__init__()
        self.eps = eps

    def forward(self, signals: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        signals = signals.view(-1)
        target_returns = target_returns.view(-1)

        # Standardize target returns per batch to scale loss gradients
        target_std = torch.std(target_returns) + self.eps
        scaled_targets = target_returns / target_std

        portfolio_returns = signals * scaled_targets

        mean_ret = torch.mean(portfolio_returns)
        std_ret = torch.std(portfolio_returns) + self.eps

        sharpe_ratio = mean_ret / std_ret
        return -sharpe_ratio
