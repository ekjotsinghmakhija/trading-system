import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class CrossAssetAttention(nn.Module):
    def __init__(self, feature_dim: int):
        super(CrossAssetAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5

    def forward(self, nifty_feat: torch.Tensor, banknifty_feat: torch.Tensor) -> torch.Tensor:
        q = self.query(nifty_feat)
        k = self.key(banknifty_feat)
        v = self.value(banknifty_feat)

        attn_weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        fused_context = torch.matmul(attn_weights, v)
        return fused_context

class DualAlphaTCN(nn.Module):
    """
    Dual-Asset Signal Architecture using Temporal Convolutions and Cross-Asset Fusion.
    Outputs:
        z_nifty: Continuous signal [-1.0, +1.0]
        z_banknifty: Continuous signal [-1.0, +1.0]
    """
    def __init__(self, in_channels: int, num_channels: list = [32, 64, 32], kernel_size: int = 3, dropout: float = 0.2):
        super(DualAlphaTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size-1)*dilation_size, dropout=dropout))

        self.tcn = nn.Sequential(*layers)
        self.cross_attn = CrossAssetAttention(num_channels[-1])

        self.nifty_head = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.banknifty_head = nn.Sequential(
            nn.Linear(num_channels[-1] * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x_nifty: torch.Tensor, x_banknifty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_nifty = self.tcn(x_nifty)[:, :, -1]
        h_banknifty = self.tcn(x_banknifty)[:, :, -1]

        fused_nifty = self.cross_attn(h_nifty, h_banknifty)
        fused_banknifty = self.cross_attn(h_banknifty, h_nifty)

        nifty_concat = torch.cat([h_nifty, fused_nifty], dim=-1)
        banknifty_concat = torch.cat([h_banknifty, fused_banknifty], dim=-1)

        z_nifty = self.nifty_head(nifty_concat)
        z_banknifty = self.banknifty_head(banknifty_concat)

        return z_nifty, z_banknifty
