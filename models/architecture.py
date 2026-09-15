# models/architecture.py

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DualAlphaTCN(nn.Module):
    def __init__(self, in_channels: int, num_channels=[32, 64, 32], kernel_size=3, dropout=0.2):
        super(DualAlphaTCN, self).__init__()
        self.tcn_nifty = TemporalConvNet(in_channels, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn_banknifty = TemporalConvNet(in_channels, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.fc_nifty = nn.Sequential(
            nn.Linear(num_channels[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        self.fc_banknifty = nn.Sequential(
            nn.Linear(num_channels[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x_nifty, x_banknifty):
        # x input shape: [Batch, Channels, Seq_Len]
        feat_nifty = self.tcn_nifty(x_nifty)[:, :, -1]
        feat_banknifty = self.tcn_banknifty(x_banknifty)[:, :, -1]

        z_nifty = self.fc_nifty(feat_nifty)
        z_banknifty = self.fc_banknifty(feat_banknifty)

        return z_nifty, z_banknifty
