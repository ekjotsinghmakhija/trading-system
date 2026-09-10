import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ChokuTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if out.shape[-1] != res.shape[-1]:
            out = out[..., :res.shape[-1]]

        return self.relu(out + res)


class ActorCriticTCNGRU(nn.Module):
    def __init__(self, input_dim: int = None, action_dim: int = 1):
        super().__init__()
        self.action_dim = action_dim

        # Temporal Sequence Extractor
        self.tcn = ChokuTemporalBlock(in_channels=1, out_channels=32, kernel_size=3, dilation=1)
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)

        # Dynamic Dense Layer - Automatically infers input vector length (handles N features + 1 position)
        self.fc_features = nn.Sequential(
            nn.LazyLinear(64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Actor & Critic Heads
        self.actor_dense = nn.Linear(64 + 64, 64)
        self.actor_mean = nn.Linear(64, action_dim)
        # Low noise sampling initialization (std ~ 0.082)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.5)

        self.critic_dense = nn.Linear(64 + 64, 64)
        self.critic_value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Sequence Path
        x_trans = x.unsqueeze(1)
        tcn_out = self.tcn(x_trans)
        tcn_permuted = tcn_out.permute(0, 2, 1)

        gru_out, _ = self.gru(tcn_permuted)
        gru_feat = gru_out[:, -1, :]

        # Dense Path (Dynamic Input Shape Adaptor)
        dense_feat = self.fc_features(x)
        combined = torch.cat([gru_feat, dense_feat], dim=-1)

        # Actor and Value Heads
        act_hidden = F.relu(self.actor_dense(combined))
        action_mean = torch.tanh(self.actor_mean(act_hidden))

        crit_hidden = F.relu(self.critic_dense(combined))
        state_value = self.critic_value(crit_hidden)

        return action_mean, state_value

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        action_mean, value = self.forward(x)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        action = action_mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        action_mean, values = self.forward(x)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, values.view(-1)
