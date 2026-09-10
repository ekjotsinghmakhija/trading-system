import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ChokuTemporalBlock(nn.Module):
    """
    Dilated Temporal Convolutional Block with Residual Connection.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, sequence_length)
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        # Trim padding overhang to maintain exact sequence dimension
        if out.shape[-1] != res.shape[-1]:
            out = out[..., :res.shape[-1]]

        return self.relu(out + res)


class ActorCriticTCNGRU(nn.Module):
    """
    Hybrid Actor-Critic Network with Temporal Convolutional Networks (TCN)
    and Gated Recurrent Units (GRU) for Feature Representation.
    """
    def __init__(self, input_dim: int, action_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # 1. Temporal Feature Extractor Stack
        self.tcn = ChokuTemporalBlock(in_channels=1, out_channels=32, kernel_size=3, dilation=1)
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)

        # 2. Dense Feature Representation
        self.fc_features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 3. Policy (Actor) Head
        self.actor_dense = nn.Linear(64 + 64, 64)
        self.actor_mean = nn.Linear(64, action_dim)
        # Initialize log_std to -1.5 for controlled action Gaussian spread
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.5)

        # 4. Value (Critic) Head
        self.critic_dense = nn.Linear(64 + 64, 64)
        self.critic_value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        # Enforce 2D tensor layout: (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        # --- Temporal Convolutional Pass ---
        # Shape: (batch_size, 1, input_dim)
        x_trans = x.unsqueeze(1)
        tcn_out = self.tcn(x_trans)  # Shape: (batch_size, 32, input_dim)

        # Permute for GRU: (batch_size, input_dim, 32)
        tcn_permuted = tcn_out.permute(0, 2, 1)
        gru_out, _ = self.gru(tcn_permuted)  # Shape: (batch_size, input_dim, 64)

        # Take the last sequence output across batch items
        gru_feat = gru_out[:, -1, :]  # Shape: (batch_size, 64)

        # --- Dense Feature Pass ---
        dense_feat = self.fc_features(x)  # Shape: (batch_size, 64)

        # Feature Concatenation: (batch_size, 128)
        combined = torch.cat([gru_feat, dense_feat], dim=-1)

        # --- Actor Output ---
        act_hidden = F.relu(self.actor_dense(combined))
        action_mean = torch.tanh(self.actor_mean(act_hidden))

        # --- Critic Output ---
        crit_hidden = F.relu(self.critic_dense(combined))
        state_value = self.critic_value(crit_hidden)  # Shape: (batch_size, 1)

        return action_mean, state_value

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        action_mean, value = self.forward(x)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        if deterministic:
            action = action_mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        # Forward pass across current batch
        action_mean, values = self.forward(x)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Flatten value prediction to match batch dimension (batch_size,)
        return log_prob, entropy, values.view(-1)
