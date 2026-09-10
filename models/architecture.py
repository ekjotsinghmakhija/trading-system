import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Removes right-side padding to guarantee causal time-series convolutions."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        # LayerNorm over channels (C, L) instead of BatchNorm1d
        self.norm1 = nn.GroupNorm(1, n_outputs)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.GroupNorm(1, n_outputs)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.norm1,
            self.conv2, self.chomp2, self.relu2, self.norm2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ActorCriticTCNGRU(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(input_dim, hidden_dim, kernel_size=3, stride=1, dilation=1, padding=2),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, dilation=2, padding=4)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Policy & Value Heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, features) or (batch_size, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq_len dimension -> (batch_size, 1, features)

        x_trans = x.transpose(1, 2)  # (batch_size, features, seq_len)
        tcn_out = self.tcn(x_trans).transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        gru_out, _ = self.gru(tcn_out)

        # Return last hidden state
        return gru_out[:, -1, :]

    @torch.no_grad()
    def get_action(self, obs_tensor):
        self.eval()  # Set model to evaluation mode for inference/rollouts
        features = self.forward(obs_tensor)
        action_mean = self.actor(features)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(features)

        return action, log_prob, value

    def evaluate_actions(self, obs_batch, action_batch):
        features = self.forward(obs_batch)
        action_mean = self.actor(features)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)

        log_prob = dist.log_prob(action_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features)

        return log_prob, entropy, value
