import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticTCNGRU(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 1):
        super().__init__()

        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.gru = nn.GRU(64, 128, batch_first=True)

        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.8)

        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_trans = x.transpose(1, 2)
        tcn_out = self.tcn(x_trans).transpose(1, 2)
        gru_out, _ = self.gru(tcn_out)
        return gru_out[:, -1, :]

    def get_action(self, obs_tensor, deterministic=False):
        features = self.forward(obs_tensor)
        mean = torch.tanh(self.actor_head(features))
        value = self.critic_head(features)

        if deterministic:
            return mean, torch.tensor([0.0], device=obs_tensor.device), value

        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs_tensor, action_tensor):
        features = self.forward(obs_tensor)
        mean = torch.tanh(self.actor_head(features))
        value = self.critic_head(features)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        return dist.log_prob(action_tensor).sum(axis=-1), dist.entropy().sum(axis=-1), value
