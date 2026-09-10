import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticTCNGRU(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 3):
        super().__init__()
        # Feature Extractor: 1D Conv -> GRU
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(64, 128, batch_first=True)

        # Policy Head (3 discrete logits: 0 = Short, 1 = Flat, 2 = Long)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Value Head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for Conv1d if required: [batch, features, channels]
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        c_out = self.relu(self.conv1(x)).transpose(1, 2)
        g_out, _ = self.gru(c_out)
        return g_out[:, -1, :]

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.forward(obs)
        logits = self.actor(features)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy()
        values = self.critic(features)

        return log_prob, entropy, values
