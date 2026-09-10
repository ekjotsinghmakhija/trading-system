import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticTCNGRU(nn.Module):
    """
    Combined Actor-Critic network with high action exploration standard deviation.
    """
    def __init__(self, input_dim: int, action_dim: int = 1):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Actor Network
        self.actor_dense = nn.Linear(128, action_dim)
        # log_std set to -0.8 (~0.45 std dev) to break dead-zone inertia
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.8)

        # Critic Network
        self.critic_dense = nn.Linear(128, 1)

    def forward(self, x):
        features = self.feature_net(x)
        return features

    def get_action(self, obs_tensor, deterministic=False):
        features = self.forward(obs_tensor)
        mean = torch.tanh(self.actor_dense(features))
        value = self.critic_dense(features)

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
        mean = torch.tanh(self.actor_dense(features))
        value = self.critic_dense(features)

        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action_tensor).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        return log_prob, entropy, value
