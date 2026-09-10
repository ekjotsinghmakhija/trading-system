import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer to ensure model predictions at time t
    depend only on past and present observations (t' <= t), preventing lookahead bias.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad left along sequence dimension and slice off extra trailing steps
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Residual Causal TCN Block for temporal feature extraction in financial time-series.
    Employs dual Causal Convolutions with residual downsampling connection.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        """Kaiming Normal Initialization optimized for ReLU activations."""
        nn.init.kaiming_normal_(self.conv1.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.conv.weight, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class ActorCriticTCNGRU(nn.Module):
    """
    Hybrid TCN-GRU Architecture for PPO Intraday Options Strategy.

    Structure:
      1. TCN Block: Extracts multi-scale local temporal patterns.
      2. GRU Layer: Captures long-term sequential dependencies.
      3. Bottleneck Layer: Dense layer with LayerNorm for stable representations.
      4. Dual Heads:
         - Actor Head: Continuous action policy mean (tanh bounded [-1, 1]).
         - Critic Head: State value estimation V(s).
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 60,
        tcn_channels: int = 64,
        gru_hidden: int = 128,
        action_dim: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.action_dim = action_dim

        logger.info(
            f"Initializing ActorCriticTCNGRU | input_dim={input_dim}, seq_len={seq_len}, "
            f"tcn_channels={tcn_channels}, gru_hidden={gru_hidden}, action_dim={action_dim}"
        )

        # 1. TCN Feature Extractor
        self.tcn = TemporalBlock(
            in_channels=input_dim,
            out_channels=tcn_channels,
            kernel_size=3,
            stride=1,
            dilation=1
        )

        # 2. Sequential Memory GRU Layer
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # 3. Dense Bottleneck Representation Layer
        self.shared_dense = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 4. Policy (Actor) & Value (Critic) Output Heads
        self.actor_dense = nn.Linear(64, action_dim)
        self.critic_dense = nn.Linear(64, 1)

        # Trainable log standard deviation for continuous action exploration
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self._initialize_heads()

    def _initialize_heads(self):
        """Orthogonal initialization on output heads to prevent policy saturation traps."""
        logger.debug("Applying Orthogonal Initialization to Policy and Value heads...")

        # Policy output initialized near zero mean with low variance gain (0.01)
        nn.init.orthogonal_(self.actor_dense.weight, gain=0.01)
        nn.init.constant_(self.actor_dense.bias, 0.0)

        # Critic value output standard unit variance gain (1.0)
        nn.init.orthogonal_(self.critic_dense.weight, gain=1.0)
        nn.init.constant_(self.critic_dense.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward Pass.

        Args:
            x: Input tensor of shape [Batch, Seq_Len, Features] or [Seq_Len, Features]

        Returns:
            action_mean: Tensor of shape [Batch, Action_Dim] mapped to [-1, 1]
            action_log_std: Expanded log standard deviation tensor [Batch, Action_Dim]
            state_value: Critic state value estimation V(s) [Batch, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Reshape to [Batch, Features, Seq_Len] for 1D convolution
        x_trans = x.transpose(1, 2)

        # Process through TCN
        tcn_out = self.tcn(x_trans)
        tcn_out = tcn_out.transpose(1, 2)  # Back to [Batch, Seq_Len, Channels]

        # Process through GRU
        gru_out, _ = self.gru(tcn_out)
        last_step_features = gru_out[:, -1, :]  # Extract last hidden temporal state

        # Shared feature representations
        shared_rep = self.shared_dense(last_step_features)

        # Actor and Critic head predictions
        action_mean = torch.tanh(self.actor_dense(shared_rep))
        state_value = self.critic_dense(shared_rep)

        # Match batch dimension for action distribution log_std
        action_log_std = self.log_std.expand_as(action_mean)

        return action_mean, action_log_std, state_value

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from Gaussian distribution parameterized by policy network outputs.

        Args:
            x: Observation state input tensor
            deterministic: If True, returns policy mean without exploration noise

        Returns:
            action: Sampled bounded action tensor [-1.0, 1.0]
            log_prob: Log probability density of sampled action
            state_value: Critic state value V(s)
        """
        action_mean, action_log_std, state_value = self.forward(x)

        if deterministic:
            return action_mean, torch.zeros_like(action_mean), state_value

        std = torch.exp(action_log_std)
        dist = Normal(action_mean, std)

        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        bounded_action = torch.clamp(raw_action, -1.0, 1.0)

        return bounded_action, log_prob, state_value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate batch of actions for PPO policy loss updates.

        Returns:
            log_prob: Log probability of input actions
            entropy: Entropy of current policy distribution
            state_value: Estimated state values V(s)
        """
        action_mean, action_log_std, state_value = self.forward(x)
        std = torch.exp(action_log_std)
        dist = Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, state_value
