import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Removes trailing padding to preserve strict causal temporal sequence constraints."""
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalTemporalBlock(nn.Module):
    """Dilated 1D Causal Convolutional Block with BatchNorm, Dropout, and Residual Connection."""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.15):
        super(TemporalTemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvolutionalNetwork(nn.Module):
    """3-Layer TCN with exponentially increasing dilations (d = 1, 2, 4)."""
    def __init__(self, num_inputs: int, num_channels: list = [64, 64, 64], kernel_size: int = 3, dropout: float = 0.15):
        super(TemporalConvolutionalNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalTemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size, padding=padding, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape expected: (Batch, Features, Sequence_Length)
        return self.network(x)

class ActorCriticTCNGRU(nn.Module):
    """
    Combined TCN + GRU Actor-Critic Policy & Value Estimator.
    Processes (Batch, 60_bars, D_features) tensors with < 4ms execution overhead.
    """
    def __init__(self, input_dim: int, hidden_gru_dim: int = 128, num_actions: int = 1):
        super(ActorCriticTCNGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_gru_dim = hidden_gru_dim

        # 1. Temporal Feature Extraction (TCN)
        self.tcn = TemporalConvolutionalNetwork(num_inputs=input_dim, num_channels=[64, 64, 64], kernel_size=3, dropout=0.15)

        # 2. Sequential Context Tracker (GRU)
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_gru_dim, num_layers=1, batch_first=True)

        # 3. Actor (Policy) Head
        self.actor_dense = nn.Linear(hidden_gru_dim, 64)
        self.actor_out = nn.Linear(64, num_actions)

        # 4. Critic (Value) Head
        self.critic_dense = nn.Linear(hidden_gru_dim, 64)
        self.critic_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, h_state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        x input shape: (Batch, Seq_Len=60, Features=D)
        returns: (action_a_t [-1, 1], state_value_v_t, next_h_state)
        """
        batch_size = x.size(0)

        # Reshape for 1D Conv TCN: (Batch, Features, Seq_Len)
        x_tcn_in = x.transpose(1, 2)
        tcn_out = self.tcn(x_tcn_in)

        # Reshape back for GRU: (Batch, Seq_Len, Channels=64)
        gru_in = tcn_out.transpose(1, 2)

        if h_state is None:
            h_state = torch.zeros(1, batch_size, self.hidden_gru_dim, device=x.device)

        gru_out, h_next = self.gru(gru_in, h_state)

        # Take last time step embedding from GRU output
        last_step_repr = gru_out[:, -1, :]

        # Compute Action Output a_t bounded by Tanh
        actor_h = F.relu(self.actor_dense(last_step_repr))
        action = torch.tanh(self.actor_out(actor_h))

        # Compute Critic State Value Output V(s_t)
        critic_h = F.relu(self.critic_dense(last_step_repr))
        state_value = self.critic_out(critic_h)

        return action, state_value, h_next

if __name__ == "__main__":
    # Latency & Shape Verification Benchmark
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Benchmarking network on device: {device}")

    batch_size = 32
    seq_len = 60
    num_features = 18

    # Create dummy observation batch
    dummy_input = torch.randn(batch_size, seq_len, num_features, device=device)
    model = ActorCriticTCNGRU(input_dim=num_features).to(device)
    model.eval()

    # Measure execution latency
    start_time = time.perf_counter()
    with torch.no_grad():
        actions, values, _ = model(dummy_input)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"[✓] Forward pass complete in {elapsed_ms:.2f} ms")
    print(f"    Actions Tensor Shape : {actions.shape} (Range: [{actions.min():.2f}, {actions.max():.2f}])")
    print(f"    Values Tensor Shape  : {values.shape}")
