from dataclasses import dataclass


@dataclass(frozen=True)
class TradingConfig:
    # Portfolio & Risk Limits
    initial_capital: float = 50000.0
    max_drawdown_limit: float = 0.30  # 30% Hard Kill-Switch
    max_position_size: float = 5.0    # Max contracts
    sebi_stt_rate: float = 0.00125    # Securities Transaction Tax
    sebi_slippage_bps: float = 1.8    # 1.8 bps per execution

    # Architecture & RL Parameters
    seq_len: int = 60                 # 60-bar lookback window
    gamma: float = 0.99               # Discount factor
    gae_lambda: float = 0.95          # GAE parameter
    clip_eps: float = 0.12            # PPO clip ratio
    lr: float = 3e-6
    entropy_floor: float = 0.02

    # Data & Paths
    db_path: str = "data/duckdb/market_data.duckdb"
    telemetry_path: str = "logs/experiments/trade_telemetry.parquet"
    factor_ledger_path: str = "logs/experiments/factor_ledger.parquet"
