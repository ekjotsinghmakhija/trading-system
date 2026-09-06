# Quant System v2.0 - Standard Architecture

## 1. Directory Structure
/data
  /raw             # Immutable raw tick/OHLCV data (US & NSE)
  /processed       # Stationarity-transformed, purged datasets
/features
  engineering.md   # Feature formulation rules
/env
  strict_sim.md    # Gym environment specifications (+10% friction)
/models
  /checkpoints     # Saved PPO/SAC weights
/eval
  metrics.md       # CPCV and walk-forward validation rules

## 2. Core Principles
1. Zero Lookahead: Features at time `t` use exclusively `t-1` and backward.
2. Stationarity Mandatory: Neural networks only ingest stationary, bounded vectors.
3. Strictness Buffer: Simulation friction is statically multiplied by 1.10.
4. Drawdown-First: Reward functions heavily penalize equity curve degradation.
