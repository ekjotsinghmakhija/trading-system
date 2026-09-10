import os
import logging
import duckdb
import torch
import numpy as np
from torch.distributions import Categorical
from models.architecture import ActorCriticTCNGRU
from models.train_engine import HighThroughputPPOTrainer
from features.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
logger = logging.getLogger("quantum_engine")


def compute_gae(rewards, values, next_value, gamma=0.99, gae_lambda=0.95):
    """Computes Generalized Advantage Estimation (GAE) and Returns-to-Go."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        v_next = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def run_production_pipeline():
    logger.info("Initializing Quantum Engine Institutional Alpha Pipeline...")
    os.makedirs("checkpoints", exist_ok=True)

    db_path = "data/duckdb/market_data.duckdb"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}.")

    conn = duckdb.connect(db_path)
    df_raw = conn.execute("SELECT * FROM ohlcv_bars ORDER BY timestamp ASC").df()
    conn.close()

    logger.info("Computing microstructural alpha feature matrix...")
    engineer = FeatureEngineer()
    df_features = engineer.compute_18_alpha_matrix(df_raw).dropna().reset_index(drop=True)

    exclude_cols = {'timestamp', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'oi'}
    feature_cols = [c for c in df_features.columns if c.lower() not in exclude_cols]
    log_returns = df_features['log_return'].values

    # Robust Feature Scaling (Median Absolute Deviation / Quantile Standardization)
    raw_obs = torch.tensor(df_features[feature_cols].values, dtype=torch.float32)
    obs_clean = torch.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)

    mean = obs_clean.mean(dim=0, keepdim=True)
    std = obs_clean.std(dim=0, keepdim=True) + 1e-8
    obs_data = torch.clamp((obs_clean - mean) / std, -5.0, 5.0)

    # Action Mapping: Index 0 -> Short (-1), Index 1 -> Flat (0), Index 2 -> Long (+1)
    action_map = torch.tensor([-1.0, 0.0, 1.0])
    model = ActorCriticTCNGRU(input_dim=len(feature_cols), action_dim=3)
    trainer = HighThroughputPPOTrainer(model=model, lr=3e-5)
    device = trainer.device
    action_map = action_map.to(device)

    rollout_horizon = 4096
    num_epochs = 10
    total_steps = len(obs_data) - 1
    cost_bps = 0.0001  # 1 bp transaction friction
    best_sharpe = -float("inf")

    logger.info("Starting Institutional Alpha RL Optimization Loop...")

    for epoch in range(1, num_epochs + 1):
        epoch_loss, batch_count = 0.0, 0
        all_step_rewards = []

        for start_idx in range(0, total_steps - rollout_horizon, rollout_horizon):
            end_idx = start_idx + rollout_horizon
            obs_batch = obs_data[start_idx:end_idx].to(device)
            future_returns = torch.tensor(log_returns[start_idx+1:end_idx+1], dtype=torch.float32, device=device)

            model.eval()
            with torch.no_grad():
                features = model(obs_batch)
                logits = model.actor(features)
                values = model.critic(features).squeeze(-1)
                next_val = model.critic(model(obs_data[end_idx:end_idx+1].to(device))).squeeze(-1)

                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            # Map Discrete Actions -> Net Positions
            positions = action_map[actions]

            # Calculate Transaction Friction
            pos_diff = torch.cat([torch.tensor([0.0], device=device), torch.abs(positions[1:] - positions[:-1])])
            rewards = (positions * future_returns) - (cost_bps * pos_diff)

            # GAE Calculation
            advantages, returns_to_go = compute_gae(rewards, values, next_val)
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Train Batch
            loss = trainer.train_epoch_amp(
                obs_batch.cpu(),
                actions.unsqueeze(-1).cpu(),
                log_probs.cpu(),
                adv_norm.cpu(),
                returns_to_go.cpu()
            )
            epoch_loss += loss
            batch_count += 1
            all_step_rewards.append(rewards.cpu().numpy())

        # Metric Reporting
        step_returns = np.concatenate(all_step_rewards)
        mean_ret = np.mean(step_returns)
        std_ret = np.std(step_returns) + 1e-8
        annualized_sharpe = (mean_ret / std_ret) * np.sqrt(252 * 375)

        avg_loss = epoch_loss / max(1, batch_count)
        logger.info(
            f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {avg_loss:.5f} | Step Return: {mean_ret*1e4:+.2f} bps | Sharpe: {annualized_sharpe:+.2f}"
        )

        if annualized_sharpe > best_sharpe:
            best_sharpe = annualized_sharpe
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'sharpe': best_sharpe},
                "checkpoints/actor_critic_nifty_best.pt"
            )
            logger.info(f"Saved Checkpoint (Best Sharpe: {best_sharpe:.2f})")

    logger.info("Pipeline Execution Complete.")


if __name__ == "__main__":
    run_production_pipeline()
