import os
import logging
import duckdb
import torch
import numpy as np
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

    returns = advantages + values
    return advantages, returns


def run_production_pipeline():
    logger.info("Initializing Quantum Engine Production Loop...")
    os.makedirs("checkpoints", exist_ok=True)

    db_path = "data/duckdb/market_data.duckdb"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}. Run data/process_raw.py first.")

    conn = duckdb.connect(db_path)
    df_raw = conn.execute("SELECT * FROM ohlcv_bars ORDER BY timestamp ASC").df()
    conn.close()
    logger.info(f"Loaded {len(df_raw):,} market bars into memory.")

    logger.info("Computing alpha feature matrix...")
    engineer = FeatureEngineer()
    df_features = engineer.compute_18_alpha_matrix(df_raw).dropna().reset_index(drop=True)

    exclude_cols = {'timestamp', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'oi'}
    feature_cols = [c for c in df_features.columns if c.lower() not in exclude_cols]
    log_returns = df_features['log_return'].values

    # Normalize observations
    raw_obs = torch.tensor(df_features[feature_cols].values, dtype=torch.float32)
    obs_clean = torch.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0)
    mean = obs_clean.mean(dim=0, keepdim=True)
    std = obs_clean.std(dim=0, keepdim=True) + 1e-8
    obs_data = torch.clamp((obs_clean - mean) / std, min=-5.0, max=5.0)

    input_dim = len(feature_cols)
    model = ActorCriticTCNGRU(input_dim=input_dim, action_dim=1)
    trainer = HighThroughputPPOTrainer(model=model, lr=1e-5)
    device = trainer.device

    logger.info("Starting Full Production Reinforcement Learning Loop...")

    rollout_horizon = 4096
    num_epochs = 10
    total_steps = len(obs_data) - 1
    cost_bps = 0.0003  # 3 bps transaction cost penalty
    best_pnl = -float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        total_epoch_pnl = 0.0
        actions_list = []

        for start_idx in range(0, total_steps - rollout_horizon, rollout_horizon):
            end_idx = start_idx + rollout_horizon
            obs_batch = obs_data[start_idx:end_idx].to(device)
            future_returns = torch.tensor(log_returns[start_idx+1:end_idx+1], dtype=torch.float32, device=device)

            # 1. Collect Trajectory Rollouts from Current Policy
            model.eval()
            with torch.no_grad():
                features = model(obs_batch)
                values = model.critic(features).squeeze(-1)

                # Predict next value for GAE terminal step
                next_obs = obs_data[end_idx:end_idx+1].to(device)
                next_val = model.critic(model(next_obs)).squeeze(-1)

                action_mean = model.actor(features)
                std = torch.exp(model.log_std)
                dist = torch.distributions.Normal(action_mean, std)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(dim=-1)

            # 2. Compute Cost-Adjusted Trading Rewards ($r_t = a_t \cdot R_{t+1} - \text{cost} \cdot |\Delta a_t|$)
            position = torch.clamp(actions.squeeze(-1), -1.0, 1.0)
            position_change = torch.cat([torch.tensor([0.0], device=device), torch.abs(position[1:] - position[:-1])])
            rewards = (position * future_returns) - (cost_bps * position_change)

            # 3. Calculate GAE Advantages and Returns-to-Go
            advantages, returns_to_go = compute_gae(rewards, values, next_val)
            adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 4. Perform PPO Optimization Step
            loss = trainer.train_epoch_amp(obs_batch.cpu(), actions.cpu(), log_probs.cpu(), adv_normalized.cpu(), returns_to_go.cpu())

            epoch_loss += loss
            batch_count += 1
            total_epoch_pnl += rewards.sum().item()
            actions_list.append(position.cpu().numpy())

        avg_loss = epoch_loss / max(1, batch_count)
        cum_pnl_pct = total_epoch_pnl * 100.0
        all_actions = np.concatenate(actions_list)
        avg_pos = np.mean(np.abs(all_actions))

        logger.info(
            f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {avg_loss:.5f} | Cum PnL: {cum_pnl_pct:+.2f}% | Avg Exposure: {avg_pos:.2f}"
        )

        # Save Best Model Checkpoint
        if cum_pnl_pct > best_pnl:
            best_pnl = cum_pnl_pct
            checkpoint_path = "checkpoints/actor_critic_nifty_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_pnl': best_pnl
            }, checkpoint_path)
            logger.info(f"Saved new best model checkpoint to {checkpoint_path}")

    logger.info("Production Pipeline Execution Complete. Model ready for inference.")


if __name__ == "__main__":
    run_production_pipeline()
