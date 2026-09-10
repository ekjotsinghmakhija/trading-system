import os
import time
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from models.architecture import ActorCriticTCNGRU
from env.strict_sim_env import StrictOptionSimEnv
from models.callbacks import EvaluationCallback

logger = logging.getLogger(__name__)


def train_ppo_engine(
    total_timesteps: int = 12_000_000,
    eval_interval: int = 100_000,
    exp_name: str = "ppo_12m"
):
    """
    Stabilized High-Throughput PPO Training Engine.
    Scales rewards cleanly and bounds policy variance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = int(time.time())
    run_id = f"{exp_name}_{timestamp}"
    log_dir = os.path.join("logs", "experiments", run_id)
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    print("=" * 60)
    print(f"[🚀] Launching Training Engine on Device: {device}")
    print(f"     Experiment Logs: {log_dir}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60 + "\n")

    processed_data_path = "data/processed/nifty_options_features.parquet"
    if not os.path.exists(processed_data_path):
        processed_data_path = "data/processed/features.parquet"

    if os.path.exists(processed_data_path):
        df = pd.read_parquet(processed_data_path)
    else:
        dates = pd.date_range("2024-01-01", periods=10000, freq="1min")
        df = pd.DataFrame({
            "close": np.sin(np.linspace(0, 100, 10000)) * 10 + 100,
            "feature_1": np.random.randn(10000),
            "feature_2": np.random.randn(10000),
            "feature_3": np.random.randn(10000),
        })

    feature_cols = [col for col in df.columns if col not in ["datetime", "date", "timestamp"]]
    env = StrictOptionSimEnv(df=df, feature_cols=feature_cols)

    input_dim = len(feature_cols)
    model = ActorCriticTCNGRU(input_dim=input_dim).to(device)

    # Learning rate schedule
    initial_lr = 3e-5
    min_lr = 1e-6
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4, eps=1e-5)

    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    batch_size = 2048
    n_epochs = 10
    rollout_steps = 4096

    eval_callback = EvaluationCallback(
        eval_env=env,
        model=model,
        device=device,
        eval_interval=eval_interval,
        checkpoint_dir=checkpoint_dir,
        writer=writer
    )

    obs, _ = env.reset()
    global_step = 0

    while global_step < total_timesteps:
        lr_now = max(min_lr, initial_lr - (initial_lr - min_lr) * (global_step / total_timesteps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        model.eval()
        for _ in range(rollout_steps):
            global_step += 1
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = model.get_action(obs_tensor, deterministic=False)

            action_np = action.cpu().numpy()[0]
            next_obs, raw_reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Reward Scaling: Normalizes raw monetary values down to reasonable RL scales [-5.0, 5.0]
            scaled_reward = float(np.clip(raw_reward / 100.0, -5.0, 5.0))

            obs_buf.append(obs)
            act_buf.append(action_np)
            logp_buf.append(log_prob.cpu().numpy()[0])
            rew_buf.append(scaled_reward)
            val_buf.append(value.cpu().numpy()[0][0])
            done_buf.append(done)

            obs = next_obs
            if done:
                obs, _ = env.reset()

            if global_step % eval_interval == 0:
                eval_callback.run_evaluation(global_step)

        # Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, last_val = model.get_action(last_obs_tensor)
            last_val = last_val.cpu().numpy()[0][0]

        advantages = np.zeros(rollout_steps, dtype=np.float32)
        returns = np.zeros(rollout_steps, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(rollout_steps)):
            next_val = last_val if t == rollout_steps - 1 else val_buf[t + 1]
            next_non_terminal = 1.0 - float(done_buf[t])
            delta = rew_buf[t] + gamma * next_val * next_non_terminal - val_buf[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + val_buf[t]

        b_obs = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        b_act = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=device)
        b_logp = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        b_adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # Advantage Normalization
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Optimization Pass
        model.train()
        dataset_size = rollout_steps
        indices = np.arange(dataset_size)

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp = b_logp[mb_idx]
                mb_adv = b_adv[mb_idx]
                mb_ret = b_ret[mb_idx]

                new_logp, entropy, new_val = model.evaluate_actions(mb_obs, mb_act)

                ratios = torch.exp(new_logp - mb_logp)
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(new_val.squeeze(-1), mb_ret)
                entropy_loss = -entropy.mean()

                total_loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        writer.add_scalar("train/learning_rate", lr_now, global_step)
        writer.add_scalar("train/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("train/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("train/entropy", -entropy_loss.item(), global_step)

    writer.close()
    print("[✓] Training Finished.")


def train_ppo_12m_steps():
    train_ppo_engine(total_timesteps=12_000_000, eval_interval=100_000, exp_name="ppo_12m")
