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


def generate_rich_indicator_dataset(n_steps: int = 20000) -> pd.DataFrame:
    """
    Generates a rich dataset containing price trends, volatility, momentum,
    and technical indicators (RSI, MACD, Bollinger Bands, EMAs, ATR, Stoch).
    """
    np.random.seed(42)
    t = np.linspace(0, 200, n_steps)

    # Synthetic Base Price (Sinusoidal Trend + Random Walk)
    price = 100.0 + np.sin(t) * 15.0 + np.cumsum(np.random.randn(n_steps) * 0.2)
    df = pd.DataFrame({"close": price})

    # High / Low / Open estimation for ATR and Oscillators
    df["high"] = df["close"] + np.abs(np.random.randn(n_steps) * 0.5)
    df["low"] = df["close"] - np.abs(np.random.randn(n_steps) * 0.5)
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

    # 1. Exponential Moving Averages (EMA)
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # 2. Moving Average Convergence Divergence (MACD)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # 3. Relative Strength Index (RSI - 14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 4. Bollinger Bands (20, 2)
    rolling_mean_20 = df["close"].rolling(window=20).mean()
    rolling_std_20 = df["close"].rolling(window=20).std()
    df["bb_upper"] = rolling_mean_20 + (rolling_std_20 * 2)
    df["bb_lower"] = rolling_mean_20 - (rolling_std_20 * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (rolling_mean_20 + 1e-8)

    # 5. Stochastic Oscillator (%K, %D)
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14 + 1e-8))
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # 6. Average True Range (ATR - 14)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14).mean()

    # 7. Momentum & Rate of Change (ROC)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["roc_10"] = ((df["close"] - df["close"].shift(10)) / (df["close"].shift(10) + 1e-8)) * 100

    # Fill structural NaNs from rolling indicators
    df.bfill(inplace=True)
    df.fillna(0.0, inplace=True)

    return df


def train_ppo_engine(
    total_timesteps: int = 12_000_000,
    eval_interval: int = 20_000,  # LOG & EVAL EVERY 20k STEPS
    exp_name: str = "ppo_12m"
):
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
    print(f"     Evaluation Frequency: Every {eval_interval:,} steps")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60 + "\n")

    processed_data_path = "data/processed/nifty_options_features.parquet"
    if not os.path.exists(processed_data_path):
        processed_data_path = "data/processed/features.parquet"

    if os.path.exists(processed_data_path):
        df = pd.read_parquet(processed_data_path)
    else:
        print("[!] No feature parquet found. Generating extended synthetic feature dataset (20k rows with 15+ indicators)...")
        df = generate_rich_indicator_dataset(n_steps=20000)

    feature_cols = [col for col in df.columns if col not in ["datetime", "date", "timestamp"]]
    print(f"[✓] Environment loaded with {len(feature_cols)} feature indicators: {feature_cols}\n")

    env = StrictOptionSimEnv(df=df, feature_cols=feature_cols)

    input_dim = len(feature_cols)
    model = ActorCriticTCNGRU(input_dim=input_dim).to(device)

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

            scaled_reward = float(np.clip(raw_reward / 100.0, -5.0, 5.0))

            obs_buf.append(obs)
            act_buf.append(action_np)
            logp_buf.append(log_prob.cpu().numpy()[0])
            rew_buf.append(scaled_reward)
            val_buf.append(value.detach().cpu().numpy().reshape(-1)[0])
            done_buf.append(done)

            obs = next_obs
            if done:
                obs, _ = env.reset()

            # Logging evaluation precisely every 20k steps
            if global_step % eval_interval == 0:
                eval_callback.run_evaluation(global_step)

        # GAE Advantage Calculation
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, last_val = model.get_action(last_obs_tensor)
            last_val = last_val.detach().cpu().numpy().reshape(-1)[0]

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

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

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

                critic_loss = F.mse_loss(new_val.view(-1), mb_ret.view(-1))
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
    train_ppo_engine(total_timesteps=12_000_000, eval_interval=20_000, exp_name="ppo_12m")
