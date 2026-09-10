import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from models.architecture import ActorCriticTCNGRU
from env.strict_sim_env import StrictOptionSimEnv
from features.feature_engineer import FeatureEngine
from models.callbacks import CheckpointAndEscapeEngine

torch.backends.cudnn.benchmark = True

def train_ppo_engine(total_timesteps=12_000_000, eval_interval=100_000, batch_size=4096, minibatch_size=512, exp_name="run_ppo"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = f"logs/experiments/{exp_name}_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)

    print(f"\n==========================================================")
    print(f"[🚀] Launching Training Engine on Device: {device}")
    print(f"     Experiment Logs: {log_dir}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"==========================================================\n")

    dates = pd.date_range("2026-01-01 09:15:00", periods=20000, freq="1min", tz="UTC")
    dummy_df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(20000).cumsum() + 25000,
        "high": np.random.randn(20000).cumsum() + 25020,
        "low": np.random.randn(20000).cumsum() + 24980,
        "close": np.random.randn(20000).cumsum() + 25000,
        "volume": np.random.randint(100, 5000, size=20000)
    })

    feat_engine = FeatureEngine(dummy_df)
    matrix = feat_engine.build_feature_matrix()
    feature_cols = [c for c in matrix.columns if c.startswith("feat_")]

    env = StrictOptionSimEnv(matrix, feature_cols)
    model = ActorCriticTCNGRU(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    huber_loss = nn.HuberLoss()

    ppo_config = {"c2_entropy": 0.01, "c1_value": 0.5}
    cb_engine = CheckpointAndEscapeEngine()

    current_step = 0
    obs, _ = env.reset()
    start_time = time.time()

    while current_step < total_timesteps:
        obs_buffer, action_buffer, reward_buffer = [], [], []
        model.eval()

        for _ in range(batch_size):
            current_step += 1
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                action, value, _ = model(obs_tensor)

            act_val = action.cpu().numpy()[0]
            next_obs, reward, term, trunc, info = env.step(act_val)

            obs_buffer.append(obs)
            action_buffer.append(act_val)
            reward_buffer.append(reward)

            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()

            # Evaluation & Checkpoint Interval
            if current_step % eval_interval == 0 or current_step == total_timesteps:
                eval_metrics = cb_engine.evaluate_and_checkpoint(model, env, current_step)
                cb_engine.check_local_minima_and_trigger_escape(model, optimizer, ppo_config)

                # Write metrics to TensorBoard and Parquet
                writer.add_scalar("Eval/SharpeRatio", eval_metrics["sharpe_ratio"], current_step)
                writer.add_scalar("Eval/WinRate", eval_metrics["win_rate"], current_step)
                writer.add_scalar("Eval/Capital", eval_metrics["final_capital"], current_step)

                cb_engine.record_to_parquet_ledger({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "step": current_step,
                    "sharpe": eval_metrics["sharpe_ratio"],
                    "win_rate": eval_metrics["win_rate"],
                    "capital": eval_metrics["final_capital"]
                })
                model.eval()

        # Minibatch PPO Update
        model.train()
        obs_tensor_b = torch.tensor(np.array(obs_buffer), dtype=torch.float32, device=device)
        actions_tensor_b = torch.tensor(np.array(action_buffer), dtype=torch.float32, device=device)
        rewards_tensor_b = torch.tensor(np.array(reward_buffer), dtype=torch.float32, device=device)

        for _ in range(4):
            permutation = torch.randperm(batch_size)
            for start_idx in range(0, batch_size, minibatch_size):
                batch_indices = permutation[start_idx : start_idx + minibatch_size]
                mb_obs, mb_act, mb_rew = obs_tensor_b[batch_indices], actions_tensor_b[batch_indices], rewards_tensor_b[batch_indices]

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    pred_actions, pred_values, _ = model(mb_obs)
                    v_loss = huber_loss(pred_values.squeeze(), mb_rew)
                    p_loss = -torch.mean(pred_actions * mb_rew.unsqueeze(1))
                    e_loss = -torch.mean(pred_actions ** 2)
                    loss = p_loss + ppo_config["c1_value"] * v_loss + ppo_config["c2_entropy"] * e_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()

        if current_step % 10000 == 0:
            elapsed = time.time() - start_time
            fps = current_step / elapsed
            writer.add_scalar("Train/FPS", fps, current_step)
            writer.add_scalar("Train/Loss", loss.item(), current_step)
            print(f"Step {current_step:8d} / {total_timesteps} | FPS: {fps:5.0f} | Loss: {loss.item():.4f}")

    writer.close()
    print("\n[✓] Training Pass Completed!")

def train_ppo_12m_steps():
    train_ppo_engine(total_timesteps=12_000_000, eval_interval=100_000, exp_name="ppo_12m")
