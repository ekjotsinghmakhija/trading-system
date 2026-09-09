import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from models.architecture import ActorCriticTCNGRU
from env.strict_sim_env import StrictOptionSimEnv
from features.feature_engineer import FeatureEngine
from models.callbacks import CheckpointAndEscapeEngine

# Enable PyTorch CUDNN Benchmarks for maximum hardware throughput
torch.backends.cudnn.benchmark = True

def train_ppo_12m_steps():
    """
    Hardware-Optimized 12M Step Walk-Forward PPO Engine
    - Platform: Intel Core Ultra 7 + NVIDIA RTX 5060 (8GB VRAM) + 32GB RAM
    - Acceleration: Automatic Mixed Precision (AMP FP16) + Pinned Memory
    """
    TOTAL_TIMESTEPS = 12_000_000
    EVAL_INTERVAL = 100_000
    BATCH_SIZE = 4096  # Doubled batch size for RTX 5060 parallel throughput
    PPO_EPOCHS = 4
    MINIBATCH_SIZE = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================================")
    print(f"[🚀] Launching Training Engine on Device: {device}")
    if torch.cuda.is_available():
        print(f"     GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"     Allocated VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"==========================================================\n")

    # 1. Load Data & Build Alpha Matrix
    print("[+] Pre-processing features into RAM buffer...")
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

    # Optional JIT Compile for RTX 5060 architecture
    try:
        model = torch.compile(model)
        print("[✓] Model successfully compiled with PyTorch 2.0 Inductor compiler.")
    except Exception as e:
        print(f"[!] JIT compilation bypassed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()  # Automatic Mixed Precision Scaler
    huber_loss = nn.HuberLoss()

    ppo_config = {"c2_entropy": 0.01, "c1_value": 0.5}
    cb_engine = CheckpointAndEscapeEngine()

    current_step = 0
    obs, _ = env.reset()
    start_time = time.time()

    print("[+] Starting 12M Step Policy Optimization Loop...\n")

    while current_step < TOTAL_TIMESTEPS:
        obs_buffer, action_buffer, reward_buffer = [], [], []

        # 2. Collect Experience Batch
        model.eval()
        for _ in range(BATCH_SIZE):
            current_step += 1
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                action, value, _ = model(obs_tensor)

            act_val = action.cpu().numpy()[0]
            next_obs, reward, term, trunc, info = env.step(act_val)

            obs_buffer.append(obs)
            action_buffer.append(act_val)
            reward_buffer.append(reward)

            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()

            # 100k Step Evaluation & Escape Trigger Check
            if current_step % EVAL_INTERVAL == 0:
                print(f"\n----------------------------------------------------------")
                print(f"[📊] Milestone Reached: Step {current_step}/{TOTAL_TIMESTEPS}")
                eval_metrics = cb_engine.evaluate_and_checkpoint(model, env, current_step)
                cb_engine.check_local_minima_and_trigger_escape(model, optimizer, ppo_config)

                cb_engine.record_to_parquet_ledger({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "step": current_step,
                    "sharpe": eval_metrics["sharpe_ratio"],
                    "win_rate": eval_metrics["win_rate"],
                    "capital": eval_metrics["final_capital"]
                })
                print(f"----------------------------------------------------------\n")
                model.eval()

        # 3. Fast PPO Optimization Pass with AMP FP16
        model.train()
        obs_tensor_b = torch.tensor(np.array(obs_buffer), dtype=torch.float32, device=device)
        actions_tensor_b = torch.tensor(np.array(action_buffer), dtype=torch.float32, device=device)
        rewards_tensor_b = torch.tensor(np.array(reward_buffer), dtype=torch.float32, device=device)

        for _ in range(PPO_EPOCHS):
            permutation = torch.randperm(BATCH_SIZE)
            for start_idx in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                batch_indices = permutation[start_idx : start_idx + MINIBATCH_SIZE]

                mb_obs = obs_tensor_b[batch_indices]
                mb_act = actions_tensor_b[batch_indices]
                mb_rew = rewards_tensor_b[batch_indices]

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
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

        if current_step % 20000 == 0:
            elapsed = time.time() - start_time
            fps = current_step / elapsed
            print(f"Step {current_step:8d} | FPS: {fps:5.0f} | Batch Loss: {loss.item():.4f}")

    print("\n[✓] 12,000,000 Step Training Run Completed Successfully!")

if __name__ == "__main__":
    train_ppo_12m_steps()
