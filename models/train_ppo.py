import os
import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from features.feature_engineer import FeatureEngineer
from env.strict_sim_env import StrictFrictionEnv
from models.callbacks import TradingMetricsCallback, FrictionCurriculumCallback

def make_env(data_path: str):
    """Factory function to build and wrap the StrictFrictionEnv."""
    def _init():
        raw_df = pd.read_csv(data_path)

        # 1. Pipeline execution
        engineer = FeatureEngineer()
        features_df = engineer.process_data(raw_df)

        # Align target price data with dropped NaN index from feature creation
        aligned_data = raw_df.loc[features_df.index].reset_index(drop=True)
        features_df = features_df.reset_index(drop=True)

        # Drop non-numeric metadata columns from observation space
        numeric_features = features_df.select_dtypes(include=['float32', 'float64', 'int32', 'int64', 'number'])
        if 'timestamp' in numeric_features.columns:
            numeric_features = numeric_features.drop(columns=['timestamp'])
            
        env = StrictFrictionEnv(data=aligned_data, features=numeric_features)
        return Monitor(env)
    return _init

def train():
    # Directories
    tensorboard_log = "./tensorboard_logs/ppo_v2/"
    checkpoint_dir = "./models/checkpoints/"
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Instantiate Vectorized & Normalized Environment
    train_env_fn = make_env("data/processed/train_1min.csv")
    env = DummyVecEnv([train_env_fn])

    # Normalize features and rewards (Critical for policy stability in RL)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.98  # Match agent discount factor
    )

    # 2. Network Architecture & PPO Hyperparameters
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[256, 256],  # Policy Network (Actor)
            vf=[256, 256]   # Value Network (Critic)
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,              # Low learning rate to prevent policy collapse
        n_steps=4096,                    # High batch size for smooth gradient estimates in noisy data
        batch_size=128,
        n_epochs=10,
        gamma=0.98,                      # Slightly lower discount factor for short-term trading
        gae_lambda=0.95,
        clip_range=0.2,                  # PPO clipping ratio
        ent_coef=0.01,                   # Entropy coefficient to encourage initial exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,                 # Early stopping for policy updates if KL divergence explodes
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=42
    )

    # 3. Callbacks Setup
    metrics_callback = TradingMetricsCallback()
    curriculum_callback = FrictionCurriculumCallback(total_curriculum_steps=200000, max_multiplier=1.10)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="ppo_strict_v2"
    )

    callbacks = CallbackList([metrics_callback, curriculum_callback, checkpoint_callback])

    # 4. Train Model
    total_timesteps = 1_000_000
    print(f"--- Launching Training Pipeline ({total_timesteps} steps) ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # 5. Save Final Artifacts
    model.save(f"{checkpoint_dir}/ppo_strict_final")
    env.save(f"{checkpoint_dir}/vec_normalize_final.pkl")
    print("--- Training Complete & Model Checkpoints Saved ---")

if __name__ == "__main__":
    train()
