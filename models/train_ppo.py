import os
import numpy as np
import pandas as pd

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.strict_sim_env import AdvancedFrictionEnv


def make_env(data_path: str):
    def _init():
        df = pd.read_csv(data_path)
        execution_cols = ['close']

        numeric_features = df.select_dtypes(include=[np.number])
        for col in ['timestamp', 'date', 'datetime', 'time']:
            if col in numeric_features.columns:
                numeric_features = numeric_features.drop(columns=[col])

        # FIXED: Pass df[execution_cols] as DataFrame instead of execution_cols list
        return AdvancedFrictionEnv(data=df[execution_cols], features=numeric_features)
    return _init


def train(train_csv: str = "data/processed/train_data.csv", total_timesteps: int = 2_000_000):
    print(f"--- Launching RecurrentPPO (LSTM) Training ({total_timesteps} steps) ---")

    # DummyVecEnv is required for RecurrentPPO hidden state tracking
    raw_env = DummyVecEnv([make_env(train_csv)])

    # Wrap with feature and reward normalization
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Encourages exploration across position sizing spectrum
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device="cuda"
    )

    model.learn(total_timesteps=total_timesteps)

    model.save("models/checkpoints/ppo_strict_final")
    env.save("models/checkpoints/vec_normalize_final.pkl")
    print("RecurrentPPO model and normalization stats saved successfully.")


if __name__ == "__main__":
    train()
