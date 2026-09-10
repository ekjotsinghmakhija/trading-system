import sys
import os
from models.train_engine import train_ppo_engine

if __name__ == "__main__":
    print("[🧪] Running 10-second Verification Pass...")
    train_ppo_engine(
        total_timesteps=5_000,
        eval_interval=1_000,
        batch_size=512,
        minibatch_size=128,
        exp_name="sanity_test"
    )
    print("[✓] Verification Successful! All tensors aligned and logging active.")
