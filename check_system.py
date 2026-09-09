"""
Pre-Training System & Environment Verification Script
Verifies: GPU support, dataset existence, non-numeric column leaks, and Gym env resets.
"""

import sys
import os
import torch
import pandas as pd
import numpy as np

def verify_system():
    print("=" * 60)
    print("      TRADING SYSTEM PRE-FLIGHT VERIFICATION CHECK       ")
    print("=" * 60)

    # 1. PyTorch & CUDA Check
    print("\n[1/5] Checking PyTorch & GPU Acceleration...")
    print(f"  - PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  - CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  - Target Device: {torch.cuda.get_device_name(0)}")
    else:
        print("  - WARNING: CUDA is not available. SB3 will fall back to CPU.")

    # 2. File & Dataset Check
    print("\n[2/5] Checking Required Data Files...")
    required_files = [
        "data/processed/train_data.csv",
        "data/processed/train_1min.csv",
        "models/train_ppo.py",
        "env/strict_sim_env.py",
        "main.py"
    ]
    missing_files = []
    for filepath in required_files:
        exists = os.path.exists(filepath)
        status = "FOUND" if exists else "MISSING"
        print(f"  - [{status}] {filepath}")
        if not exists and "train_data.csv" in filepath:
            print("    -> Hint: Run 'cp data/processed/train_1min.csv data/processed/train_data.csv'")
            missing_files.append(filepath)

    if missing_files:
        print("\n  ❌ VERIFICATION FAILED: Missing required data files.")
        return False

    # 3. Non-Numeric Column Inspection
    print("\n[3/5] Inspecting Observation Features for Non-Numeric Leaks...")
    df = pd.read_csv("data/processed/train_data.csv", nrows=100)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if non_numeric_cols:
        print(f"  - WARNING: Found non-numeric columns in raw CSV: {non_numeric_cols}")
        print("    -> Ensure environment wrapper filters these before PyTorch conversion.")
    else:
        print("  - All loaded columns are strictly numeric.")

    # 4. Imports & Type Hint Check
    print("\n[4/5] Testing Module Imports...")
    try:
        from env.strict_sim_env import StrictFrictionEnv
        print("  - StrictFrictionEnv imported successfully.")
    except Exception as e:
        print(f"  - ❌ Import Error in StrictFrictionEnv: {e}")
        return False

    try:
        from models.train_ppo import train
        print("  - train_ppo module imported successfully.")
    except Exception as e:
        print(f"  - ❌ Import Error in train_ppo: {e}")
        return False

    # 5. Gym Environment Reset Test
    print("\n[5/5] Testing Gym Environment Initialization & Reset...")
    try:
        features_df = df.select_dtypes(include=[np.number])
        env = StrictFrictionEnv(data=df, features=features_df)
        obs, info = env.reset()
        print(f"  - Gym Env initialized successfully!")
        print(f"  - Observation Shape: {obs.shape}")
        print(f"  - Observation Dtype: {obs.dtype}")
    except Exception as e:
        print(f"  - ❌ Environment Initialization Failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("  ✅ PRE-FLIGHT VERIFICATION COMPLETE: Ready for PPO Training.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify_system()
    sys.exit(0 if success else 1)
