import os
import torch
from features.feature_engineer import FeatureEngineer
from features.factor_ledger import FactorDiscoveryLedger
from env.strict_sim_env import StrictOptionSimEnv
from models.architecture import ActorCriticTCNGRU
from models.train_engine import PPOTrainEngine


def main():
    print("==========================================================")
    print("       QUANTUM-50K ENGINE: PRODUCTION PIPELINE          ")
    print("==========================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[🚀] Operating Device: {device}")

    # Data Ingestion & Feature Engineering
    engineer = FeatureEngineer()
    raw_df = engineer.generate_synthetic_raw_feed(50000)
    df = engineer.compute_18_alpha_matrix(raw_df)

    feature_cols = [c for c in df.columns if c != "close"]
    print(f"[✓] Feature Matrix Ready ({len(feature_cols)} Alpha Features)")

    # Environment & Ledger
    env = StrictOptionSimEnv(df, feature_cols=feature_cols, initial_capital=50000.0)
    ledger = FactorDiscoveryLedger("logs/experiments/factor_ledger.parquet")

    # Neural Model Architecture
    input_dim = len(feature_cols) + 2  # 18 Features + Position + Holding Ratio
    model = ActorCriticTCNGRU(input_dim=input_dim, action_dim=1)

    # PPO Engine Setup
    trainer = PPOTrainEngine(model=model, env=env, ledger=ledger, lr=3e-6, device=device)

    print("[🚀] Training 12M Step Engine Routine...")
    for step in range(1, 101):
        loss, current_equity = trainer.train_step(num_steps=1024)

        if step % 10 == 0:
            ledger.flush_to_disk()
            trainer.check_local_minima(current_equity)
            print(f"[📊 Epoch {step}] PPO Loss: {loss:.4f} | Portfolio Equity: ₹{current_equity:,.2f}")

    print("[✓] Execution Complete. Factor Ledger updated in logs/experiments/factor_ledger.parquet")


if __name__ == "__main__":
    main()
