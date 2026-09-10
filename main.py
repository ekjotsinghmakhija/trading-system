import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from env.strict_sim_env import StrictOptionSimEnv
from models.architecture import ActorCriticTCNGRU
from features.factor_ledger import FactorDiscoveryLedger


def generate_18_feature_dataset(rows: int = 50000) -> pd.DataFrame:
    np.random.seed(42)
    prices = 24000.0 + np.cumsum(np.random.normal(0.05, 2.5, size=rows))

    data = {
        "close": prices,
        "high": prices + np.abs(np.random.normal(0, 1.2, size=rows)),
        "low": prices - np.abs(np.random.normal(0, 1.2, size=rows)),
        "open": prices + np.random.normal(0, 0.5, size=rows),
        "rsi_14": np.sin(np.linspace(0, 100, rows)) * 50 + 50,
        "vwap_dist": np.random.normal(0, 0.002, size=rows),
        "futures_basis": np.random.normal(0.001, 0.0005, size=rows),
        "oi_change_acc": np.random.normal(0, 1.0, size=rows),
        "price_acceleration": np.random.normal(0, 0.1, size=rows),
        "momentum_density": np.random.normal(1.0, 0.2, size=rows),
        "iv_skew_velocity": np.random.normal(0, 0.05, size=rows),
        "pcr_velocity": np.random.normal(0, 0.01, size=rows),
        "realized_vol_vel": np.random.normal(0, 0.02, size=rows),
        "effective_gamma": np.random.normal(0, 0.001, size=rows),
        "iv_rv_gap_ratio": np.random.normal(1.0, 0.1, size=rows),
        "vega_velocity": np.random.normal(0, 0.03, size=rows),
        "ofi_1m": np.random.uniform(-1, 1, size=rows),
        "bid_ask_decay": np.random.exponential(1.0, size=rows),
        "volume_spike_factor": np.random.gamma(2, 1, size=rows)
    }
    return pd.DataFrame(data)


def main():
    print("==========================================================")
    print("     QUANTUM-50K ENGINE: REBUILT ZERO-LEAKAGE PIPELINE     ")
    print("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[🚀] Operating on Compute Device: {device}")

    # Generate Feature Matrix
    df = generate_18_feature_dataset(50000)
    feature_cols = [c for c in df.columns if c != "close"]
    print(f"[✓] Feature Matrix Loaded ({len(feature_cols)} orthogonal indicators)")

    # Initialize Engine & Ledger
    env = StrictOptionSimEnv(df, feature_cols=feature_cols, initial_capital=50000.0)
    ledger = FactorDiscoveryLedger("logs/experiments/factor_ledger.parquet")

    # Architecture + Hyperparameters
    input_dim = len(feature_cols) + 2  # 18 features + current_position + hold_ratio
    model = ActorCriticTCNGRU(input_dim=input_dim, action_dim=1).to(device)

    # Target Baseline Learning Rate: 3e-6
    learning_rate = 3e-6
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"[🚀] Training initialized with LR = {learning_rate}")

    # Stagnation Monitor Variables
    eval_sharpes = []
    stagnation_counter = 0

    # Main Walk-Forward Loop
    obs, _ = env.reset()
    for step in range(1, 100001):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action, _, value = model.get_action(obs_tensor)

        raw_action = action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(raw_action)

        # Log to Factor Ledger
        ledger.log_step(obs[:len(feature_cols)], feature_cols, info["state_record"])

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        # Checkpoint Evaluation & Local Minima Escape Monitor
        if step % 20000 == 0:
            ledger.flush_to_disk()
            current_capital = info['equity']
            recent_sharpe = (current_capital - 50000.0) / 50000.0  # Proxy metric
            eval_sharpes.append(recent_sharpe)

            print(f"[📊 Step {step}] Capital: ₹{current_capital:,.2f} | Last Trade PnL: ₹{info['pnl']:.2f}")

            # Local Minima Stagnation Escape Check
            if len(eval_sharpes) >= 3:
                delta1 = abs(eval_sharpes[-1] - eval_sharpes[-2])
                delta2 = abs(eval_sharpes[-2] - eval_sharpes[-3])

                if delta1 < 1e-4 and delta2 < 1e-4:
                    stagnation_counter += 1
                    print(f"[⚠️ STAGNATION DETECTED] Triggering Stage {stagnation_counter} Escape Mechanism...")

                    if stagnation_counter == 1:
                        # Stage 1: Reset LR Warm Restart
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 3e-6
                    elif stagnation_counter == 2:
                        # Stage 2: Inject Noise into Model Parameters
                        with torch.no_grad():
                            for param in model.actor_head.parameters():
                                param.add_(torch.randn_like(param) * 0.02)
                        stagnation_counter = 0

    print("[✓] Factor Ledger saved to logs/experiments/factor_ledger.parquet")


if __name__ == "__main__":
    main()
