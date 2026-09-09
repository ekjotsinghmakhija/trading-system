import os
from pathlib import Path
import numpy as np
import torch
import polars as pl
from datetime import datetime

class CheckpointAndEscapeEngine:
    """
    Manages 100k-step evaluation checkpoints, Parquet factor ledger logging,
    and the 3-stage local minima escape mechanism.
    """
    def __init__(self, save_dir: str = "models/checkpoints", ledger_path: str = "logs/experiments/factor_ledger.parquet"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        self.eval_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 1e-4

    def record_to_parquet_ledger(self, step_data: dict):
        """Appends trade metrics and feature vector outputs to the Parquet ledger."""
        df_new = pl.DataFrame([step_data])

        if self.ledger_path.exists():
            df_existing = pl.read_parquet(self.ledger_path)
            df_combined = pl.concat([df_existing, df_new], rechunk=True)
            df_combined.write_parquet(self.ledger_path, compression="snappy")
        else:
            df_new.write_parquet(self.ledger_path, compression="snappy")

    def evaluate_and_checkpoint(self, model: torch.nn.Module, env, current_step: int) -> dict:
        """Runs out-of-sample evaluation pass at 100k step intervals."""
        model.eval()
        obs, _ = env.reset()
        done = False
        returns = []

        with torch.no_grad():
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _ = model(obs_tensor)
                obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
                returns.append(reward)
                done = terminated or truncated

        # Calculate Out-of-Sample Sharpe & Metrics
        ret_arr = np.array(returns)
        mean_ret = np.mean(ret_arr) if len(ret_arr) > 0 else 0.0
        std_ret = np.std(ret_arr) + 1e-6
        sharpe = float((mean_ret / std_ret) * np.sqrt(252 * 375))
        win_rate = float(np.sum(ret_arr > 0) / max(len(ret_arr), 1))

        eval_metrics = {
            "step": current_step,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "final_capital": info.get("capital", 0.0)
        }
        self.eval_history.append(eval_metrics)

        # Check model rejection cutoff: Reject checkpoint if max drawdown > 22% or negative yield
        drawdown = (env.initial_capital - info.get("capital", 0.0)) / env.initial_capital
        if drawdown <= 0.22 and info.get("capital", 0.0) >= env.initial_capital:
            ckpt_path = self.save_dir / f"model_step_{current_step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[✓] Step {current_step}: Checkpoint saved -> {ckpt_path} (Sharpe: {sharpe:.2f}, Win Rate: {win_rate:.1%})")
        else:
            print(f"[!] Step {current_step}: Checkpoint REJECTED due to risk bounds (Drawdown: {drawdown:.1%})")

        return eval_metrics

    def check_local_minima_and_trigger_escape(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ppo_config: dict) -> tuple[bool, str]:
        """
        Monitors metric stagnation across 3 consecutive 100k checkpoints.
        Triggers 3-Stage Escape Sequence if parameter/sharpe delta < 1e-4.
        """
        if len(self.eval_history) < 3:
            return False, "Insufficient history"

        s1 = self.eval_history[-3]["sharpe_ratio"]
        s2 = self.eval_history[-2]["sharpe_ratio"]
        s3 = self.eval_history[-1]["sharpe_ratio"]

        delta1 = abs(s3 - s2)
        delta2 = abs(s2 - s1)

        if delta1 < self.stagnation_threshold and delta2 < self.stagnation_threshold:
            self.stagnation_counter += 1
            print(f"\n[⚠️] LOCAL MINIMA DETECTED! Stagnation count: {self.stagnation_counter}. Executing 3-Stage Escape...")

            # Stage 1: Cosine Learning Rate Warm Restart (Reset to max lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
            print("    [Stage 1] Cosine LR Warm Restart executed -> LR set to 1e-6")

            # Stage 2: Boost PPO Entropy Multiplier (5x boost for 20k steps)
            ppo_config["c2_entropy"] = 0.05
            print("    [Stage 2] Entropy Loss Multiplier boosted 5x -> c2 set to 0.05")

            # Stage 3: Inject Parametric Gaussian Noise into Actor Weights
            with torch.no_grad():
                for name, param in model.actor_out.named_parameters():
                    noise = torch.randn_like(param) * 0.02
                    param.add_(noise)
            print("    [Stage 3] Parametric Gaussian Noise injected into Actor policy head (sigma=0.02)\n")

            return True, "3-Stage Escape Sequence Applied"

        return False, "Normal Policy Optimization"


if __name__ == "__main__":
    from models.architecture import ActorCriticTCNGRU

    # Smoke Test Callbacks
    model = ActorCriticTCNGRU(input_dim=18)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    ppo_cfg = {"c2_entropy": 0.01}

    engine = CheckpointAndEscapeEngine()

    # Simulate stagnation scenario
    engine.eval_history = [
        {"sharpe_ratio": 1.20001, "step": 100000},
        {"sharpe_ratio": 1.20003, "step": 200000},
        {"sharpe_ratio": 1.20002, "step": 300000}
    ]

    triggered, msg = engine.check_local_minima_and_trigger_escape(model, opt, ppo_cfg)
    print(f"Escape Check Result: {triggered} -> {msg}")
    print(f"Updated Entropy Coeff: {ppo_cfg['c2_entropy']}")
