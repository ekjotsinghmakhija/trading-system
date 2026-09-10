import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

class CheckpointAndEscapeEngine:
    def __init__(self, checkpoint_dir: str = "models/checkpoints", ledger_path: str = "logs/experiments/factor_ledger.parquet"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_sharpe = -np.inf
        self.stagnant_milestones = 0
        self.escape_stage = 0

    def evaluate_and_checkpoint(self, model, env, current_step: int) -> dict:
        """Evaluates model performance over a full episode and saves best weights."""
        model.eval()
        device = next(model.parameters()).device  # Get active GPU/CPU device

        obs, _ = env.reset()
        done = False
        returns = []
        wins = 0
        trades = 0

        with torch.no_grad():
            while not done:
                # Ensure observation tensor matches model device
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = model(obs_tensor)
                act_val = action.cpu().numpy()[0]

                obs, reward, terminated, truncated, info = env.step(act_val)
                done = terminated or truncated

                pnl = info.get("step_pnl", 0.0)
                if pnl != 0:
                    trades += 1
                    if pnl > 0:
                        wins += 1
                returns.append(reward)

        ret_arr = np.array(returns)
        mean_ret = np.mean(ret_arr) if len(ret_arr) > 0 else 0.0
        std_ret = np.std(ret_arr) + 1e-6
        sharpe = float((mean_ret / std_ret) * np.sqrt(252 * 375))
        win_rate = float(wins / trades) if trades > 0 else 0.0
        final_capital = float(env.capital)

        print(f"[📊 Eval Step {current_step}] Sharpe: {sharpe:.2f} | Win Rate: {win_rate*100:.1f}% | Capital: ₹{final_capital:,.2f}")

        # Checkpoint Best Model
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.stagnant_milestones = 0
            ckpt_path = self.checkpoint_dir / f"best_model_step_{current_step}_sharpe_{sharpe:.2f}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[✓] New Best Model Saved -> {ckpt_path}")
        else:
            self.stagnant_milestones += 1

        return {"sharpe_ratio": sharpe, "win_rate": win_rate, "final_capital": final_capital}

    def check_local_minima_and_trigger_escape(self, model, optimizer, ppo_config: dict):
        """3-Stage Escape Mechanism for Policy Traps"""
        if self.stagnant_milestones < 3:
            return

        self.escape_stage += 1
        print(f"\n[⚠️ LOCAL MINIMA TRAP] Stagnant for {self.stagnant_milestones} evaluation windows!")

        if self.escape_stage == 1:
            print("[🚀 Escape Stage 1] Injecting Gaussian Noise into Policy Head...")
            with torch.no_grad():
                for param in model.actor_head.parameters():
                    param.add_(torch.randn_like(param) * 0.02)

        elif self.escape_stage == 2:
            print("[🚀 Escape Stage 2] Boosting Entropy Coefficient (0.01 -> 0.05)...")
            ppo_config["c2_entropy"] = 0.05

        elif self.escape_stage == 3:
            print("[🚀 Escape Stage 3] Escalating Learning Rate x5 temporarily...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 5.0
            self.escape_stage = 0  # Reset escape cycle

        self.stagnant_milestones = 0

    def record_to_parquet_ledger(self, metrics: dict):
        """Appends milestone evaluation telemetry to a Parquet file."""
        new_df = pd.DataFrame([metrics])
        if self.ledger_path.exists():
            df = pd.read_parquet(self.ledger_path)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_parquet(self.ledger_path, index=False)
