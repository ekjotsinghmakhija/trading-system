import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

class CheckpointAndEscapeEngine:
    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        ledger_path: str = "logs/experiments/factor_ledger.parquet"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_sharpe = -np.inf
        self.stagnant_milestones = 0
        self.escape_stage = 0

    def evaluate_and_checkpoint(self, model, env, current_step: int) -> dict:
        """Evaluates model performance over a full episode with realistic metric calculation."""
        model.eval()
        device = next(model.parameters()).device

        obs, _ = env.reset()
        done = False
        wins = 0
        trades = 0
        portfolio_track = [env.initial_capital]

        with torch.no_grad():
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = model(obs_tensor)
                act_val = action.cpu().numpy()[0]

                obs, reward, terminated, truncated, info = env.step(act_val)
                done = terminated or truncated

                pnl = info.get("step_pnl", 0.0)
                friction = info.get("friction_cost", 0.0)
                if friction > 0 or pnl != 0:
                    trades += 1
                    if pnl > 0:
                        wins += 1

                portfolio_track.append(env.capital)

        # Standardized Sharpe Ratio Calculation over Portfolio Capital Curve
        port_arr = np.array(portfolio_track)
        pct_returns = np.diff(port_arr) / port_arr[:-1]

        if len(pct_returns) > 1 and np.std(pct_returns) > 1e-8:
            # Annualize based on 375 1-min bars per trading day
            sharpe = float((np.mean(pct_returns) / np.std(pct_returns)) * np.sqrt(252 * 375))
        else:
            sharpe = 0.0

        win_rate = float(wins / trades) if trades > 0 else 0.0
        final_capital = float(env.capital)

        print(
            f"[📊 Eval Step {current_step}] Sharpe: {sharpe:.2f} | "
            f"Win Rate: {win_rate*100:.1f}% | Trades: {trades} | Capital: ₹{final_capital:,.2f}"
        )

        # Periodic Model Tensor Checkpoint (For recovery and checkpointing every step milestone)
        step_ckpt_path = self.checkpoint_dir / f"model_step_{current_step}.pt"
        torch.save({
            "step": current_step,
            "model_state_dict": model.state_dict(),
            "sharpe": sharpe,
            "capital": final_capital
        }, step_ckpt_path)

        # Save Best Model state dict if Sharpe improves
        if sharpe > self.best_sharpe and trades > 0:
            self.best_sharpe = sharpe
            self.stagnant_milestones = 0
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"[✓] New Best Sharpe ({sharpe:.2f}) Saved -> {best_path}")
        else:
            self.stagnant_milestones += 1

        return {
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "final_capital": final_capital,
            "trades": trades
        }

    def check_local_minima_and_trigger_escape(self, model, optimizer, ppo_config: dict):
        """3-Stage Adaptive Escape Mechanism for Stagnant Policies"""
        if self.stagnant_milestones < 3:
            return

        self.escape_stage += 1
        print(f"\n[⚠️ LOCAL MINIMA TRAP] Stagnant for {self.stagnant_milestones} evaluation windows!")

        if self.escape_stage == 1:
            print("[🚀 Escape Stage 1] Injecting Controlled Noise into Policy Head (actor_dense)...")
            with torch.no_grad():
                for param in model.actor_dense.parameters():
                    param.add_(torch.randn_like(param) * 0.05)

        elif self.escape_stage == 2:
            print("[🚀 Escape Stage 2] Increasing Entropy Loss Weight (c2_entropy -> 0.05)...")
            ppo_config["c2_entropy"] = 0.05

        elif self.escape_stage == 3:
            print("[🚀 Escape Stage 3] Scaling Optimizer Learning Rate 3x temporarily...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 3.0
            self.escape_stage = 0

        self.stagnant_milestones = 0

    def record_to_parquet_ledger(self, metrics: dict):
        new_df = pd.DataFrame([metrics])
        if self.ledger_path.exists():
            df = pd.read_parquet(self.ledger_path)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_parquet(self.ledger_path, index=False)
