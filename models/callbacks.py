import os
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class EvaluationCallback:
    """
    Robust Evaluation Callback.
    Calculates realistic equity returns and prevents Sharpe ratio standard-deviation explosions.
    """
    def __init__(
        self,
        eval_env,
        model,
        device: torch.device,
        eval_interval: int = 100_000,
        checkpoint_dir: str = "models/checkpoints",
        writer: SummaryWriter = None
    ):
        self.eval_env = eval_env
        self.model = model
        self.device = device
        self.eval_interval = eval_interval
        self.checkpoint_dir = checkpoint_dir
        self.writer = writer

        self.best_sharpe = -np.inf
        self.stagnant_eval_count = 0

    def run_evaluation(self, global_step: int) -> float:
        self.model.eval()
        obs, _ = self.eval_env.reset()
        done = False

        trades_count = 0
        total_pnl = 0.0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, _, _ = self.model.get_action(obs_tensor, deterministic=True)

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = self.eval_env.step(action_np)
            done = terminated or truncated

            trade_cost = info.get("trade_cost", info.get("turnover_cost", 0.0))
            if trade_cost > 0.0:
                trades_count += 1

            total_pnl += info.get("step_pnl", info.get("pnl", 0.0))
            obs = next_obs

        # Calculate True Equity Curve Returns (Step-by-Step)
        equity_curve = np.array(getattr(self.eval_env, "history_capital", [10000.0]), dtype=np.float64)

        if len(equity_curve) > 1:
            pct_returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)
        else:
            pct_returns = np.array([0.0])

        std_ret = np.std(pct_returns)
        mean_ret = np.mean(pct_returns)

        # Clamped Sharpe Calculation
        if std_ret > 1e-5 and trades_count > 0:
            raw_sharpe = (mean_ret / std_ret) * np.sqrt(252 * 375)
            sharpe = float(np.clip(raw_sharpe, -100.0, 100.0))
        else:
            sharpe = 0.0

        final_capital = getattr(self.eval_env, "capital", getattr(self.eval_env, "equity", 10000.0))
        win_rate = (np.sum(pct_returns > 0) / len(pct_returns)) * 100.0 if len(pct_returns) > 0 else 0.0

        print(
            f"[📊 Eval Step {global_step}] Sharpe: {sharpe:.2f} | "
            f"Win Rate: {win_rate:.1f}% | Trades: {trades_count} | Capital: ₹{final_capital:,.2f}"
        )

        if self.writer is not None:
            self.writer.add_scalar("eval/sharpe", sharpe, global_step)
            self.writer.add_scalar("eval/trades", trades_count, global_step)
            self.writer.add_scalar("eval/capital", final_capital, global_step)
            self.writer.add_scalar("eval/win_rate", win_rate, global_step)

        # Auto-Rescue: Inject noise if model locks into 0 trades for 2 consecutive evals
        if trades_count == 0:
            self.stagnant_eval_count += 1
            if self.stagnant_eval_count >= 2:
                print(f"[⚠️ STAGNATION DETECTED] Policy executed 0 trades across multiple windows.")
                print("[🚀 Auto-Rescue] Perturbing actor weights to break deadlock...")
                with torch.no_grad():
                    self.model.actor_dense.weight.add_(torch.randn_like(self.model.actor_dense.weight) * 0.05)
                self.stagnant_eval_count = 0
        else:
            self.stagnant_eval_count = 0

        # Save Checkpoints
        if sharpe > self.best_sharpe and trades_count > 0:
            self.best_sharpe = sharpe
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"[✓] Best Model Saved -> {best_model_path} (Sharpe: {sharpe:.2f})")

        step_checkpoint_path = os.path.join(self.checkpoint_dir, f"model_step_{global_step}.pt")
        torch.save(self.model.state_dict(), step_checkpoint_path)

        return sharpe
