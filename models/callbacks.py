import os
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class EvaluationCallback:
    """
    Periodic Evaluation & Auto-Rescue Callback for Intraday Options PPO Strategy.
    Prevents policy deadlock by measuring active performance and saving best checkpoints.
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
        returns_list = []

        # Run 1 evaluation episode pass
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                # Allow minor exploration during evaluation to prevent zero-action lock
                action, _, _ = self.model.get_action(obs_tensor, deterministic=False)

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = self.eval_env.step(action_np)
            done = terminated or truncated

            if info.get("step_pnl", 0.0) != 0.0:
                trades_count += 1
                total_pnl += info["step_pnl"]

            returns_list.append(reward)
            obs = next_obs

        # Calculate Sharpe
        returns_arr = np.array(returns_list)
        std_ret = np.std(returns_arr)
        sharpe = (np.mean(returns_arr) / (std_ret + 1e-8)) * np.sqrt(252 * 375) if std_ret > 1e-6 else 0.0
        final_capital = self.eval_env.capital
        win_rate = (np.sum(returns_arr > 0) / len(returns_arr)) * 100.0 if len(returns_arr) > 0 else 0.0

        print(
            f"[📊 Eval Step {global_step}] Sharpe: {sharpe:.2f} | "
            f"Win Rate: {win_rate:.1f}% | Trades: {trades_count} | Capital: ₹{final_capital:,.2f}"
        )

        if self.writer is not None:
            self.writer.add_scalar("eval/sharpe", sharpe, global_step)
            self.writer.add_scalar("eval/trades", trades_count, global_step)
            self.writer.add_scalar("eval/capital", final_capital, global_step)
            self.writer.add_scalar("eval/win_rate", win_rate, global_step)

        # Auto-rescue: If 2 consecutive evals generate 0 trades, inject policy perturbation
        if trades_count == 0:
            self.stagnant_eval_count += 1
            if self.stagnant_eval_count >= 2:
                print(f"[⚠️ LOCAL MINIMA TRAP] Stagnant for {self.stagnant_eval_count} evaluation windows!")
                print("[🚀 Escape Stage 1] Injecting Controlled Noise into Policy Head (actor_dense)...")
                with torch.no_grad():
                    self.model.actor_dense.weight.add_(torch.randn_like(self.model.actor_dense.weight) * 0.1)
                    self.model.actor_dense.bias.add_(torch.randn_like(self.model.actor_dense.bias) * 0.05)
                self.stagnant_eval_count = 0
        else:
            self.stagnant_eval_count = 0

        # Save Best Checkpoint
        if sharpe > self.best_sharpe and trades_count > 0:
            self.best_sharpe = sharpe
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"[✓] New Best Sharpe ({sharpe:.2f}) Saved -> {best_model_path}")

        step_checkpoint_path = os.path.join(self.checkpoint_dir, f"model_step_{global_step}.pt")
        torch.save(self.model.state_dict(), step_checkpoint_path)

        return sharpe
