import os
import torch
from torch.utils.tensorboard import SummaryWriter


class ModelCheckpointCallback:
    """
    Monitors portfolio equity and saves optimal model state checkpoints.
    """
    def __init__(self, save_dir: str = "models/checkpoints", log_dir: str = "tensorboard_logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_equity = -float("inf")

    def on_epoch_end(self, epoch: int, current_equity: float, loss: float, model: torch.nn.Module):
        self.writer.add_scalar("Portfolio/Equity", current_equity, epoch)
        self.writer.add_scalar("Train/Loss", loss, epoch)

        if current_equity > self.best_equity:
            self.best_equity = current_equity
            save_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[💾 CHECKPOINT] Peak Equity ₹{current_equity:,.2f} -> Model saved to {save_path}")

    def close(self):
        self.writer.close()
