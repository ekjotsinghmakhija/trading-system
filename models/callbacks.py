import os
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter


class PerformanceMonitorCallback:
    """
    Streams hardware performance and internal training state metrics to TensorBoard.
    """
    def __init__(self, log_dir: str = "tensorboard_logs", checkpoint_dir: str = "models/checkpoints"):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_sharpe = -float("inf")

    def log_metrics(self, epoch: int, metrics: dict):
        # 1. Hardware Utilization Diagnostics
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            self.writer.add_scalar("Hardware/VRAM_Allocated_MB", vram_mb, epoch)
            self.writer.add_scalar("Hardware/VRAM_Reserved_MB", vram_reserved, epoch)

        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        self.writer.add_scalar("Hardware/CPU_Usage_Percent", cpu_usage, epoch)
        self.writer.add_scalar("Hardware/RAM_Usage_Percent", ram_usage, epoch)

        # 2. RL & Training Diagnostics
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(f"RL_Engine/{key}", val, epoch)

    def save_checkpoint(self, model, optimizer, epoch: int, sharpe: float):
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sharpe": sharpe
            }, path)

    def close(self):
        self.writer.close()
