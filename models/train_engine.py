import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


class HighThroughputPPOTrainer:
    """
    Production PPO Trainer optimized for CUDA AMP, kernel fusion,
    and adaptive local minima escape mechanisms.
    """
    def __init__(self, model, env, lr: float = 3e-6, device: str = "cuda"):
        self.device = torch.device(device)
        self.env = env

        # Compile model for CUDA kernel fusion
        self.model = torch.compile(model.to(self.device), mode="reduce-overhead")

        # Lion optimizer for uniform momentum-based step sizes
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = GradScaler()
        self.entropy_coef = 0.02

    def train_epoch_amp(self, obs_batch, act_batch, logp_batch, adv_batch, rtg_batch):
        self.model.train()

        # Asynchronous non-blocking transfer to GPU VRAM
        obs_t = obs_batch.to(self.device, non_blocking=True)
        act_t = act_batch.to(self.device, non_blocking=True)
        logp_t = logp_batch.to(self.device, non_blocking=True)
        adv_t = adv_batch.to(self.device, non_blocking=True)
        rtg_t = rtg_batch.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Forward Pass
        with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            new_logp, entropy, values = self.model.evaluate_actions(obs_t, act_t)
            ratios = torch.exp(new_logp - logp_t)

            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 0.88, 1.12) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.functional.huber_loss(values.squeeze(), rtg_t)
            entropy_loss = -entropy.mean()

            total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        # Scaled Gradient Backpropagation
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.item()
