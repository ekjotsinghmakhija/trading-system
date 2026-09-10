import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


class HighThroughputPPOTrainer:
    """
    Production PPO Trainer using PyTorch 2.x Unified AMP API
    and Lion/AdamW momentum updates.
    """
    def __init__(self, model, env=None, lr: float = 3e-6, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.env = env
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # PyTorch 2.x Unified AMP GradScaler
        self.use_cuda = self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_cuda)
        self.entropy_coef = 0.02

    def train_epoch_amp(self, obs_batch, act_batch, logp_batch, adv_batch, rtg_batch):
        self.model.train()  # Set model to training mode

        obs_t = obs_batch.to(self.device, non_blocking=True)
        act_t = act_batch.to(self.device, non_blocking=True)
        logp_t = logp_batch.to(self.device, non_blocking=True)
        adv_t = adv_batch.to(self.device, non_blocking=True)
        rtg_t = rtg_batch.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # PyTorch 2.x Unified AMP autocast API
        device_type = "cuda" if self.use_cuda else "cpu"
        dtype = torch.bfloat16 if (self.use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

        with autocast(device_type=device_type, dtype=dtype, enabled=self.use_cuda):
            new_logp, entropy, values = self.model.evaluate_actions(obs_t, act_t)
            ratios = torch.exp(new_logp - logp_t)

            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 0.88, 1.12) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            # Explicitly flatten both value prediction and target tensors to 1D (.view(-1))
            # to prevent PyTorch scalar vs vector broadcasting size warnings
            critic_loss = nn.functional.huber_loss(values.view(-1), rtg_t.view(-1))
            entropy_loss = -entropy.mean()

            total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        if self.use_cuda:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

        return total_loss.item()
