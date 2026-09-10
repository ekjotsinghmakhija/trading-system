import torch
import torch.optim as optim
import numpy as np


class PPOTrainEngine:
    def __init__(self, model, env, ledger, lr: float = 3e-6, device: str = "cuda"):
        self.model = model.to(device)
        self.env = env
        self.ledger = ledger
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.entropy_coef = 0.01
        self.clip_eps = 0.12
        self.gamma = 0.99
        self.lam = 0.95

        self.stagnation_counter = 0
        self.eval_history = []

    def train_step(self, num_steps: int = 2048):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
        obs, _ = self.env.reset()

        for _ in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                act, logp, val = self.model.get_action(obs_tensor)

            next_obs, rew, term, trunc, info = self.env.step(act.cpu().numpy()[0])
            self.ledger.log_step(obs[:18], self.env.feature_cols, info["state_record"])

            obs_buf.append(obs)
            act_buf.append(act.cpu().numpy()[0])
            logp_buf.append(logp.cpu().numpy()[0])
            rew_buf.append(rew)
            val_buf.append(val.cpu().numpy()[0][0])

            obs = next_obs
            if term or trunc:
                obs, _ = self.env.reset()

        # Compute GAE
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(self.device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32).to(self.device)
        logp_t = torch.tensor(np.array(logp_buf), dtype=torch.float32).to(self.device)

        advantages, rewards_to_go = self._compute_gae(rew_buf, val_buf)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        rtg_t = torch.tensor(rewards_to_go, dtype=torch.float32).to(self.device)

        # Optimization Pass
        new_logp, entropy, values = self.model.evaluate_actions(obs_t, act_t)
        ratios = torch.exp(new_logp - logp_t)

        surr1 = ratios * adv_t
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = torch.nn.functional.huber_loss(values.squeeze(), rtg_t)
        entropy_loss = -entropy.mean()

        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return total_loss.item(), info["equity"]

    def check_local_minima(self, current_equity: float):
        self.eval_history.append(current_equity)
        if len(self.eval_history) >= 3:
            d1 = abs(self.eval_history[-1] - self.eval_history[-2])
            d2 = abs(self.eval_history[-2] - self.eval_history[-3])

            if d1 < 10.0 and d2 < 10.0:  # Stagnation threshold[cite: 6, 7]
                self.stagnation_counter += 1
                print(f"[⚠️ STAGNATION DETECTED] Executing Stage {self.stagnation_counter} Escape...")

                if self.stagnation_counter == 1:
                    # Stage 1: Cosine LR Warm Restart
                    for g in self.optimizer.param_groups:
                        g['lr'] = 1e-6
                elif self.stagnation_counter == 2:
                    # Stage 2: Boost Entropy Loss Coef 5x
                    self.entropy_coef = 0.05
                elif self.stagnation_counter == 3:
                    # Stage 3: Inject Parametric Noise
                    with torch.no_grad():
                        for param in self.model.actor_head.parameters():
                            param.add_(torch.randn_like(param) * 0.02)
                    self.stagnation_counter = 0

    def _compute_gae(self, rewards, values):
        advantages = []
        gae = 0
        values = values + [0]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        rewards_to_go = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, rewards_to_go
