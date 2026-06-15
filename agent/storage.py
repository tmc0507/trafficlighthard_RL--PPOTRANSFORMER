# storage.py
import torch

class RolloutStorage:
    def __init__(self, num_steps: int, obs_dim: int, num_envs: int, device):
        # signature: (num_steps, obs_dim, num_envs, device)
        self.device = device
        self.num_steps = num_steps
        self.step = 0
        self.num_envs = num_envs
        self.obs_dim = obs_dim

        # shapes: (num_steps, num_envs, ...)
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.logp = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

    def clear(self):
        self.step = 0

    def add(self, obs, action, logp, reward, done, value):
        """
        obs: tensor (num_envs, obs_dim)
        action: tensor (num_envs,)
        logp: tensor (num_envs,)
        reward: tensor (num_envs,)
        done: tensor (num_envs,)
        value: tensor (num_envs,)
        """
        if self.step >= self.num_steps:
            raise RuntimeError("RolloutStorage overflow")
        # copy along envs axis
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(action)
        self.logp[self.step].copy_(logp)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)
        self.step += 1

    def compute_returns(self, last_value, gamma: float, lam: float):
        """
        last_value: tensor (num_envs,)
        compute GAE per env separately
        """
        # ensure shapes
        gae = torch.zeros(self.num_envs, device=self.device)
        for env_idx in range(self.num_envs):
            gae_env = 0.0
            for t in reversed(range(self.step)):
                if t == self.step - 1:
                    next_value = last_value[env_idx]
                    next_non_terminal = 1.0 - self.dones[t, env_idx]
                else:
                    next_value = self.values[t + 1, env_idx]
                    next_non_terminal = 1.0 - self.dones[t + 1, env_idx]

                delta = self.rewards[t, env_idx] + gamma * next_value * next_non_terminal - self.values[t, env_idx]
                gae_env = delta + gamma * lam * next_non_terminal * gae_env
                self.advantages[t, env_idx] = gae_env

            # returns = advantages + values
            self.returns[: self.step, env_idx] = self.advantages[: self.step, env_idx] + self.values[: self.step, env_idx]

        # normalize advantages across (time * env)
        adv = self.advantages[: self.step].reshape(-1)
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False)
        if adv_std.item() == 0:
            adv_std = adv_std + 1e-8
        self.advantages[: self.step] = (self.advantages[: self.step] - adv_mean) / (adv_std + 1e-8)

    def mini_batches(self, num_mini_batches: int):
        """
        Yield mini-batches flattening (time, env) -> batch
        returns obs_mb: (mb_size, obs_dim), actions_mb: (mb_size,), etc.
        """
        batch_size = self.step * self.num_envs
        mini_batch_size = max(1, batch_size // num_mini_batches)

        # flatten first two dims
        obs_flat = self.obs[: self.step].reshape(batch_size, self.obs_dim)
        actions_flat = self.actions[: self.step].reshape(batch_size)
        logp_flat = self.logp[: self.step].reshape(batch_size)
        returns_flat = self.returns[: self.step].reshape(batch_size)
        adv_flat = self.advantages[: self.step].reshape(batch_size)
        values_flat = self.values[: self.step].reshape(batch_size)

        indices = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb = indices[start:end]
            yield (
                obs_flat[mb],
                actions_flat[mb],
                logp_flat[mb],
                returns_flat[mb],
                adv_flat[mb],
                values_flat[mb],
            )