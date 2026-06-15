import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticTransformer(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, 
                 seq_len: int = 15,   # số chiều tuần tự (ví dụ phần "scan", hoặc các feature đặc biệt)
                 extra_dim: int = 6,  # phần còn lại (global features như state, target,...)
                 d_model: int = 32, n_layers: int = 2, n_heads: int = 2):
        super().__init__()
        assert seq_len + extra_dim == obs_dim
        self.seq_len = seq_len      # số chiều đầu chuỗi (dạng scan, các feature tuần tự)
        self.extra_dim = extra_dim  # phần còn lại

        # --- Embedding cho chuỗi tuần tự ---
        self.seq_embed = nn.Linear(1, d_model)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Embedding cho extra/global features ---
        self.extra_fc = nn.Linear(extra_dim, 64)
        
        # --- Fusion + Actor/Critic ---
        fusion_dim = seq_len * d_model + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256), # Giúp ổn định Gradient khi train Transformer
            nn.ReLU(),
            nn.Dropout(0.1)    # Chống Overfitting nếu dữ liệu ít
        )

        self.actor = nn.Linear(256, act_dim)
        self.critic = nn.Linear(256, 1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """
        obs: (obs_dim,) or (batch, obs_dim)
        returns: action, logp, entropy, value
        """
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)

        policy_logits, value = self.forward(obs)
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()

        if single:
            return action[0], logp[0], entropy[0], value[0]
        return action, logp, entropy, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        For PPO update step.
        obs: (batch, obs_dim)
        actions: (batch,)
        returns: logp, entropy, value
        """
        policy_logits, value = self.forward(obs)
        dist = Categorical(logits=policy_logits)
        actions = actions.to(policy_logits.device)   # <-- Quan trọng! ép cùng device
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value

    @torch.no_grad()
    def value(self, obs: torch.Tensor):
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        _, value = self.forward(obs)
        if single:
            return value[0]
        return value

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        - self.seq_len: số chiều đầu tiên sẽ coi là chuỗi tuần tự cho Attention (ví dụ scan, vector cảm biến)
        - self.extra_dim: các features còn lại (state, global info, ...).
        """
        device = next(self.parameters()).device
        obs = obs.to(device)
        # 1. Chuỗi feature tuần tự cho Attention
        seq_input = obs[:, :self.seq_len].unsqueeze(-1)         # (B, seq_len, 1)
        extra_input = obs[:, self.seq_len:]                     # (B, extra_dim)
        # 2. Embedding & Attention
        seq_emb = self.seq_embed(seq_input)                     # (B, seq_len, d_model)
        seq_ctx = self.transformer(seq_emb)                     # (B, seq_len, d_model)
        seq_feat = seq_ctx.flatten(start_dim=1)                 # (B, seq_len * d_model)
        # 3. Extra features
        extra_feat = self.extra_fc(extra_input)                 # (B, 64)
        # 4. Fusion
        x = torch.cat([seq_feat, extra_feat], dim=-1)           # (B, seq_len*d_model + 64)
        x = self.fusion(x)                                      # (B, 256)
        logits = self.actor(x)                                  # (B, act_dim)
        value = self.critic(x).squeeze(-1)                      # (B,)
        return logits, value