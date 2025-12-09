import torch 
import torch.nn as nn
import torch.nn.functional as F

def mdn_loss(logits, means, logstds, targets):
    u = targets.unsqueeze(1)  # (B, 1, action_dim)
    var = torch.exp(2 * logstds)  # (B, num_guassians, action_dim)

    log_prob = -0.5 * (
        means.size(-1) * torch.log(torch.tensor([2 * torch.pi],device=targets.device)) + 
        torch.sum(2*logstds + (u - means)**2 / var, dim=-1)
    )

    log_pi = F.log_softmax(logits, dim=-1)  # (B, num_guassians)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B,)

    return -torch.mean(log_mix)


class BehavioralCloner(nn.Module):
    def __init__(self, *, num_hist, action_dim, num_gaussians=5, hidden=128):
        super().__init__()
        self.h = num_hist - 1 # because of the control affine structure
        self.p = 196
        self.l = 404
        self.token_dim = self.h * self.l  # flatten per token
        self.a = action_dim
        self.K = num_gaussians

        # Token-wise MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.token_dim + action_dim),
            nn.Linear(self.token_dim + action_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )

        # Pool over tokens
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MDN heads (small!)
        self.fc_logits  = nn.Linear(hidden, self.K)
        self.fc_means   = nn.Linear(hidden, self.K * self.a)
        self.fc_logstds = nn.Linear(hidden, self.K * self.a)

    def forward(self, z, u_now):
        """
        z:      [B, h, p, l]
        u_now:  [B, p, a]
        """
        B = z.shape[0]

        # flatten per token
        z = z.permute(0, 2, 1, 3).reshape(B, self.p, self.token_dim)  # [B,p,h*l]

        # concat tokenwise action
        x = torch.cat([z, u_now], dim=-1)        # [B,p,h*l + a]

        # token-wise MLP
        h = self.mlp(x)                          # [B,p,hidden]

        # pool over tokens
        h_global = h.mean(dim=1)                 # [B,hidden]

        # MDN outputs
        logits  = self.fc_logits(h_global)                     # [B,K]
        means   = self.fc_means(h_global).view(B, self.K, self.a)
        logstds = self.fc_logstds(h_global).view(B, self.K, self.a)

        return logits, means, logstds


# class BehavioralCloner(nn.Module):
#     def __init__(self, *, num_hist, action_dim, num_guassians=5, hidden=128):
#         super().__init__()

#         self.input_dim = (num_hist-1) * 196 * 404 + action_dim

#         self.trunk = nn.Sequential(
#             nn.LayerNorm(self.input_dim),
#             nn.Linear(self.input_dim, self.input_dim//2),
#             nn.ELU(),
#             nn.Linear(self.input_dim//2, hidden),
#             nn.ELU(),
#             nn.Linear(hidden, hidden),
#             nn.ELU(),
#         )

#         # each token outputs mixture params
#         self.fc_logits = nn.Linear(hidden, num_guassians)  # mixture logits
#         self.fc_means = nn.Linear(hidden, num_guassians * action_dim)  # mixture means
#         self.fc_logstds = nn.Linear(hidden, num_guassians * action_dim)  # mixture log stds

#     def forward(self, z, u_now):
#         # INPUTS:
#         # z: (B, num_hist, 196, 404)
#         # u_now: (B, frameskip * action_dim)
#         # OUTPUTS:
#         # logits: (B, num_guassians)
#         # means: (B, num_guassians, action_dim)
#         # logstds: (B, num_guassians, action_dim)

#         B = z.shape[0]
#         z_flat = z.view(B, -1)  # (B, num_hist * 404)
#         x = torch.cat([z_flat, u_now], dim=-1)  # (B, num_hist * 404 + frameskip * action_dim)

#         h = self.trunk(x)  # (B, hidden)

#         logits = self.fc_logits(h)  # (B, num_guassians)
#         means = self.fc_means(h).view(B, -1, u_now.shape[-1])  # (B, num_guassians, action_dim)
#         logstds = self.fc_logstds(h).view(B, -1, u_now.shape[-1])  # (B, num_guassians, action_dim)

#         return logits, means, logstds