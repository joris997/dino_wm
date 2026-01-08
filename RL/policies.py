import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DinoCNNHead(nn.Module):
    def __init__(self, embed_dim=404, out_channels=128):
        super().__init__()

        self.conv = nn.Sequential(
            # [B*H, 404, 14, 14]
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # reduce spatial size

    def forward(self, z):
        """
        z: [B, H, 196, 404]
        """
        B, H, P, D = z.shape
        assert P == 196

        z = z.view(B * H, 14, 14, D).permute(0, 3, 1, 2)
        # -> [B*H, 384, 14, 14]

        z = self.conv(z)
        z = self.pool(z)
        # -> [B*H, C, 4, 4]

        z = z.flatten(start_dim=1)
        # -> [B*H, C*4*4]

        z = z.view(B, H, -1)
        return z

class TemporalFlatten(nn.Module):
    def forward(self, z):
        # z: [B, H, F]
        return z.flatten(start_dim=1)

class PolicyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 256)):
        super().__init__()

        layers = []
        dims = (input_dim,) + hidden_dims
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DinoRLStateEncoder(nn.Module):
    def __init__(self, hist_size=3, embed_dim=404):
        print(f"Creating DinoRLStateEncoder with hist_size={hist_size}, embed_dim={embed_dim}")
        super().__init__()

        self.cnn = DinoCNNHead(embed_dim=embed_dim)
        self.temporal = TemporalFlatten()
        
        # compute output dim
        dummy = torch.zeros(1, hist_size, 196, embed_dim)
        with torch.no_grad():
            f = self.cnn(dummy)
            f = self.temporal(f)
        state_dim = f.shape[-1]

        self.mlp = PolicyMLP(state_dim)

    def forward(self, z):
        z = self.cnn(z)
        z = self.temporal(z)
        z = self.mlp(z)
        return z

class DinoRLFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        hist_size,
        embed_dim=384,
    ):
        # observation_space shape: [H, 196, 384]
        super().__init__(observation_space, features_dim=1)

        self.encoder = DinoRLStateEncoder(
            hist_size=hist_size,
            embed_dim=embed_dim,
        )

        # infer feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            feat = self.encoder(dummy)
        self._features_dim = feat.shape[-1]

    def forward(self, obs):
        return self.encoder(obs)
