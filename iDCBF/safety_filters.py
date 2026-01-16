import torch
import torch.nn as nn

class LatentIDBF(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))  # output is scalar B(z)
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # z shape: [B, latent_dim]
        return self.net(z).squeeze(-1)  # shape [B]
