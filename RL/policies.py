from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch 
import torch.nn as nn

class LatentCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        C, T, E = observation_space.shape  # (3,196,404) if using CxHxW naming

        self.cnn = nn.Sequential(
            # (B, 3, 196, 404) → (B, 404, 3, 196)
            nn.Identity(),  # permute in forward

            # Project embedding dimension
            nn.Conv2d(404, 64, kernel_size=1),
            nn.ReLU(),

            # Token/time processing
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(self._permute(sample)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def _permute(self, x):
        # (B, 3, 196, 404) → (B, 404, 3, 196)
        return x.permute(0, 3, 1, 2)

    def forward(self, observations):
        x = self._permute(observations)
        return self.linear(self.cnn(x))
