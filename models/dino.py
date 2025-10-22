import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

# TODO: do this the nice way with hydra
REPO_DIR = "/home/joris/gits/dinov3"
WEIGHTS  = "/home/joris/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb

class DinoV3Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        # self.base_model = torch.hub.load("facebookresearch/dinov3", name)
        self.base_model = torch.hub.load(REPO_DIR, name, source='local', weights=WEIGHTS)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb