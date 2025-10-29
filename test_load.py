import os, sys
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import gymnasium as gym

from models.ca_visual_world_model import VWorldModel
from models.ca_vit import ViTPredictor
from models.dino import DinoV3Encoder
from models.proprio import ProprioceptiveEmbedding, ProprioceptiveDecoding
from models.vqvae import VQVAE
from datasets.pusht_dset import PushTDataset
from utils import load_vit

def add_batch_dim(tensor):
    # if tensor is a dict, add batch dim to each value
    if isinstance(tensor, dict):
        return {k: v.unsqueeze(0) for k, v in tensor.items()}
    # else add batch dim to tensor
    return tensor.unsqueeze(0)

folder = '/home/none/gits/dino_wm/outputs'
run = '2025-10-27/15-59-22'
ckpt_folder = os.path.join(folder, run)


world_model, cfg = load_vit(ckpt_folder)


# TODO: this is not the proper one, ignores frameskip, here the control input is 2 instead of 5*2
dataset = PushTDataset(n_rollout=50,
                       transform=None,
                       data_path="datasets/data/pusht_noise/val",
                       normalize_action=True,
                       with_velocity=True)

# TODO: naive CBF/CLF convergence to desired latent state
# get the initial state
obs, act, state, _ = dataset.get_frames(0, range(cfg.frameskip*cfg.num_hist))
# obs = obs[:, ::main_cfg.frameskip, ...]  # subsample according to frameskip
print(f"act.shape: {act.shape}")
# act = rearrange(act, "()
obs = add_batch_dim(obs)
act = add_batch_dim(act)
print('obs["visual"]', obs["visual"].shape)
print('obs["proprio"]', obs["proprio"].shape)
print('act', act.shape)
o, z, _ = world_model.encode(obs, act)

# get a target image, embed it to the feature space and append
# zero control history and proprio history. This is the desired state.
obs_target, _, _, _ = dataset.get_frames(0, range(102, 103))
# give batch dim
obs_target = add_batch_dim(obs_target)
u_target = torch.zeros((1, 2, cfg.action_emb_dim))  # (b, num_hist, action_dim)

print(f"obs_target['visual']: {obs_target['visual'].shape}")
print(f"obs_target['proprio']: {obs_target['proprio'].shape}")
print(f"u_target: {u_target.shape}")
o_target, z_target, _ = world_model.encode(obs_target, u_target)

print(f'z_target: {z_target.shape}')

# Lambda function of the CLF, z_hist[0, -1, ...] to remove batch and take last in history
V = lambda z_hist: torch.norm(z_hist[0, -1, ...] - z_target)**2
dVdx = lambda z_hist: 2 * (z_hist[0, -1, ...] - z_target)

print(f"evaluate V at initial state: {V(z)}")
# display initial image and target image
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].set_title("Initial Image")
axs[0].imshow(obs["visual"][0].permute(1, 2, 0).cpu().numpy())
axs[0].axis("off")
axs[1].set_title("Target Image")
axs[1].imshow(obs_target["visual"][-1,0].permute(1, 2, 0).cpu().numpy())
axs[1].axis("off")
plt.show()

# Define a CLF in the latent space as the 2-norm distance
# between current and desired latent state (with history)

# Create a CLF-QP controller 
# min_u     0.5 u^T H u
# s.t.      dV/dt + c V <= 0
# where dV/dt = (∂V/∂z) (f(z) + g(z) u)

# if V = || z - z_des ||^2
# then ∂V/∂z = 2 (z - z_des)^T