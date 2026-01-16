import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_vit
from preprocessor import Preprocessor
from datasets.planarcircle_dset import PlanarCircleDataset
import torch

device = 'cuda'

folder = '/home/planiacs/gits/dino_wm/outputs'
# run_folder = '2025-12-23/13-32-59' # only A_to_B data
run_folder = '2026-01-08/14-17-51' # A_to_B + biased_brown + white
ckpt_folder = os.path.join(folder, run_folder)
world_model, cfg = load_vit(ckpt_folder)

dataset = PlanarCircleDataset(n_rollout=None,   # all data
                              transform=None,
                              data_path="datasets/data/planarcircle/A_to_B/val",
                              normalize_action=cfg.env.dataset.normalize_action,
                              with_velocity=cfg.env.dataset.with_velocity)

data_preprocessor = Preprocessor(action_mean=dataset.action_mean.to(device),
                                 action_std=dataset.action_std.to(device),
                                 state_mean=dataset.state_mean.to(device),
                                 state_std=dataset.state_std.to(device),
                                 proprio_mean=dataset.proprio_mean.to(device),
                                 proprio_std=dataset.proprio_std.to(device),
                                 transform=dataset.transform)

world_model.to(device)
world_model.eval()

zs = []
for i in range(len(dataset.seq_lengths)):
    j = dataset.seq_lengths[i]
    obs, act, state, _ = dataset.get_frames(idx=i, frames=list(range(j)))

    # obs always have one more frame than act and state
    obs['visual'], obs['proprio'] = obs['visual'][:(j-j%cfg.frameskip),...], obs['proprio'][:(j-j%cfg.frameskip),...]
    act = act[:(j - j % cfg.frameskip),...]

    # downsample by cfg.frameskip
    obs['visual'] = obs['visual'][::cfg.frameskip,...].to(device).unsqueeze(0)
    obs['proprio'] = obs['proprio'][::cfg.frameskip,...].to(device).unsqueeze(0)

    # reshape act from [T, action_dim] to [downsampled_T, frameskip*action_dim]
    act = act.reshape(-1, cfg.frameskip * act.shape[1]).to(device)
    # add another act at the end because it needs to be +1 length
    act = torch.cat([act, torch.zeros((1, act.shape[1]), device=device)], dim=0).unsqueeze(0)

    with torch.no_grad():
        _, z, _ = world_model.encode(obs, act)
    z = z.squeeze(0)  # remove batch dimension

    zs.append(z)
 
zs = torch.cat(zs, dim=0)

zs_mean = zs.mean(dim=0)
zs_std = zs.std(dim=0)

# save the latent std to a file
save_path = os.path.join(ckpt_folder, 'latent_std.npy')
np.save(save_path, zs_std.cpu().numpy())
# save the latent mean to a file
save_path = os.path.join(ckpt_folder, 'latent_mean.npy')
np.save(save_path, zs_mean.cpu().numpy())