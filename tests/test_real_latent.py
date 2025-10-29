import os, sys
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import gymnasium as gym

# add .. to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ca_visual_world_model import VWorldModel
from models.ca_vit import ViTPredictor
from models.dino import DinoV3Encoder
from models.proprio import ProprioceptiveEmbedding, ProprioceptiveDecoding
from models.vqvae import VQVAE
from datasets.pusht_dset import PushTDataset
from env.pusht.pusht_env import PushTEnv
from preprocessor import Preprocessor
from utils import load_vit

folder = '/home/none/gits/dino_wm/outputs'
run = '2025-10-27/16-34-58'
ckpt_folder = os.path.join(folder, run)

# world model
world_model, cfg = load_vit(ckpt_folder)
world_model.to('cuda')
cfg.debug = True

# dataset
dataset = PushTDataset(n_rollout=50,
                       transform=None,
                       data_path="datasets/data/pusht_noise/val",
                       normalize_action=True,
                       with_velocity=True)
obs, act, state, _ = dataset.get_frames(0, range(100))

# preprocessor to denormalize actions/states/proprios
data_preprocessor = Preprocessor(action_mean=dataset.action_mean,
                                 action_std=dataset.action_std,
                                 state_mean=dataset.state_mean,
                                 state_std=dataset.state_std,
                                 proprio_mean=dataset.proprio_mean,
                                 proprio_std=dataset.proprio_std,
                                 transform=dataset.transform)

# mujoco environment
env = PushTEnv(reset_to_state=state[0].numpy())
observation, info = env.reset()

# create initial state. first create num_hist-1 history with frameskip
obs_skip = {
    'visual': obs['visual'][::cfg.frameskip,...].unsqueeze(0),
    'proprio': obs['proprio'][::cfg.frameskip,...].unsqueeze(0)
}
obs0 = {'visual': obs_skip['visual'][:,:cfg.num_hist-1,...],
        'proprio': obs_skip['proprio'][:,:cfg.num_hist-1,...]}
act0 = torch.zeros((1, cfg.num_hist-1, cfg.frameskip*env.action_space.shape[0]),device='cuda')

# now loop through the dataset and create frameskip control input
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
obsi_hist, acti_hist = obs0, act0
print(f"obsi_hist.shape: {obsi_hist['visual'].shape}, acti_hist.shape: {acti_hist.shape}")

for i in range(0,int(100/cfg.frameskip)):
    acti = act[i*cfg.frameskip : (i+1)*cfg.frameskip,:]

    # take the step in the real environment
    act_denorm = np.array([data_preprocessor.denormalize_actions(a).numpy() for a in acti])
    for a in act_denorm:
        observation, _, _, _ = env.step(a)
    image = env.render('rgb_array')

    # make right shape and set to cuda
    acti = acti.reshape(1, -1)          # reshape to (1, frameskip * action_dim)
    acti_hist = torch.cat([acti_hist, torch.tensor(acti, device='cuda').unsqueeze(1)], dim=1)
    print(f"acti.shape after reshape: {acti.shape}")
    obsi_hist = {'visual': obsi_hist['visual'].to('cuda'), 'proprio': obsi_hist['proprio'].to('cuda')}
    acti_hist = acti_hist.to('cuda')

    # take the step in the world model
    with torch.no_grad():
        obs_pred, z_pred, dz_pred = world_model.take_step(obsi_hist, acti_hist)

    # plotting 
    axs[0].imshow(image)
    axs[0].axis("off")
    obs_pred_vis = obs_pred['visual'].cpu().detach()
    axs[1].imshow(obs_pred_vis[0,-1].permute(1, 2, 0).cpu().numpy())
    axs[1].axis("off")
    plt.suptitle(f"Step {i}: Real | Predicted | Reconstructed")
    plt.pause(0.5)

    # update history
    # TODO: proprioception decoder?
    obsi_hist = {'visual': torch.cat([obsi_hist['visual'][:,1:,...], obs_pred['visual'][:,-1:,...].to('cuda')], dim=1),
                 'proprio': torch.cat([obsi_hist['proprio'][:,1:,...], obsi_hist['proprio'][:,-1:,...].to('cuda')], dim=1)}
    acti_hist = acti_hist[:,1:,...]

