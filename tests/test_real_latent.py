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
from datasets.img_transforms import default_transform
from env.pusht.pusht_env import PushTEnv
from preprocessor import Preprocessor
from utils import load_vit

folder = '/home/none/gits/dino_wm/outputs'
# run = '2025-12-02/21-40-54'
# run = '2025-12-03/23-14-57'
# run = '2025-12-04/15-39-40'
run = '2025-12-09/15-10-22'
ckpt_folder = os.path.join(folder, run)

# world model
world_model, cfg = load_vit(ckpt_folder)
world_model.to('cuda')
cfg.debug = True

# dataset
dataset = PushTDataset(n_rollout=50,
                       transform=default_transform(cfg.img_size),
                       data_path="datasets/data/pusht_noise/val",
                       normalize_action=cfg.env.dataset.normalize_action,
                       with_velocity=cfg.env.dataset.with_velocity)
# get a range of data
# obs: dict, ['visual']: [100, C, H, W], ['proprio']: [100, P]
# act: [100, A]
obs, act, state, _ = dataset.get_frames(4, range(100))

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

# We only get observations every 'frameskip' frames, so subsample the obs
obs_skip = {
    'visual': obs['visual'][::cfg.frameskip,...].unsqueeze(0),
    'proprio': obs['proprio'][::cfg.frameskip,...].unsqueeze(0)
}
# Initial history: [num_hist, C, H, W], [num_hist, P]
# Initial actions: [num_hist-1, frameskip * A]
obs0 = {'visual': obs_skip['visual'][:,:cfg.num_hist,...].to('cuda'),
        'proprio': obs_skip['proprio'][:,:cfg.num_hist,...].to('cuda')}

# act is now [100,2], need to do frameskip
# so for each i in 0 to 100//frameskip, take act[i*frameskip : (i+1)*frameskip,:]
# and reshape to (1, 100//frameskip, frameskip * action_dim)
act_fs = []
for i in range(0,100//cfg.frameskip-1):
    acti = act[i*cfg.frameskip : (i+1)*cfg.frameskip,:]
    acti = acti.reshape(1, -1)  # reshape to (1, frameskip * action_dim)
    act_fs.append(acti)
act_fs = torch.cat(act_fs, dim=0).to('cuda')  # shape now (100//frameskip, frameskip * action_dim)

# rollout in the world model and decode observations
with torch.no_grad():
    obsz, _ = world_model.rollout(obs0, act_fs.unsqueeze(0))
    obss, _ = world_model.decode_obs(obsz)
print(f"obss['visual'].shape: {obss['visual'].shape}, obss['proprio'].shape: {obss['proprio'].shape}")

# # plot all obss
# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# for i in range(obss['visual'].shape[1]):
#     axs[0].imshow(obs['visual'][(i+1)*cfg.frameskip].permute(1, 2, 0).cpu().numpy())
#     axs[0].axis("off")
#     axs[1].imshow(obss['visual'][0,i].permute(1, 2, 0).cpu().numpy())
#     axs[1].axis("off")
#     plt.suptitle(f"Step {i}: Real | Predicted")
#     plt.draw()
#     plt.waitforbuttonpress()




obsi_hist = {'visual': obs_skip['visual'][:,:cfg.num_hist,...].to('cuda'),
             'proprio': obs_skip['proprio'][:,:cfg.num_hist,...].to('cuda')}
acti_hist_raw = act[0 : (cfg.num_hist-1)*cfg.frameskip,:]
print(f"indexing act with: {0} : {(cfg.num_hist-1)*cfg.frameskip}")
acti_hist = acti_hist_raw.reshape(1, cfg.num_hist-1, 2*cfg.frameskip).to('cuda')
print(f"acti_hist.shape: {acti_hist.shape}")

# perform the acti_hist in the real environment to get the initial observation
act_denorm = np.array([data_preprocessor.denormalize_actions(a).numpy() for a in acti_hist_raw])
for a in act_denorm:
    observation, _, _, _ = env.step(a)

# # now loop through the dataset and create frameskip control input
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# obsi_hist, acti_hist = obs0, act0
# print(f"obsi_hist.shape: {obsi_hist['visual'].shape}, {obsi_hist['proprio'].shape}")
# print(f"acti_hist.shape: {acti_hist.shape}")

for i in range(0,100//cfg.frameskip-1):
    acti = act[(i)*cfg.frameskip+(cfg.num_hist-1)*cfg.frameskip : (i+1)*cfg.frameskip+(cfg.num_hist-1)*cfg.frameskip,:]
    print(f"\n=== Step {i} ===")
    print(f"indexing act with: {(i)*cfg.frameskip+(cfg.num_hist-1)*cfg.frameskip} : {(i+1)*cfg.frameskip+(cfg.num_hist-1)*cfg.frameskip}")

    # take the step in the real environment
    act_denorm = np.array([data_preprocessor.denormalize_actions(a).numpy() for a in acti])
    for a in act_denorm:
        observation, _, _, _ = env.step(a)
    image = env.render('rgb_array')
    axs[0].imshow(image)
    axs[0].axis("off")

    # take the next step
    # make right shape and set to cuda
    acti = acti.reshape(1, -1)          # reshape to (1, frameskip * action_dim)
    print(f"acti.shape: {acti.shape}")
    acti_hist = torch.cat([acti_hist, torch.tensor(acti, device='cuda').unsqueeze(0)], dim=1)
    obsi_hist = {key: value.to('cuda') for key, value in obsi_hist.items()}
    acti_hist = acti_hist.to('cuda')

    # take the step in the world model
    with torch.no_grad():
        obs_pred, z_pred, dz_pred, obs_now = world_model.take_step(obsi_hist, acti_hist)
    
    obs_pred_vis = obs_now['visual'].cpu().detach()
    # obs_pred_vis = obs_pred['visual'].cpu().detach()
    # create all images in obs_pred_vis in a row
    imgs = []
    for j in range(obs_pred_vis.shape[1]):
        imgs.append(obs_pred_vis[0,j].permute(1, 2, 0).cpu().numpy())
    obs_pred_vis_row = np.concatenate(imgs, axis=1)
    axs[1].imshow(np.clip(obs_pred_vis_row, 0, 1))
    # axs[1].imshow(np.clip(obs_pred_vis[0,-1].permute(1, 2, 0).cpu().numpy(), 0, 1))
    axs[1].axis("off")
    plt.suptitle(f"Step {i}: Real | Predicted | Reconstructed \
                 \nProprioception Predicted: {obs_pred['proprio'][0,-1].cpu().numpy()}")
    plt.draw()
    plt.waitforbuttonpress()

    # update history
    print(f"obsi_hist['visual'].shape: {obsi_hist['visual'].shape},    obs_pred['visual'].shape: {obs_pred['visual'].shape}")
    print(f"obsi_hist['proprio'].shape: {obsi_hist['proprio'].shape},  obs_pred['proprio'].shape: {obs_pred['proprio'].shape}")
    obsi_hist = {'visual': torch.cat([obsi_hist['visual'][:,1:,...], obs_pred['visual'][:,-1:,...].to('cuda')], dim=1),
                 'proprio': torch.cat([obsi_hist['proprio'][:,1:,...], obs_pred['proprio'][:,-1:,...].to('cuda')], dim=1)}
    acti_hist = acti_hist[:,1:,...]

