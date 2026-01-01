import os, sys
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import gymnasium as gym

# add .. to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.insert(0, str(Path("/home/planiacs/gits/latentRL")))
from datasets.pusht_dset import PushTDataset
from datasets.planarcircle_dset import PlanarCircleDataset
from datasets.img_transforms import default_transform
from env.pusht.pusht_env import PushTEnv
from env.planarcircle.planarcircle_env import PlanarCircleEnv
from preprocessor import Preprocessor
from utils import load_vit

def fifo_append(tensor:torch.tensor, new_element:torch.tensor, 
                dim:int, max_size:int):
    y = torch.cat([tensor, new_element], dim=dim)
    if y.shape[dim] > max_size:
        y = y.narrow(dim, y.shape[dim] - max_size, max_size)
    return y

# folder = '/home/none/gits/dino_wm/outputs'
folder = '/home/planiacs/gits/dino_wm/outputs'
# run = '2025-12-02/21-40-54'
# run = '2025-12-03/23-14-57'
# run = '2025-12-04/15-39-40'
# run = '2025-12-09/15-10-22'
# run = '2025-12-15/14-28-04' 
# run = '2025-12-15/22-36-40'
# run = '2025-12-22/17-07-49'
run = '2025-12-23/13-32-59'
ckpt_folder = os.path.join(folder, run)

# world model
world_model, cfg = load_vit(ckpt_folder)
world_model.to('cuda')
cfg.debug = False

# dataset
# dataset = PushTDataset(n_rollout=50,
#                        transform=default_transform(cfg.img_size),
#                        data_path="datasets/data/planarcircle/A_to_B/val",
#                        normalize_action=cfg.env.dataset.normalize_action,
#                        with_velocity=cfg.env.dataset.with_velocity)
dataset = PlanarCircleDataset(n_rollout=100,
                              transform=None,#default_transform(cfg.img_size),
                              data_path="datasets/data/planarcircle/A_to_B/val",
                              normalize_action=cfg.env.dataset.normalize_action,
                              with_velocity=cfg.env.dataset.with_velocity)
# get a range of data
# obs: dict, ['visual']: [100, C, H, W], ['proprio']: [100, P]
# act: [100, A]
obs, act, state, _ = dataset.get_frames(5, range(100))

# preprocessor to denormalize actions/states/proprios
data_preprocessor = Preprocessor(action_mean=dataset.action_mean,
                                 action_std=dataset.action_std,
                                 state_mean=dataset.state_mean,
                                 state_std=dataset.state_std,
                                 proprio_mean=dataset.proprio_mean,
                                 proprio_std=dataset.proprio_std,
                                 transform=dataset.transform)

# We only get observations every 'frameskip' frames, so subsample the obs
obs_skip = {
    'visual': obs['visual'][::cfg.frameskip,...].unsqueeze(0),
    'proprio': obs['proprio'][::cfg.frameskip,...].unsqueeze(0)
}
# Initial history: [num_hist, C, H, W], [num_hist, P]
# Initial actions: [num_hist-1, frameskip * A]
obs0 = {'visual': obs_skip['visual'][:,:cfg.num_hist,...].to('cuda'),
        'proprio': obs_skip['proprio'][:,:cfg.num_hist,...].to('cuda')}








#! plot all obss
# denormalize state and action so that we can rollout the actions in the env
state_denorm = data_preprocessor.denormalize_states(state)

# act is now [100,2], need to do frameskip
# so for each i in 0 to 100//frameskip, take act[i*frameskip : (i+1)*frameskip,:]
# and reshape to (1, 100//frameskip, frameskip * action_dim)
act_fs = []
act_env = []
for i in range(0,100//cfg.frameskip-1):
    acti = act[i*cfg.frameskip : (i+1)*cfg.frameskip,:]
    act_fs.append(acti.reshape(1,-1)) # reshape to (1, frameskip * action_dim)
    act_env.append(data_preprocessor.denormalize_actions(acti))
act_fs = torch.cat(act_fs, dim=0).to('cuda')  # shape now (100//frameskip, frameskip * action_dim)
act_env = torch.cat(act_env, dim=0).numpy()

# rollout in the real env to get the real observations
env = PlanarCircleEnv(reset_to_state=state_denorm[0].numpy(),render_mode='rgb_array')
observation, info = env.reset()
obsenv = []
for i in range(len(act_env)):
    observation, _, _, _ = env.step(act_env[i])
    obsenv.append(env.render('rgb_array'))  # no

# rollout in the world model and decode observations
with torch.no_grad():
    obsz, _ = world_model.rollout(obs0, act_fs.unsqueeze(0))
    obss, _ = world_model.decode_obs(obsz)
print(f"obss['visual'].shape: {obss['visual'].shape}, obss['proprio'].shape: {obss['proprio'].shape}")

fig, axs = plt.subplots(1, 3, figsize=(8, 4))
for i in range(obss['visual'].shape[1]):
    a = obss['visual'][0,i].permute(1, 2, 0).cpu().numpy()
    axs[0].imshow(obsenv[(i+1)*cfg.frameskip])
    axs[0].axis("off")
    axs[1].imshow(obs['visual'][(i+1)*cfg.frameskip].permute(1, 2, 0).cpu().numpy())
    axs[1].axis("off")
    axs[2].imshow((obss['visual'][0,i].permute(1, 2, 0).cpu().numpy()+1)/2)
    axs[2].axis("off")
    plt.suptitle(f"Step {i}: Real | Dataset | Predicted")
    plt.draw()
    plt.waitforbuttonpress()








#! Test the rollout in the world model and in the real env
print(f"num_hist: {cfg.num_hist}, frameskip: {cfg.frameskip}")
state_denorm = data_preprocessor.denormalize_states(state)
env = PlanarCircleEnv(reset_to_state=state_denorm[0].numpy(),render_mode='rgb_array')
observation, info = env.reset()

fig, axs = plt.subplots(1, 3, figsize=(8, 4))
acti_hist = torch.tensor([], device='cuda')  # shape (1, 0, action_dim)
for i in range(0, 100//cfg.frameskip-1):
    acti = act[(i)*cfg.frameskip : (i+1)*cfg.frameskip,:]
    print(f"\n=== Step {i} ===")
    print(f"indexing act with: {(i)*cfg.frameskip} : {(i+1)*cfg.frameskip}")

    # first take all the num_hist steps in the real env
    # so that we have the correct observation to start with
    if i < cfg.num_hist:
        # apply the first acti to the env
        for a in data_preprocessor.denormalize_actions(acti).numpy():
            print("taking step")
            observation, _, _, _ = env.step(a)
        acti = acti.reshape(1, -1)
        acti_hist = fifo_append(acti_hist, acti.unsqueeze(0).to('cuda'), 
                                dim=1, max_size=cfg.num_hist-1)
        obsi_hist = {'visual':  obs_skip['visual'][:,1:cfg.num_hist+1,...].to('cuda'),
                     'proprio': obs_skip['proprio'][:,1:cfg.num_hist+1,...].to('cuda')}
        # append the action to acti_hist so that we can obtain the 
        # latent state as well. We can do control in that space
        # without accumulating error from encoder/decoders
        acti_hist_with_current = fifo_append(acti_hist, acti.unsqueeze(0).to('cuda'), 
                                            dim=1, max_size=cfg.num_hist)
        _, z_hist, _ = world_model.encode(obsi_hist, acti_hist)
    else:
        # take the step in the real environment
        images = []
        for a in data_preprocessor.denormalize_actions(acti).numpy():
            print("taking step")
            observation, _, _, _ = env.step(a)
            # images.append(env.render('rgb_array'))
        # axs[0].imshow(np.concatenate(images, axis=1))
        image = env.render('rgb_array')
        axs[0].imshow(image)
        axs[0].axis("off")

        # plot the dataset observation
        # print(f"indexing obs with: {(i+1)*cfg.frameskip-1}")
        obs_data_vis = obs['visual'][(i+1)*cfg.frameskip]
        # obs_data_vis = obs_skip['visual'][0,i]
        axs[1].imshow(obs_data_vis.permute(1, 2, 0).cpu().numpy())
        axs[1].axis("off")

        # take the next step
        # make right shape and set to cuda
        acti = acti.reshape(1, -1)          # reshape to (1, frameskip * action_dim)
        acti_hist = fifo_append(acti_hist, acti.detach().clone().to('cuda').unsqueeze(0), 
                                dim=1, max_size=cfg.num_hist)
        # take the step in the world model
        with torch.no_grad():
            obs_pred, z_pred, dz_pred, obs_now = world_model.take_step(obsi_hist, acti_hist)
        # obs_pred_vis = obs_now['visual'].cpu().detach()
        obs_pred_vis = obs_pred['visual'].cpu().detach()
   

        # create all images in obs_pred_vis in a row
        imgs = []
        for j in range(obs_pred_vis.shape[1]):
            imgs.append(obs_pred_vis[0,j].permute(1, 2, 0).cpu().numpy())
        obs_pred_vis_row = np.concatenate(imgs, axis=1)
        axs[2].imshow(np.clip((obs_pred_vis_row+1)/2, 0, 1))
        # axs[2].imshow(np.clip(obs_pred_vis[0,-1].permute(1, 2, 0).cpu().numpy(), 0, 1))
        axs[2].axis("off")
        plt.suptitle(f"Step {i}: Real | Dataset | Predicted ")#\
                    # \nProprioception Predicted: {obs_pred['proprio'][0,-1].cpu().numpy()}")
        plt.draw()
        plt.waitforbuttonpress()


        obsi_hist = {key: fifo_append(obsi_hist[key], obs_pred[key][:,-1:,...].to('cuda'), 
                                      dim=1, max_size=cfg.num_hist-1) for key in obsi_hist.keys()}
        # update history
        # print(f"obsi_hist['visual'].shape: {obsi_hist['visual'].shape},    obs_pred['visual'].shape: {obs_pred['visual'].shape}")
        # print(f"obsi_hist['proprio'].shape: {obsi_hist['proprio'].shape},  obs_pred['proprio'].shape: {obs_pred['proprio'].shape}")
        # obsi_hist = {'visual': torch.cat([obsi_hist['visual'][:,1:,...], obs_pred['visual'][:,-1:,...].to('cuda')], dim=1),
        #              'proprio': torch.cat([obsi_hist['proprio'][:,1:,...], obs_pred['proprio'][:,-1:,...].to('cuda')], dim=1)}
        # obsi_hist = {'visual': torch.cat([obsi_hist['visual'][:,1:,...], obs_pred['visual'][:,-1:,...].to('cuda')], dim=1),
        #             'proprio': torch.cat([obsi_hist['proprio'][:,1:,...], data_preprocessor.normalize_proprios(torch.tensor(observation.reshape(1,1,-1))).to('cuda')], dim=1)}
        # acti_hist = acti_hist[:,1:,...]





# #! test rollout in env and from data, should be the same
# state_denorm = data_preprocessor.denormalize_states(state)
# action_denorm = data_preprocessor.denormalize_actions(act)

# env = PlanarCircleEnv(reset_to_state=state_denorm[0].numpy(),render_mode='rgb_array')
# observation, info = env.reset()
# image = env.render('rgb_array')

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# for i in range(len(action_denorm)):
#     observation, _, _, _ = env.step(action_denorm[i].numpy())
#     image = env.render('rgb_array')

#     axs[0].imshow(obs['visual'][i].permute(1, 2, 0).cpu().numpy())
#     axs[1].imshow(image)
#     axs[0].set_xlabel(f"Action: {act[i].numpy()}")
#     axs[1].set_xlabel(f"Action: {action_denorm[i].numpy()}")
#     plt.suptitle(f"Step {i}: Dataset | Env")
#     plt.draw()
#     plt.waitforbuttonpress()