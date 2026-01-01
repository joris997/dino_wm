import gymnasium as gym
from typing import Optional
import numpy as np
import os
import sys
import joblib
import wandb
from wandb.integration.sb3 import WandbCallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_vit
from preprocessor import Preprocessor
from datasets.planarcircle_dset import PlanarCircleDataset
from env.planarcircle.planarcircle_env import PlanarCircleEnv
from env.planarcircle.latent_planarcircle_env import LatentPlanarCircleEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt
import torch

device = 'cuda'

wandb.login()
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000,
    "env_name": "LatentPlanarCircle-v0",
}
# wandb.tensorboard.patch(root_logdir="./outputs/")
run = wandb.init(
    project="latentRL",    # Specify your project
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=False
)

#! Create the environment
# folder = '/home/none/gits/dino_wm/outputs'
folder = '/home/planiacs/gits/dino_wm/outputs'
# run = '2025-12-02/21-40-54'
# run = '2025-12-03/23-14-57'
# run = '2025-12-04/15-39-40'
# run = '2025-12-09/15-10-22'
# run = '2025-12-15/14-28-04' 
# run = '2025-12-15/22-36-40'
# run = '2025-12-22/17-07-49'
run_folder = '2025-12-23/13-32-59'
ckpt_folder = os.path.join(folder, run_folder)

# create 'a' target state
real_env = PlanarCircleEnv(render_mode='rgb_array')
obs = real_env.reset_model()
proprio = real_env._get_obs()



def make_env()->gym.Env:
    """ Utility function for multiprocessed env. """
    real_env = PlanarCircleEnv(render_mode='rgb_array')
    world_model, cfg = load_vit(ckpt_folder)
    world_model.to(device)
    world_model.eval()

    dataset = PlanarCircleDataset(n_rollout=1,
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
    # create a target state:
    # - n_hist images
    # - n_hist the corresponding proprio
    # - u_hist all zeros
    obs, act, state, _ = dataset.get_frames(idx=0, frames=[0])
    obss = {'visual': torch.repeat_interleave(obs['visual'].unsqueeze(0), world_model.local_hist, dim=1).to(device),
            'proprio': torch.repeat_interleave(obs['proprio'].unsqueeze(0), world_model.local_hist, dim=1).to(device)}
    acts = torch.zeros((1,world_model.num_hist, 2*cfg.frameskip), device=device)
    print(f"obss['visual'].shape: {obss['visual'].shape}, obss['proprio'].shape: {obss['proprio'].shape}, acts.shape: {acts.shape}")
    with torch.no_grad():
        _, z, _ = world_model.encode(obss,acts)

    env = LatentPlanarCircleEnv(real_env=real_env,
                                world_model=world_model,
                                preprocessor=data_preprocessor,
                                target_state=z,

                                render_mode='rgb_array',
                                camera_id=0,
                                camera_width=64,
                                camera_height=64)
    env = Monitor(env)
    return env
    

# Nenvs = 1
# env_fns = [lambda i=i: make_env(i) for i in range(Nenvs)]
# env = SubprocVecEnv(env_fns, start_method='fork')
# env = make_env(0)
env = DummyVecEnv([make_env])
env = VecVideoRecorder(env,
                       f"logs/{run.id}/videos",
                       record_video_trigger=lambda x: x % 2000 == 0,
                       video_length=200)

#! RL in the environment
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
n_u = env.action_space.shape[-1]
u_noise = NormalActionNoise(mean=env.env.envs[0].env.world_model.cfg_dict.frameskip*np.zeros(n_u), 
                            sigma=env.env.envs[0].env.world_model.cfg_dict.frameskip*0.1*np.ones(n_u))

# policy_kwargs = dict(net_arch=dict(pi=[64,64], qf=[64,64]))
policy_kwargs = dict(net_arch=dict(pi=[128,128,128], qf=[128,128,128]))

policy = "PPO"
if policy == "TD3":
    model = TD3("MlpPolicy", env, action_noise=u_noise, verbose=0,
                gamma=0.8, #learning_rate=1e-4,
                policy_kwargs=policy_kwargs)
                # tensorboard_log=f"./logs/{run.id}/")
elif policy == "PPO":
    model = PPO("MlpPolicy", env, verbose=0,
                gamma=0.8, #learning_rate=1e-3,
                policy_kwargs=policy_kwargs,device='cuda',
                tensorboard_log=f"logs/{run.id}/runs") 
elif policy == "SAC":
    model = SAC("MlpPolicy", env, action_noise=u_noise, verbose=1,
                gamma=0.8, #learning_rate=1e-3,
                policy_kwargs=policy_kwargs)
                # tensorboard_log=f"./logs/{run.id}/")


print(f"Created model with policy: {model.policy}")
model.learn(total_timesteps=200_000,progress_bar=True,
            callback=WandbCallback(
                 gradient_save_freq=100,
                 model_save_path=f"logs/{run.id}/models",
                 verbose=2)
)
model.save(f"exp_dir/controller/{policy}_latent_dynamics")
wandb.finish()

# # load the model
# if policy == "TD3":
#     model = TD3.load(f"exp_dir/controller/{policy}_latent_dynamics", env=env)
# elif policy == "PPO":
#     model = PPO.load(f"exp_dir/controller/{policy}_latent_dynamics", env=env)
# elif policy == "SAC":
#     model = SAC.load(f"exp_dir/controller/{policy}_latent_dynamics", env=env)
# obs = env.reset()


# #! Plotting
# # reload the environments with a longer horizon as now we don't care about 
# # hallucinations
# from utils.plotting import record_grid_video, record_latent_grid_video
# record_latent_grid_video(env_id="LatentPlanarCircle-v0",
#                         model=model,
#                         video_length=100,
#                         grid_size=(4,4),
#                         filename=f"{policy}_closed_loop.mp4",
#                         video_folder=f"./logs/{run.id}/videos",
#                         plot_reward=True,
#                         plot_value=True,
#                         open_loop=True) # open_loop: use the policy (True), use CLF (False)

