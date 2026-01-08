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
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise

from RL.main import make_env

import matplotlib.pyplot as plt
import torch
import cv2

env = make_env(step_in_real_env=True, 
               average_pool=False,
               plot_target=False)

folder = '/home/planiacs/gits/dino_wm/outputs/RL'
run_folder = '2026-01-06/15-05-15'
ckpt_folder = os.path.join(folder, run_folder, 'models')

# load the model
policy = "PPO"
if policy == "TD3":
    model = TD3.load(f"outputs/RL/{run_folder}/models/{policy}_latent_dynamics", env=env)
elif policy == "PPO":
    model = PPO.load(f"outputs/RL/{run_folder}/models/{policy}_latent_dynamics", env=env)
elif policy == "SAC":
    model = SAC.load(f"outputs/RL/{run_folder}/models/{policy}_latent_dynamics", env=env)
z, _ = env.reset()

# evaluate the model 
zs, obss, rewards, dones = [], [], [], []
for i in range(100):
    action, _ = model.predict(z, deterministic=True)
    # z, reward, done, _, _ = env.step(action)
    z, reward, done, _, _ = env.step(action)#, step_in_real_env=False)
    obs = env.render()
    print(f"Step {i}: reward={reward}, done={done}")
    if i % (env.env._max_episode_steps) == 0:
        # create red border for the last frame
        obs[:5,:,:] = [255,0,0]
        obs[-5:,:,:] = [255,0,0]
        obs[:,:5,:] = [255,0,0]
        obs[:,-5:,:] = [255,0,0]
        z, _ = env.reset()
        print("Resetting environment")

    zs.append(z)
    obss.append(obs)
    rewards.append(reward)
    dones.append(done)

# plot the obss rewards and dones as a video
height, width, _ = obss[0].shape
print(f"Video shape: {len(obss)} frames of size {width}x{height}")
video = cv2.VideoWriter(os.path.join(folder, run_folder,'runs/eval_latent_planarcircle.mp4'), 
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        10, 
                        (width, height))
for i in range(len(obss)):
    frame = cv2.cvtColor(obss[i], cv2.COLOR_RGB2BGR)
    video.write(frame)
video.release()

fig, axs = plt.subplots(4,25, figsize=(25,4))
for idx in range(100):
    ax = axs[idx // 25, idx % 25]
    ax.imshow(obss[idx])
    ax.axis("off")
plt.suptitle("Evaluation Rollout Frames")
plt.savefig(os.path.join(folder, run_folder,'runs/eval_latent_planarcircle_frames.png'), bbox_inches='tight', pad_inches=0)
plt.close(fig)

env.close()
del env