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
from RL.policies import LatentCNN

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise

import matplotlib.pyplot as plt
import torch

# get date in yyyy-mm-dd format
from datetime import datetime

device = 'cuda'

folder = '/home/planiacs/gits/dino_wm/outputs'
# run_folder = '2025-12-23/13-32-59' # only A_to_B data
run_folder = '2026-01-02/13-16-08' # A_to_B + biased_brown + white
ckpt_folder = os.path.join(folder, run_folder)


def make_env(step_in_real_env:bool=False)->gym.Env:
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
    # # plot the target state
    # fig, ax = plt.subplots(1,world_model.local_hist, figsize=(15,5))
    # for i in range(world_model.local_hist):
    #     latent_obs = obss['visual'][0,i].permute(1,2,0).cpu().numpy()*255.0
    #     ax[i].imshow(latent_obs.astype(np.uint8))
    #     ax[i].axis('off')
    # fig.savefig(f"logs/{current_date}/{current_time}/target_latent_state.png", 
    #             bbox_inches='tight', pad_inches=0)
    # plt.close(fig)

    # create the latent environment
    env = LatentPlanarCircleEnv(real_env=real_env,
                                world_model=world_model,
                                preprocessor=data_preprocessor,
                                target_state=z,

                                render_mode='rgb_array',
                                camera_id=0,
                                camera_width=64,
                                camera_height=64,
                                
                                step_in_real_env=step_in_real_env)
    env = Monitor(gym.wrappers.TimeLimit(env, max_episode_steps=25))
    return env
    
if __name__ == "__main__":
    current_date = datetime.now().strftime("%Y-%m-%d")
    # get time in hh-mm-ss format
    current_time = datetime.now().strftime("%H-%M-%S")
    # create the ckpt folder path
    os.makedirs(f"logs/{current_date}/{current_time}/videos", exist_ok=True)

    wandb.login()
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 200_000,
        "env_name": "LatentPlanarCircle-v0",
    }
    # wandb.tensorboard.patch(root_logdir="./outputs/")
    run = wandb.init(
        project="latentRL",    # Specify your project
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False
    )

    #! Create the environment
    # create 'a' target state
    real_env = PlanarCircleEnv(render_mode='rgb_array')
    obs = real_env.reset_model()
    proprio = real_env._get_obs()
    # Nenvs = 1
    # env_fns = [lambda i=i: make_env(i) for i in range(Nenvs)]
    # env = SubprocVecEnv(env_fns, start_method='fork')
    # env = make_env(0)
    env = DummyVecEnv([lambda: make_env(step_in_real_env=False)])

    #! RL in the environment
    n_u = env.action_space.shape[-1]
    u_noise = NormalActionNoise(mean=env.envs[0].env.env.world_model.cfg_dict.frameskip*np.zeros(n_u), 
                                sigma=env.envs[0].env.env.world_model.cfg_dict.frameskip*0.1*np.ones(n_u))

    # Add additional wrappers
    env = VecVideoRecorder(env,
                        f"logs/{current_date}/{current_time}/videos",
                        record_video_trigger=lambda x: x % 10000 == 0,
                        video_length=200)

    # policy_kwargs = dict(net_arch=dict(pi=[64,64], qf=[64,64]))
    # policy_kwargs = dict(net_arch=dict(pi=[128,128,128], qf=[128,128,128]))
    # policy_kwargs = dict(net_arch=dict(pi=[1024,1024,512], 
    #                                    qf=[1024,1024,512]))
    policy_kwargs = dict(
        features_extractor_class=LatentCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    policy = "PPO"
    if policy == "TD3":
        model = TD3("MlpPolicy", env, action_noise=u_noise, verbose=0,
                    gamma=0.95, buffer_size=int(1e4),#learning_rate=1e-4,
                    policy_kwargs=policy_kwargs, device='cuda', 
                    tensorboard_log=f"logs/{current_date}/{current_time}/runs")
    elif policy == "PPO":
        model = PPO("CnnPolicy", env, verbose=0,
                    gamma=0.95, #learning_rate=1e-3,
                    policy_kwargs=policy_kwargs,device='cuda',
                    tensorboard_log=f"logs/{current_date}/{current_time}/runs") 
    elif policy == "SAC":
        model = SAC("MlpPolicy", env, action_noise=u_noise, verbose=0,
                    gamma=0.95, buffer_size=int(1e4),#learning_rate=1e-3,
                    policy_kwargs=policy_kwargs, device='cuda',
                    tensorboard_log=f"logs/{current_date}/{current_time}/runs")


    print(f"Created model with policy: {model.policy}")
    model.learn(total_timesteps=config['total_timesteps'],progress_bar=True,
                callback=WandbCallback(
                    gradient_save_freq=10000,
                    model_save_path=f"logs/{current_date}/{current_time}/models",
                    verbose=2)
    )
    model.save(f"logs/{current_date}/{current_time}/models/{policy}_latent_dynamics")
    wandb.finish()
