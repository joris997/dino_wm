import os, sys
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
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

dataset = PushTDataset(n_rollout=50,
                       transform=None,
                       data_path="datasets/data/pusht_noise/val",
                       normalize_action=True,
                       with_velocity=True)

data_preprocessor = Preprocessor(action_mean=dataset.action_mean,
                                 action_std=dataset.action_std,
                                 state_mean=dataset.state_mean,
                                 state_std=dataset.state_std,
                                 proprio_mean=dataset.proprio_mean,
                                 proprio_std=dataset.proprio_std,
                                 transform=dataset.transform)

obs, act, state, _ = dataset.get_frames(0, range(100))

env = PushTEnv(reset_to_state=state[0].numpy())
observation, info = env.reset()


# some initial plotting
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for i in range(100):
    exec_act = data_preprocessor.denormalize_actions(act[i]).numpy()
    observation, reward, terminated, info = env.step(exec_act)
    image = env.render('rgb_array')

    axs[0].imshow(obs['visual'][i].permute(1, 2, 0).numpy())
    axs[0].axis("off")
    axs[1].imshow(image)
    axs[1].axis("off")
    # show now
    plt.pause(0.01)

# pass through the world model and the real model simultaneously and check results
