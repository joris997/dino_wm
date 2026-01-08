import gymnasium as gym
from gymnasium import spaces

from typing import Optional
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import os
import sys
import torch
import matplotlib.pyplot as plt

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
from models.ca_visual_world_model import VWorldModel


class LatentPlanarCircleEnv(gym.Env):
    # this is a gym env from a mujoco xml file
    metadata = {"render_modes": ["human","rgb_array"],
                "video.frames_per_second": 10}

    def __init__(self, 
                 real_env:gym.Env,
                 world_model:VWorldModel,
                 preprocessor:Preprocessor,
                 target_state:np.ndarray,

                 weights:dict={'goal_distance':1.0},
                 reward_info:dict=None,

                 render_mode=None, 
                 camera_id:int=0,
                 camera_width:int=64,
                 camera_height:int=64,
                 reset_to_state:Optional[np.ndarray]=None,
                 
                 average_pool:bool=False,
                 step_in_real_env:bool=False,

                 log_dir:str="logs",
                 **kwargs):
        
        self.real_env = real_env
        self.world_model = world_model
        self.preprocessor = preprocessor
        self.target_state = target_state
        print(f"target_state.shape: {self.target_state.shape}")

        self.epsilon = 0.1
        self.weights = weights
        self.reward_info = reward_info

        self.render_mode = render_mode
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.reset_to_state = reset_to_state

        self.average_pool = average_pool
        self.step_in_real_env = step_in_real_env

        # total latent state by multiplying the sizes
        if self.average_pool:
            self.observation_space = spaces.Box(low=-1e4, high=1e4, shape=(3,404), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-1e4, high=1e4, shape=(3,196,384), dtype=np.int32)
        # print(f"n_u for setting action_space: {np.repeat(self.real_env.action_space.low, self.world_model.cfg_dict.frameskip)}")
        self.action_space = spaces.Box(low=np.repeat(self.real_env.action_space.low, self.world_model.cfg_dict.frameskip), 
                                       high=np.repeat(self.real_env.action_space.high, self.world_model.cfg_dict.frameskip),
                                       dtype=np.float32)

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

    def step(self, action:np.ndarray)->np.ndarray | np.float64 | bool | bool | dict:
        """ 
        Take a step in the environment, either by taking the action in the real environment and updating z
        based on the updated image of the real environment, or by taking the action in the latent world model
        and using the predicted latent state as the new z.
        In:
            action: np.ndarray of shape (local_hist*2,)
            step_in_real_env: bool, whether to take the action in the real environment and update z or just stay in latent space
        Out:
            obs: np.ndarray of shape (LSIZE,)
            reward: float
            done: bool
            info: dict
        """
        # reshape (local_hist*2) to (local_hist, 2)
        action = action.reshape(self.world_model.cfg_dict.frameskip, 2)
        # take the action in the real environments
        for a in action:
            self.real_env.step(a)

        action = torch.tensor(action).float().to('cuda')
        action = self.preprocessor.normalize_actions(action).reshape(1,-1).unsqueeze(0)
        
        # take the action in the latent environment
        if self.step_in_real_env:
            obs = torch.from_numpy(self.real_env.render().copy()).float().permute(2,0,1)/255.0
            obs = obs.unsqueeze(0).unsqueeze(0).to('cuda') # add batch dim and hist size
            proprio = torch.from_numpy(self.real_env._get_obs()).float()
            proprio = proprio.unsqueeze(0).unsqueeze(0).to('cuda') # add batch dim 
            proprio = self.preprocessor.normalize_proprios(proprio)
            obs_dict = {'visual': obs, 'proprio': proprio}
            # transform obs to z and encode proprio and actions
            with torch.no_grad():
                o_dict = self.world_model.encode_obs(obs_dict)
                u = self.world_model.encode_act(action)
            self.z = torch.cat((
                self.z[:,1:,...],
                torch.cat([o_dict['visual'],
                           o_dict['proprio'],
                           u
                ], dim=-1)
            ), dim=1)
            self.z = self.z.detach()
            self.z_avg = self.z.mean(dim=2)

        else:
            with torch.no_grad():
                u_now = self.world_model.encode_act(action)
                z_pred, dz_pred = self.world_model.predict(self.z[:, -self.world_model.local_hist:],
                                                           u_now)
            self.z = torch.cat((self.z[:,1:,...], 
                                z_pred[:,-1:,...]), dim=1).detach()
            self.z_avg = self.z.mean(dim=2)

        # create a flat z as per return requirements of gym.Env    
        self.flat_z = self._flatten_latent(self.z)

        if self.render_mode == "human":
            self.render()

        # reward is negative 2-norm distance to target state
        # print(f"self.z.shape: {self.z.shape}, self.target_state.shape: {self.target_state.shape}")
        # remove batch dimension and only take latest time step, then remove action history
        num_embedding = 384
        if self.average_pool:
            z_reward = self.z_avg[0,-1,:num_embedding]
            target_reward = torch.mean(self.target_state, dim=2)[0,-1,:num_embedding]
            z_std = torch.mean(self.world_model.info_dict['latent_std'], dim=0)[:num_embedding]
        else:
            z_reward = self.z[0,-1,:,:num_embedding]
            target_reward = self.target_state[0,-1,:,:num_embedding]
            z_std = self.world_model.info_dict['latent_std'][:,:num_embedding]

        if self.reward_info is not None:
            term = (z_reward - target_reward) / (z_std + 1e-8)
            norm = torch.norm(term).cpu().numpy()**2
        else:
            term = z_reward - target_reward
            norm = torch.norm(term).cpu().numpy()**2
        loss = norm
        
        # # try cosine similarity
        # # patch-wise cosine
        # assert z_reward.dim() == 2
        # assert target_reward.dim() == 2
        # reward = torch.nn.functional.cosine_similarity(z_reward, target_reward, dim=1).mean().cpu().numpy()
        # loss = 1 - reward

        reward = -self.weights['goal_distance'] * loss
        # we're done if we're within epsilon of the target state
        done = reward > -self.epsilon

        obs = self._get_obs()
        return obs, reward, done, False, {}
    
    def reset_model(self)->np.ndarray:
        if self.reset_to_state is not None:
            self.z = torch.tensor(self.reset_to_state).float().unsqueeze(0).to('cuda')
            return self._get_obs()
        
        proprio, info = self.real_env.reset()
        proprio = torch.from_numpy(proprio.reshape(1,-1)).float().to('cuda')
        # print(f"proprio.shape: {proprio.shape}")
        observation_ = self.real_env.render()
        observation = torch.from_numpy(observation_.copy()).float().permute(2,0,1).unsqueeze(0).to('cuda')/255.0 # add hist size
        obs = {'visual': torch.repeat_interleave(observation.unsqueeze(0), self.world_model.local_hist, dim=1).to('cuda'), # add batch size
               'proprio': torch.repeat_interleave(proprio.unsqueeze(0), self.world_model.local_hist, dim=1).to('cuda')} # add batch size
        
        random_action = self.real_env.action_space.sample()
        act = torch.zeros((1, self.world_model.num_hist, random_action.shape[-1]*self.world_model.cfg_dict.frameskip), device='cuda')
        # print(f"obs['visual'].shape: {obs['visual'].shape}, obs['proprio'].shape: {obs['proprio'].shape}, act.shape: {act.shape}")
        
        o,z,u = self.world_model.encode(obs, act)
        self.z = z
        self.z_avg = self.z.mean(dim=2) # average pool over latent dimension

        # fig, ax = plt.subplots(figsize=(4,4))
        # latent_obs = self.get_latent_obs(self.z)
        # ax.imshow(latent_obs)
        # # ax.imshow(observation_)
        # # print(f"max(observation_): {np.max(observation_)}, min(observation_): {np.min(observation_)}")
        # # ax.imshow(obs['visual'][0,-1].permute(1,2,0).cpu().numpy()/255.0, alpha=0.5)
        # ax.axis('off')
        # fig.savefig("temp_latent_reset.png", bbox_inches='tight', pad_inches=0)
        # plt.close(fig)
        if self.average_pool:
            return self._flatten_latent(self.z_avg)
        else:
            return self._flatten_latent(self.z)
    
    def reset(self,
              seed:Optional[int]=None, 
              options:Optional[dict]=None)->np.ndarray:
        self.flat_z = self.reset_model()
        return self._get_obs(), {}

    def get_latent_obs(self, latent):
        """ Get latent image """
        # print(f"latent.shape in get_latent_obs: {latent.shape}")
        with torch.no_grad():
            # decode only the visual part
            obs, _ = self.world_model.decode_obs(self._get_visual_from_latent(latent))
            latent_obs = obs['visual'][0,-1].permute(1,2,0).cpu().numpy()
            latent_obs = np.clip((latent_obs+1)/2, 0, 1)
        
        # VecVideoRecorder expects (0,255) range
        return latent_obs*255.0

    def render(self)->np.ndarray:
        y = self.get_latent_obs(self.z).astype(np.uint8)

        if self.render_mode == "human":
            plt.imshow(y)
            plt.axis('off')
            plt.pause(0.1)
            return y
        elif self.render_mode == "rgb_array":
            return y
    
    def _get_obs(self)->np.ndarray:
        # just return z but remove batch dimension
        if self.average_pool:
            return self.z_avg[0,...].detach().cpu().numpy()
        else:
            return self.z[0,:,:,:384].detach().cpu().numpy()
        # return self._flatten_latent(self.z).detach().cpu().numpy()#np.squeeze(self.z.detach().cpu().numpy())

    def _flatten_latent(self, latent:torch.Tensor)->torch.Tensor:
        """ Flatten latent tensor """
        return latent.flatten()
    
    def _get_visual_from_latent(self, latent:torch.Tensor)->torch.Tensor:
        """ Get visual part from latent tensor """
        return latent[:, :, :, :384]