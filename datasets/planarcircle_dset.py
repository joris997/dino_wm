import torch
import decord
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# count the number of files in the /datasets/data/planarcircle/A_to_B/thread_0/ directory
# data_path = Path("datasets/data/planarcircle/A_to_B/thread_0/")
# num_files = len(list(data_path.glob("rollout_*.npz")))
# # get the length of a rollout
# with open(data_path / "rollout_0.npz", "rb") as f:
#     data = np.load(f)
#     rollout_length = data['observations'].shape[0]
#     # allocate tensors to hold all data
#     actions = torch.zeros((num_files*rollout_length, data['actions'].shape[1]))
#     proprio = torch.zeros((num_files*rollout_length, data['states'].shape[1]))  # assuming last 4 are image pixels
#     states  = torch.zeros((num_files*rollout_length, data['states'].shape[1]))
# # loop through the rollouts and fill in the tensors
# for i in range(num_files):
#     with open(data_path / f"rollout_{i}.npz", "rb") as f:
#         data = np.load(f)
#         start_idx = i * rollout_length
#         end_idx = start_idx + rollout_length
#         actions[start_idx:end_idx,:] = torch.from_numpy(data['actions']).float()
#         states[start_idx:end_idx,:] = torch.from_numpy(data['states']).float()
#         proprio[start_idx:end_idx,:] = torch.from_numpy(data['states']).float()  # assuming last 4 are image pixels
# # compute mean and std of actions, states, and proprio
# ACTION_MEAN = actions.mean(dim=0)
# ACTION_STD = actions.std(dim=0)
# STATE_MEAN = states.mean(dim=0)
# STATE_STD = states.std(dim=0)
# PROPRIO_MEAN = proprio.mean(dim=0)
# PROPRIO_STD = proprio.std(dim=0)
# print("Computed dataset statistics.")
# print(f"ACTION_MEAN = torch.{ACTION_MEAN}")
# print(f"ACTION_STD = torch.{ACTION_STD}")
# print(f"STATE_MEAN = torch.{STATE_MEAN}")
# print(f"STATE_STD = torch.{STATE_STD}")
# print(f"PROPRIO_MEAN = torch.{PROPRIO_MEAN}")
# print(f"PROPRIO_STD = torch.{PROPRIO_STD}")
ACTION_MEAN = torch.tensor([-0.0960, -0.0831])
ACTION_STD = torch.tensor([6.2808, 6.2708])
STATE_MEAN = torch.tensor([ 0.5383,  0.5404, -0.0007, -0.0006])
STATE_STD = torch.tensor([1.6620, 1.6459, 0.5362, 0.5340])
PROPRIO_MEAN = torch.tensor([ 0.5383,  0.5404, -0.0007, -0.0006])
PROPRIO_STD = torch.tensor([1.6620, 1.6459, 0.5362, 0.5340])

class PlanarCircleDataset(TrajDataset):
    def __init__(
            self,
            n_rollout: Optional[int] = None,
            transform: Optional[Callable] = None,
            data_path: str = "data/planarcircle/A_to_B",
            normalize_action: bool = True,
            relative=True,
            action_scale=10.0,
            with_velocity: bool = False, # agent's velocity
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.action_scale = action_scale
        self.with_velocity = with_velocity

        # get seq_length which is the number of files in each folder in data_path
        self.seq_lengths = []
        # create a map that maps from rollout index to (thread_folder, file_name)
        self.rollout_map = []
        for thread_folder in self.data_path.glob("thread_*"):
            for file in thread_folder.glob("rollout_*.npz"):
                with open(file, "rb") as f:
                    data = np.load(f)
                    self.seq_lengths.append(data['actions'].shape[0])
                self.rollout_map.append((thread_folder, file.name))
        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.seq_lengths)
        self.seq_lengths = self.seq_lengths[:n]

        # open one file to get the dimensions, this is not necessarily thread_0/rollout_0.npz
        sample_file = None
        for thread_folder in self.data_path.glob("thread_*"):
            sample_file = thread_folder / "rollout_0.npz"
            if sample_file.exists():
                break
        if sample_file is None:
            raise FileNotFoundError("No rollout files found in data_path")
        with open(sample_file, "rb") as f:
            data = np.load(f)
            self.action_dim = data['actions'].shape[1]
            self.state_dim = data['states'].shape[1]
            self.proprio_dim = self.state_dim  # assuming all state is proprio

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[:self.state_dim]
            self.state_std = STATE_STD[:self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
            self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
    
    def get_seq_length(self, idx: int) -> int:
        return self.seq_lengths[idx]

    def get_all_actions(self) -> torch.Tensor:
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            thread_folder, file_name = self.rollout_map[i]
            with open(thread_folder / file_name, "rb") as f:
                data = np.load(f)
                actions = torch.from_numpy(data['actions'][:T,:]).float()
                # normalize actions
                actions = (actions - self.action_mean) / self.action_std
                result.append(actions)
        return torch.cat(result, dim=0)
    
    def get_frames(self, idx, frames):
        thread_folder, file_name = self.rollout_map[idx]
        # get the actions, states, proprio from the file
        with open(thread_folder / file_name, "rb") as f:
            data = np.load(f)
            actions = torch.from_numpy(data['actions']).float()
            states = torch.from_numpy(data['states']).float()
            proprio = states  # assuming all state is proprio
            # optionally drop velocity from states and proprio
            if not self.with_velocity:
                states = states[:,:2]
                proprio = proprio[:,:2]
            # normalize actions, states, and proprio
            actions = (actions - self.action_mean) / self.action_std
            states = (states - self.state_mean) / self.state_std
            proprio = (proprio - self.proprio_mean) / self.proprio_std

        # get the video frames
        # replace .npz with .mp4 to get video file
        video_file = file_name.replace(".npz", ".mp4")
        video_path = thread_folder / video_file
        vr = VideoReader(str(video_path), num_threads=1)
        image = vr.get_batch(frames)
        image = image / 255.0
        image = rearrange(image, "b h w c -> b c h w")
        if self.transform:
            image = self.transform(image)
        # pack it up
        obs = {"visual": image, "proprio": proprio[frames]}
        act = actions[frames]
        state = states[frames]
        return obs, act, state, {'shape': 'circle'}
        
    def __getitem__(self, index):
        return self.get_frames(index, range(self.get_seq_length(index)))
    
    def __len__(self):
        return len(self.seq_lengths)
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        
def load_planarcircle_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/planarcircle/A_to_B",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=False
):
    train_dset = PlanarCircleDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )
    val_dset = PlanarCircleDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset