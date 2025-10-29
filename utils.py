import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from typing import Callable, Dict
import psutil

from models.ca_visual_world_model import VWorldModel
from models.ca_vit import ViTPredictor
from models.dino import DinoV3Encoder
from models.proprio import ProprioceptiveEmbedding, ProprioceptiveDecoding
from models.vqvae import VQVAE
from datasets.pusht_dset import PushTDataset

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Memory usage in MB

def get_available_ram():
    mem = psutil.virtual_memory()
    return mem.available / (1024 * 1024 * 1024)  # Available memory in MB

def dict_to_namespace(cfg_dict):
    args = argparse.Namespace()
    for key in cfg_dict:
        setattr(args, key, cfg_dict[key])
    return args

def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct

def slice_trajdict_with_t(data_dict, start_idx=0, end_idx=None, step=1):
    if end_idx is None:
        end_idx = max(arr.shape[1] for arr in data_dict.values())
    return {key: arr[:, start_idx:end_idx:step, ...] for key, arr in data_dict.items()}

def concat_trajdict(dcts):
    full_dct = {}
    for k in dcts[0].keys():
        if isinstance(dcts[0][k], np.ndarray):
            full_dct[k] = np.concatenate([dct[k] for dct in dcts], axis=1)
        elif isinstance(dcts[0][k], torch.Tensor):
            full_dct[k] = torch.cat([dct[k] for dct in dcts], dim=1)
        else:
            raise TypeError(f"Unsupported data type: {type(dcts[0][k])}")
    return full_dct

def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def sample_tensors(tensors, n, indices=None):
    if indices is None:
        b = tensors[0].shape[0]
        indices = torch.randperm(b)[:n]
    indices = torch.tensor(indices)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            tensors[i] = tensor[indices]
    return tensors


def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(cfg_dict[key])
    return cfg_dict

def reduce_dict(f: Callable, d: Dict):
    return {k: reduce_dict(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

def strip_targets_from_cfg(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # loop through all keys and remove the ones for which the value 
    # is a dict which contains the key '_target_'
    keys_to_remove = []
    for key, value in cfg_dict.items():
        if isinstance(value, dict) and '_target_' in value:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del cfg_dict[key]
    return cfg_dict

def load_vit(checkpoint_folder:str):
    model_ckpt = os.path.join(checkpoint_folder, 'checkpoints', 'model_latest.pth')
    with open(model_ckpt, "rb") as f:
        payload = torch.load(f, map_location='cpu', weights_only=False)
    
    cfg = OmegaConf.load(os.path.join(checkpoint_folder,'hydra.yaml'))

    #! create the DinoV3 encoder
    encoder = DinoV3Encoder(name=cfg.encoder.name, 
                            feature_key=cfg.encoder.feature_key)
    encoder.eval()

    #! create proprio encoder 
    proprio_encoder = ProprioceptiveEmbedding(num_frames=cfg.proprio_encoder.num_frames,
                                            tubelet_size=cfg.proprio_encoder.tubelet_size,
                                            in_chans=4,
                                            emb_dim=cfg.proprio_emb_dim,
                                            use_3d_pos=cfg.proprio_encoder.use_3d_pos)
    proprio_encoder.load_state_dict(payload['proprio_encoder'].state_dict())
    proprio_encoder.eval()

    #! create action encoder 
    action_encoder = ProprioceptiveEmbedding(num_frames=cfg.action_encoder.num_frames,
                                            tubelet_size=cfg.action_encoder.tubelet_size,
                                            in_chans=10,
                                            emb_dim=cfg.action_emb_dim,
                                            use_3d_pos=cfg.action_encoder.use_3d_pos)
    action_encoder.load_state_dict(payload['action_encoder'].state_dict())
    action_encoder.eval()

    #! create action decoder
    action_decoder = ProprioceptiveDecoding(num_frames=cfg.action_decoder.num_frames,
                                            tubelet_size=cfg.action_decoder.tubelet_size,
                                            out_chans=10,
                                            emb_dim=cfg.action_emb_dim)
    action_decoder.load_state_dict(payload['action_decoder'].state_dict())
    action_decoder.eval()

    #! create decoder
    decoder = VQVAE(channel=cfg.decoder.channel,
                    n_res_block=cfg.decoder.n_res_block,
                    n_res_channel=cfg.decoder.n_res_channel,
                    n_embed=cfg.decoder.n_embed,
                    emb_dim=384,
                    quantize=cfg.decoder.quantize)
    decoder.load_state_dict(payload['decoder'].state_dict())
    decoder.eval()

    #! create world model
    predictor = ViTPredictor(num_patches=196,
                            num_frames=cfg.num_hist,
                            dim=404,
                            action_dim=cfg.action_emb_dim,
                            depth=cfg.predictor.depth,
                            heads=cfg.predictor.heads,
                            mlp_dim=cfg.predictor.mlp_dim,
                            pool=cfg.predictor.pool,
                            dropout=cfg.predictor.dropout,
                            emb_dropout=cfg.predictor.emb_dropout)
    predictor.load_state_dict(payload['predictor'].state_dict())
    predictor.eval()

    # Finally create the world model
    world_model = VWorldModel(image_size=cfg.img_size,
                            num_hist=cfg.num_hist,
                            num_pred=cfg.num_pred,
                            encoder=encoder,
                            proprio_encoder=proprio_encoder,
                            action_encoder=action_encoder,
                            action_decoder=action_decoder,
                            decoder=decoder,
                            cfg_dict=cfg,
                            action_dim=cfg.action_emb_dim,
                            proprio_dim=cfg.proprio_emb_dim,
                            num_action_repeat=cfg.num_action_repeat,
                            num_proprio_repeat=cfg.num_proprio_repeat,
                            predictor=predictor)
    world_model.eval()
    
    return world_model, cfg
