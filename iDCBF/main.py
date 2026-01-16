from omegaconf import OmegaConf
import torch
import numpy as np
import os
import sys
import time
import joblib
import wandb
import hydra
from tqdm import tqdm
from accelerate import Accelerator
from wandb.integration.sb3 import WandbCallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_vit
from preprocessor import Preprocessor
from datasets.planarcircle_dset import PlanarCircleDataset
from datasets.planarcircle_dset import ACTION_MEAN, ACTION_STD, STATE_MEAN, STATE_STD, PROPRIO_MEAN, PROPRIO_STD
from env.planarcircle.planarcircle_env import PlanarCircleEnv
from env.planarcircle.latent_planarcircle_env import LatentPlanarCircleEnv

from models.ca_visual_world_model import VWorldModel
from models.behavioral_cloner import BehavioralCloner
from iDCBF.safety_filters import LatentIDBF

device = 'cuda'

folder = '/home/planiacs/gits/dino_wm/outputs'
# run_folder = '2025-12-23/13-32-59' # only A_to_B data
run_folder = '2026-01-08/14-17-51' # A_to_B + biased_brown + white
ckpt_folder = os.path.join(folder, run_folder)

real_env = PlanarCircleEnv()
data_preprocessor = Preprocessor(action_mean=ACTION_MEAN.to(device),
                                action_std=ACTION_STD.to(device),
                                state_mean=STATE_MEAN.to(device),
                                state_std=STATE_STD.to(device),
                                proprio_mean=PROPRIO_MEAN.to(device),
                                proprio_std=PROPRIO_STD.to(device),
                                transform=None)

@torch.no_grad()
def sample_contrastive_ood(
    z_safe_hist: torch.Tensor,      # [B, h, 196, 404]
    bc: BehavioralCloner,
    world_model: VWorldModel,
    num_candidates: int = 20,
    logprob_thresh=-8.0
):
    """
    Returns x_unsafe samples generated via low-probability actions
    """
    z_safe_hist = z_safe_hist[:,:world_model.local_hist,...]  # [B, h, p, l]

    B, h, p, l = z_safe_hist.shape

    # Sample candidate actions (global, not token-wise)
    acts = torch.zeros((B, num_candidates, real_env.action_space.shape[0]), 
                       device=z_safe_hist.device)
    # create fast action_space from which we can batch sample
    lb, ub = real_env.action_space.low, real_env.action_space.high
    acts = np.random.uniform(lb, ub, size=(B, num_candidates, world_model.cfg_dict.frameskip, real_env.action_space.shape[0]))
    acts = torch.from_numpy(acts).to(z_safe_hist.device).float()
    # normalize actions
    acts = data_preprocessor.normalize_actions(acts) # [B, num_candidates, frameskip, action_dim]
    # reshape to [B, num_candidates, frameskip * action_dim]
    acts = acts.view(B, num_candidates, -1)
    # encode actions
    u_cand = world_model.encode_act(acts) 

    # Evaluate BC likelihood
    logps = []
    for j in range(num_candidates):
        logp = bc.mdn_logprob(
            z_safe_hist, # [B, h, p, l]
            u_cand[:,j,:]       # [B, a]
        )
        logps.append(logp)
    logp = torch.stack(logps, dim=1)  # [B, num_candidates]
    
    # Mask low-likelihood actions
    mask = logp < logprob_thresh

    # Pick one low-probability action per batch element
    idx = mask.float().argmax(dim=1)
    u_ood = u_cand[:, idx, :]  # [B, a] after indexing

    # before applying, needs to be [b, 1, p, a]
    u_ood = u_ood[:,:1,...]

    # One-step latent rollout
    z_unsafe, dz_unsafe = world_model.predict(z_safe_hist, u_ood)

    return z_unsafe

def idcbf_loss(world_model:VWorldModel, 
               B:LatentIDBF, 
               z_safe: torch.Tensor, z_unsafe: torch.Tensor, u_now: torch.Tensor):
    # Eq (7) in: 
    # https://proceedings.mlr.press/v211/castaneda23a/castaneda23a.pdf
    w_safe = 1.0
    w_unsafe = 1.0
    w_ascent = 1.0
    eps_safe = 1e-3
    eps_unsafe = 1e-3
    eps_ascent = 1e-3

    loss = 0.0

    # safe samples
    B_safe = B(z_safe).mean(dim=-1).unsqueeze(-1)
    loss += w_safe * torch.max(torch.zeros_like(B_safe), eps_safe - B_safe).mean()

    # unsafe samples
    B_unsafe = B(z_unsafe).mean(dim=-1).unsqueeze(-1)
    loss += w_unsafe * torch.max(torch.zeros_like(B_unsafe), eps_unsafe + B_unsafe).mean()

    # CBF condition term
    gradB = torch.autograd.grad(
        B_safe.sum(), z_safe, create_graph=True
    )[0]  # [B, latent_dim]
    fz = world_model.predictor.get_fz(z_safe)
    gz = world_model.predictor.get_gz(z_safe)
    gzu = torch.einsum('bndu,bnu->bnd', gz, u_now)
    gradBfg = torch.einsum('bnd,bnd->b', gradB, fz + gzu).unsqueeze(-1)
    loss += w_ascent * torch.max(torch.zeros_like(B_safe), eps_ascent - (gradBfg + 0.1 * B_safe)).mean()
    return loss

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        wandb_dict = OmegaConf.to_container(cfg, resolve=True)
        self.wandb_run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            config=wandb_dict,
            id=None,
            resume="allow",
        )

        self.init_models()

        self.accelerator = Accelerator(log_with="wandb")

        self.datasets, traj_dsets = hydra.utils.call(
            self.world_model.cfg_dict.env.dataset,
            num_hist=self.world_model.cfg_dict.num_hist,
            num_pred=self.world_model.cfg_dict.num_pred,
            frameskip=self.world_model.cfg_dict.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.idcbf.gpu_batch_size,
                shuffle=False, 
                num_workers=16,#min(16,self.world_model.cfg_dict.env.num_workers),
                pin_memory=self.world_model.cfg_dict.env.pin_memory,
                persistent_workers=False,#self.world_model.cfg_dict.env.persistent_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }
        self.dataloaders['train'], self.dataloaders['valid'] = self.accelerator.prepare(
            self.dataloaders['train'], self.dataloaders['valid']
        )

        self.epoch = 0

    def init_models(self):
        self.world_model, wm_cfg = load_vit(ckpt_folder)
        self.world_model = self.world_model.to(device)
        self.world_model.eval()

        self.CBF = LatentIDBF(
            latent_dim=404,
            hidden_dim=self.cfg.idcbf.hidden_dim,
            num_layers=self.cfg.idcbf.num_layers,
        ).to(device)

        self.wandb_run.watch(self.CBF)

        self.optimizer = torch.optim.Adam(
            self.CBF.parameters(),
            lr=self.cfg.idcbf.lr,
            weight_decay=self.cfg.idcbf.weight_decay,
        )

    def run(self):
        num_epochs = self.cfg.idcbf.num_epochs
        for epoch in range(num_epochs):
            self.train()
            self.eval()
            if (epoch + 1) % self.cfg.idcbf.save_freq == 0:
                save_path = os.path.join(
                    ckpt_folder,
                    "idcbf",
                    f"idcbf_epoch{self.epoch}.pt",
                )
                self.save_model(save_path)
        # final save
        save_path = os.path.join(
            ckpt_folder,
            "idcbf",
            f"idcbf_final_epoch{self.epoch}.pt",
        )
        self.save_model(save_path)

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data

            # get latent histories
            obs = {k:v[:,:-1,...].to(device) for k,v in obs.items()}
            act = act.to(device)
            t0 = time.time()
            o, z, u = self.world_model.encode(
                obs, act
            )  # o: [B, h, 196, 404], z: [B, h, latent_dim], u: [B, h, action_emb_dim]
            # print("Encoding time:", time.time() - t0)

            # sample unsafe latents via contrastive OOD
            t0 = time.time()
            z_unsafe = sample_contrastive_ood(
                z_safe_hist=z,  # [B, h, 1, latent_dim]
                bc=self.world_model.behavioral_cloner,
                world_model=self.world_model,
                num_candidates=self.cfg.idcbf.num_candidates)
            # print("OOD sampling time:", time.time() - t0)

            # compute IDCBF loss
            t0 = time.time()
            loss = idcbf_loss(
                world_model=self.world_model,
                B=self.CBF,
                z_safe=z[:, -1, :],       # [B, latent_dim]
                z_unsafe=z_unsafe[:, 0, :], # [B, latent_dim]
                u_now=u[:, -1, :],         # [B, action_emb_dim]
            )
            # print("Loss computation time:", time.time() - t0)
            
            if i % 10 == 0:
                wandb.log({"train/loss": loss.item(), "epoch": self.epoch})

            # t0 = time.time()
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            # print("Optimization step time:", time.time() - t0)
        
        self.epoch += 1

    def eval(self):
        self.CBF.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for i, data in enumerate(
                tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Eval")
            ):
                obs, act, state = data

                # get latent histories
                obs = {k:v[:,:-1,...].to(device) for k,v in obs.items()}
                act = act.to(device)
                o, z, u = self.world_model.encode(
                    obs, act
                )  # o: [B, h, 196, 404], z: [B, h, latent_dim], u: [B, h, action_emb_dim]

                # sample unsafe latents via contrastive OOD
                z_unsafe = sample_contrastive_ood(
                    z_safe_hist=z,  # [B, h, 1, latent_dim]
                    bc=self.world_model.behavioral_cloner,
                    world_model=self.world_model,
                    num_candidates=self.cfg.idcbf.num_candidates)

                # compute IDCBF loss
                loss = idcbf_loss(
                    world_model=self.world_model,
                    B=self.CBF,
                    z_safe=z[:, -1, :],       # [B, latent_dim]
                    z_unsafe=z_unsafe[:, 0, :], # [B, latent_dim]
                    u_now=u[:, -1, :],         # [B, action_emb_dim]
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        wandb.log({"valid/loss": avg_loss, "epoch": self.epoch})
        print(f"Eval Loss: {avg_loss}")

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_dict = self.accelerator.get_state_dict(self.CBF)
        torch.save(state_dict, save_path)
        print(f"Saved CBF model to {save_path}")    
            
@hydra.main(config_path="", config_name="iDCBF.yaml")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
    