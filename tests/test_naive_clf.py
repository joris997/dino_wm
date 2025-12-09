# get target image: T in the goal area

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
run = '2025-12-04/15-39-40'
ckpt_folder = os.path.join(folder, run)

# world model
world_model, cfg = load_vit(ckpt_folder)
world_model.to('cuda')
cfg.debug = False

# dataset
dataset = PushTDataset(n_rollout=60,
                       transform=default_transform(cfg.img_size),
                       data_path="datasets/data/pusht_noise/val",
                       normalize_action=cfg.env.dataset.normalize_action,
                       with_velocity=cfg.env.dataset.with_velocity)
# get a range of data
# obs: dict, ['visual']: [100, C, H, W], ['proprio']: [100, P]
# act: [100, A]
obs, act, state, _ = dataset.get_frames(0, range(100))

# preprocessor to denormalize actions/states/proprios
data_preprocessor = Preprocessor(action_mean=dataset.action_mean,
                                 action_std=dataset.action_std,
                                 state_mean=dataset.state_mean,
                                 state_std=dataset.state_std,
                                 proprio_mean=dataset.proprio_mean,
                                 proprio_std=dataset.proprio_std,
                                 transform=dataset.transform)

# mujoco environment
env = PushTEnv(reset_to_state=state[0].numpy(),with_velocity=True)
observation, info = env.reset()

# We only get observations every 'frameskip' frames, so subsample the obs
obs_skip = {
    'visual': obs['visual'][::cfg.frameskip,...].unsqueeze(0),
    'proprio': obs['proprio'][::cfg.frameskip,...].unsqueeze(0)
}
obsGoal = {
    'visual': repeat(obs_skip['visual'][:,-1:,...], "b 1 ... -> b repeats ...", repeats=cfg.num_hist-1).to('cuda'),
    'proprio': repeat(obs_skip['proprio'][:,-1:,...], "b 1 ... -> b repeats ...", repeats=cfg.num_hist-1).to('cuda')
}
actGoal = torch.zeros((1, cfg.num_hist, cfg.frameskip*act.shape[-1])).to('cuda')
_, zGoal, uGoal = world_model.encode(obsGoal, actGoal)
uGoal = uGoal[:,:,-1,:].flatten().detach().cpu().numpy()

print(f"Goal latent shape: {zGoal.shape}")  # (1, num_hist, 196, 404)


obs0 = {'visual': obs_skip['visual'][:,:cfg.num_hist-1,...].to('cuda'),
        'proprio': obs_skip['proprio'][:,:cfg.num_hist-1,...].to('cuda')}
act0_raw = act[0 : (cfg.num_hist-1)*cfg.frameskip,:]
act0 = act0_raw.reshape(1, cfg.num_hist-1, 2*cfg.frameskip).to('cuda')
act0 = torch.cat([act0,
                  torch.zeros_like(act0[:,-1:,...])], dim=1)

# # plot current state and target state
# plt.subplot(1,2,1)
# plt.imshow(obs0['visual'][0,-1,...].permute(1,2,0).cpu().numpy())
# plt.title('Current State')
# plt.subplot(1,2,2)
# plt.imshow(obsGoal['visual'][0,-1,...].permute(1,2,0).cpu().numpy())
# plt.title('Target State')
# plt.show()

# now we define a Lyapunov function as the L2 distance in latent space to the goal latent
V = lambda z: torch.mean((z - zGoal)**2, dim=(1,2,3))  # input z: (B, num_hist, H, W)
dVdz = lambda z: 2 * (z.flatten() - zGoal.flatten())

# define a CLF-QP controller that computes the latent space actions 
# min_u ||u||^2
# s.t. dV/dz * f(z) + dV/dz * g(z) * u <= -c * V(z)
import cvxpy as cp
u = cp.Variable((cfg.frameskip * act.shape[-1],))  # action_dim = 2

# blind: bool indicating whether to use the real observation or the predicted observation
blind = False

fig, axs = plt.subplots(2, cfg.frameskip +1, figsize=(15, 5))
V_vals = []
Vzpred_vals = []
while True:
    for ax in axs.flatten():
        ax.cla()   
    # get current latent state
    _, z, _ = world_model.encode(obs0, act0)
    fz, gz = world_model.get_fz_gz(obs0, act0)
    print(f"fz shape: {fz.shape}, gz shape: {gz.shape}")

    # get the Lyapunov function values and the dynamics
    V_val = V(z).detach().cpu().numpy()
    dVdz_val = dVdz(z).detach().cpu().numpy()
    fz_val = fz.detach().cpu().numpy()
    gz_val = gz.detach().cpu().numpy()
    # flatten to enable multiplication with cvxpy var
    dVdz_val = dVdz_val.flatten()
    fz_val = fz_val.flatten()
    gz_val = gz_val.reshape(-1,gz_val.shape[-1])

    # cost and constraints
    # cost = cp.quad_form(u.flatten(), np.eye(cfg.frameskip * act.shape[-1]))
    cost = cp.quad_form(u.flatten() - uGoal, np.eye(cfg.frameskip * act.shape[-1]))   
    constraints = []
    constraints.append(
        dVdz_val @ fz_val + dVdz_val @ gz_val @ u <= -0.9 * V_val
    )
    # solve QP
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()
    print(f"Optimal latent action u*: {u.value}")

    #! compare solution to see if it all makes sense
    # 1. fz_val + gz_val @ u.value =  dz
    lhs = fz_val + gz_val @ u.value
    act0[:, -1, ...] = torch.tensor(u.value.reshape(1, -1), dtype=torch.float32, device='cuda')
    with torch.no_grad():
        u_now = repeat(act0[:, -1:, ...].unsqueeze(0), "b t 1 d -> b t f d", f=cfg.frameskip)
        z_k1, dz, z_k = world_model.get_zk1_dz(obs0, act0, u_now=act0[:, -1:, ...])
    rhs =  dz.flatten().cpu().numpy()
    assert all(np.isclose(lhs, rhs, rtol=1e-3)), f"lhs: {lhs}, rhs: {rhs}"


    # perform the action in the real environment
    # we first decode the action from latent space to task space
    with torch.no_grad():
        acti = world_model.decode_act(act0[:, -1:, :])
    acti = acti.reshape(cfg.frameskip, -1).cpu()
    # and we denormalize the action?
    act_denorm = np.array([data_preprocessor.denormalize_actions(a).numpy() for a in acti])
    visuals, proprios = [], []
    # take the cfg.frameskip steps in the real environment
    for i, a in enumerate(act_denorm):
        observation, _, _, _ = env.step(a)
        visuals.append(observation['visual'])
        proprios.append(observation['proprio'])
        axs[0,i].imshow(observation['visual'])
        axs[0,i].axis("off")
        axs[0,i].set_title(f'Step {i}')

    # perform the action in the latent environment
    with torch.no_grad():
        obs_pred, z_pred, dz_pred, obs_now = world_model.take_step(obs0, act0)
    obs_pred_vis = obs_now['visual'].cpu().detach()
    axs[1,-2].imshow(np.clip(obs_pred_vis[0,-1,...].permute(1,2,0).cpu().numpy(), 0, 1))
    axs[1,-2].axis("off")
    axs[1,-2].set_title('Predicted')
    Vzpred_val = V(z_pred).detach().cpu().numpy()
    Vzpred_vals.append(Vzpred_val)
    
    V_vals.append(V_val)
    axs[1,-1].plot(range(len(V_vals)), V_vals,'b',label='V(z)')
    axs[1,-1].plot(range(1,len(Vzpred_vals)+1), Vzpred_vals,'r',label='V(z_pred)')
    axs[1,-1].legend()
    axs[1,-1].set_title('Lyapunov Function V(z)')
    plt.suptitle('Real Environment Steps')
    plt.draw()
    # plt.waitforbuttonpress()
    plt.pause(0.01)


    # update obs0 and act0, depending on whether we are blind (stay in latent model) or not (use real obs)
    if blind:
        obs0 = {
            'visual': torch.cat([obs0['visual'][:,1:,...], obs_pred['visual'][:,-1:,...]], dim=1),
            'proprio': torch.cat([obs0['proprio'][:,1:,...], obs_pred['proprio'][:,-1:,...]], dim=1)
        }
    else:
        # obtain the observation and the proprioception
        obsi = {
            'visual': torch.tensor(np.array(visuals), device='cuda').unsqueeze(0)/255.0,
            'proprio': torch.tensor(np.array(proprios), device='cuda').unsqueeze(0)
        }
        # convert 'visual' H W 3 to 3 H W
        obsi['visual'] = obsi['visual'].permute(0,1,4,2,3)
        obs0 = {
            'visual': torch.cat([obs0['visual'][:,1:,...], obsi['visual'][:,-1:,...]], dim=1),
            'proprio': torch.cat([obs0['proprio'][:,1:,...], obsi['proprio'][:,-1:,...]], dim=1)
        }

    act0 = torch.cat([act0[:,1:,...],
                      torch.tensor(u.value.reshape(1,1,-1), dtype=torch.float32, device='cuda')], dim=1)
