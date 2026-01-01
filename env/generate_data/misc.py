""" Various auxiliary utilities """
import math
from os.path import join, exists

# add relative path to the project
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
import numpy as np
# from models import MDRNNCell, VAE, Controller
# import gymnasium.envs.box2d
# from env.planarcircle.planarcircle_env import PlanarCircleEnv
# import gym.envs.box2d

# Hardcoded for now
# ASIZE: action size
# LSIZE: latent size
# RSIZE: hidden size
# RED_SIZE: reduced size of the image
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    2, 32, 256, 64, 64

# DATASETS: list of datasets to use in all training and evalss
# DATASETS = ['datasets/planarcircle_short']
DATASETS = ['datasets/planarcircle_biased_brown']
DATASETS += ['datasets/data/planarcircle/A_to_B'] 
# DATASETS += ['datasets/planarcircle_none'] 

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def sample_brown_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    sample = action_space.sample()
    actions = [np.zeros_like(sample)]*seq_len
    actions[0] = sample
    for i in range(1,seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions[i] = np.clip(actions[i-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high)
    return actions

def sample_biased_brown_policy(action_space, dt, state, action):
    """ Sample a continuous policy and bias it to move towards (1,1)
    
    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1) with a bias towards (1,1)

    :args action_space: gym action space
    :args dt: temporal discretization
    :args state: current state of the environment
    :args action: current action

    :returns: next action biased towards (1,1)
    """
    sample = np.zeros_like(action)
    x, y = state[0], state[1]
    bias = np.array([1.0 - x, 1.0 - y])  # bias towards (1,1)
    daction_dt = 2*np.random.randn(*sample.shape) + bias/8
    next_action = action + math.sqrt(dt) * daction_dt
    next_action = np.clip(next_action, action_space.low, action_space.high)
    return next_action

def sample_biased_white_policy(action_space, dt, state, action):
    """ Sample a continuous policy and bias it to move towards (1,1)
    
    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a white noise process with a bias towards (1,1)

    :args action_space: gym action space
    :args dt: temporal discretization
    :args state: current state of the environment
    :args action: current action

    :returns: next action biased towards (1,1)
    """
    sample = np.zeros_like(action)
    x, y = state[0], state[1]
    bias = np.array([1.0 - x, 1.0 - y])  # bias towards (1,1)
    # next_action = 2*np.random.randn(*sample.shape) + bias/10
    next_action = np.random.uniform(low=action_space.low, high=action_space.high, size=action.shape) + bias/10
    next_action = np.clip(next_action, action_space.low, action_space.high)
    return next_action

def sample_ou_policy(action_space, seq_len, dt, theta=0.15, sigma=0.2, mu=0.0):
    """Sample a sequence of actions from an Ornstein-Uhlenbeck process."""
    action_dim = action_space.shape[0]
    sample = action_space.sample()
    actions = [np.zeros_like(sample)]*seq_len
    actions[0] = sample
    
    for i in range(1,seq_len):
        noise = np.random.randn(action_dim)
        next_action = actions[i-1] + theta * (mu - actions[i-1]) * dt + sigma * np.sqrt(dt) * noise
        next_action = np.clip(next_action, action_space.low, action_space.high)
        actions[i] = next_action

    return actions

def sample_switching_ou_policy(action_space, seq_len, dt, switch_interval=20,
                                theta=0.9, sigma=0.2):
    action_dim = action_space.shape[0]
    sample = action_space.sample()
    actions = [np.zeros_like(sample)]*seq_len
    actions[0] = sample
    mu = np.zeros(action_dim)

    for i in range(1,seq_len):
        if i % switch_interval == 0:
            mu = np.random.uniform(low=action_space.low, high=action_space.high)
        
        noise = np.random.randn(action_dim)
        next_action = actions[i-1] + theta * (mu - actions[i-1]) * dt + sigma * np.sqrt(dt) * noise
        next_action = np.clip(next_action, action_space.low, action_space.high)
        actions[i] = next_action

    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        # self.env = gym.make('CarRacing-v0')
        self.env = gymnasium.make("PlanarCircle-v0", render_mode='rgb_array',
                         camera_id=0, camera_width=64, camera_height=64)
        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        self.env.render()

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                return - cumulative
            i += 1

# # testing plotting the actuation sampling
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import gymnasium
#     env = PlanarCircleEnv(render_mode='rgb_array',
#                          camera_id=0, camera_width=64, camera_height=64)
#     # actions = sample_ou_policy(env.action_space, 1000, 1/10)
#     # actions = sample_switching_ou_policy(env.action_space, 1000, 1/10, switch_interval=10)
#     # actions = sample_brown_policy(env.action_space, 1000, 1/10)
#     # plt.plot(np.array([action[0] for action in actions]))
#     # plt.plot(np.array([action[1] for action in actions]))
#     # plt.show()

#     # test the sample_biased_brown_policy
#     state = env.reset()[0]
#     action = env.action_space.sample()
#     actions = np.array([action])
#     states = np.array([state])
#     for _ in range(500):
#         # action = sample_biased_brown_policy(env.action_space, 1/10, state, action)
#         action = sample_biased_white_policy(env.action_space, 1/10, state, action)
#         state, reward, done, _, _ = env.step(action)
#         actions = np.vstack((actions, action))
#         states = np.vstack((states, state))
#     env.close()
    
#     fig, axs = plt.subplots(3, 1, figsize=(10, 6))
#     axs[0].plot(actions[:, 0], label='x')
#     axs[0].plot(actions[:, 1], label='y')
#     axs[0].set_title('Actions')
#     axs[0].set_xlabel('Time step')
#     axs[0].set_ylabel('Action value')
#     axs[0].legend()
#     axs[1].plot(states[:, 0], label='x')
#     axs[1].plot(states[:, 1], label='y')
#     axs[1].set_title('States')
#     axs[1].set_xlabel('Time step')
#     axs[1].set_ylabel('State value')
#     axs[1].legend()
#     # plot the x-y trajectory
#     axs[2].plot(states[:, 0], states[:, 1], label='Trajectory')
#     axs[2].set_title('X-Y Trajectory')
#     axs[2].set_xlabel('X position')
#     axs[2].set_ylabel('Y position')
#     axs[2].legend()
#     axs[2].axis('equal')
#     plt.tight_layout()
#     plt.show()