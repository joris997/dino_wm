"""
Simulated planarcircle environment.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from os.path import join, exists
import torch
from torch.distributions.categorical import Categorical
import gymnasium
from gymnasium import spaces
from models.vae import VAE
from models.classifier import Classifier
from models.camdrnn import CAMDRNNCell, CAMDRNN
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
import numpy as np
import time

from envs.planar_circle import PlanarCircleEnv

device = 'cuda' 



# load the classifier
classifier_file = os.path.join("exp_dir", "classifier", "best.tar")
assert os.path.exists(classifier_file), "No trained classifier in the logdir..."
classifier = Classifier(LSIZE, 128, 2, 5).to(device)
classifier_state = torch.load(classifier_file, map_location=lambda storage, location: storage)
print("Loading Classifier at epoch {}, "
        "with test error {}...".format(
            classifier_state['epoch'], classifier_state['precision']))
classifier.load_state_dict(classifier_state['state_dict'])
classifier.eval()


class SimulatedPlanarCircle(gymnasium.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Planar Circle.

    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v0 environment.

    :args directory: directory from which the vae and mdrnn are
    loaded.
    """
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 30}
    def __init__(self, directory, real_env:gymnasium.Env):
        vae_file = join(directory, 'vae', 'best.tar')
        rnn_file = join(directory, 'mdrnn', 'best.tar')
        assert exists(vae_file), "No VAE model in the directory..."
        assert exists(rnn_file), "No MDRNN model in the directory..."

        # spaces
        self.action_space = spaces.Box(np.array([-10,-10]), np.array([10, 10]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(RED_SIZE, RED_SIZE, 3),
                                            dtype=np.uint8)
        
        # load real environment
        self.real_env = real_env
        
        # load VAE
        vae = VAE(3, LSIZE).to(device)
        vae_state = torch.load(vae_file, map_location=lambda storage, location: storage)
        print("Loading VAE at epoch {}, "
              "with test error {}...".format(
                  vae_state['epoch'], vae_state['precision']))
        vae.load_state_dict(vae_state['state_dict'])
        vae.eval()
        self._vae = vae
        self._encoder = vae.encoder
        self._decoder = vae.decoder

        # load MDRNNCell
        self._rnn = CAMDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        rnn_state = torch.load(rnn_file, map_location=lambda storage, location: storage)
        print("Loading MDRNN at epoch {}, "
              "with test error {}...".format(
                  rnn_state['epoch'], rnn_state['precision']))
        rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
        self._rnn.load_state_dict(rnn_state_dict)

        # init state
        self._x = self.real_env.reset()[0]
        # need zero velocity initial state because the lstm does not know velocity
        while np.linalg.norm(self._x[2:]) > 0.1:
            self._x = self.real_env.reset()[0]
        self._y = self.real_env.render()
        self._lstate = self._encoder(torch.tensor(np.transpose(self._y, (2, 0, 1)).copy()).to(device).float().unsqueeze(0))[0]
        self._hstate = 2 * [torch.zeros(1, RSIZE).to(device)]

        # obs
        self._latent_obs = None
        self._real_obs = None

        # rendering
        self.monitor = None
        self.figure = None

    def reset(self):
        """ Resetting """
        import matplotlib.pyplot as plt
        self._x = self.real_env.reset()[0]
        # need zero velocity initial state because the lstm does not know velocity
        self._y = self.real_env.render()
        with torch.no_grad():
            self._lstate = self._encoder(self.real_obs_to_vae_input(self._y))[0]
        self._hstate = 2 * [torch.zeros(1, RSIZE).to(device)]

        # get the initial visualizations
        self._latent_obs = self.get_latent_obs(self._lstate)
        self._real_obs = self.get_real_obs(self._y)
        
        # also reset monitor
        if not self.monitor:
            self.figure, self.axs = plt.subplots(1,2,figsize=(10,5))
            self.latent_monitor = self.axs[0].imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
            self.real_monitor = self.axs[1].imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
        self.latent_monitor.set_data(self._latent_obs)
        self.real_monitor.set_data(self._real_obs)
        plt.pause(0.01)

    def step(self, action):
        """ One step forward """
        with torch.no_grad():
            t0 = time.time()
            action = torch.Tensor(action).to(device).unsqueeze(0)

            # latent environment step
            mu, sigma, pi, r, d, n_h, dynamics = self._rnn(action, self._lstate, self._hstate)
            print(f"Latent step took {time.time() - t0:.3f} seconds")
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()
            self._lstate = mu[:, mixt, :] # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            self._hstate = n_h

            t0 = time.time()
            self._latent_obs = self.get_latent_obs(self._lstate)
            print(f"Latent observation took {time.time() - t0:.3f} seconds")

            # real environment step
            self._x = self.real_env.step(action.cpu().numpy()[0])[0]
            self._y = self.real_env.render()
            self._real_obs = self.get_real_obs(self._y)

            return self._lstate, r.item(), d.item() > 0

    def real_obs_to_vae_input(self, real_obs):
        # real_obs = [64,64,3] in range [0,255]
        # input should be [3,64,64] in range [0,1]
        vae_input = np.transpose(real_obs, (2, 0, 1)) / 255
        vae_input = torch.tensor(vae_input.copy()).float().unsqueeze(0).to(device)
        return vae_input

    def get_latent_obs(self, latent):
        """ Get latent image """
        with torch.no_grad():
            obs = self._decoder(latent)
            np_obs = obs.cpu().numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze().astype(np.uint8)
        return np_obs
    
    def get_real_obs(self, real_obs):
        """ Get real image """
        return real_obs

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt
        if not self.figure:
            self.figure, self.axs = plt.subplots(1,2,figsize=(10,5))
            self.latent_monitor = self.axs[0].imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
            self.real_monitor = self.axs[1].imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
        self.latent_monitor.set_data(self._latent_obs)
        self.real_monitor.set_data(self._real_obs)
        plt.pause(0.01)

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Directory from which MDRNN and VAE are '
                        'retrieved.', default='exp_dir')
    
    real_env = PlanarCircleEnv(render_mode='rgb_array',
                            camera_id=0, camera_width=64, camera_height=64)
    
    args = parser.parse_args()
    latent_env = SimulatedPlanarCircle(args.logdir, real_env)

    latent_env.reset()
    action = np.array([0., 0.])

    u_max = 5
    def on_key_press(event):
        """ Defines key pressed behavior """
        if event.key == 'left':
            action[0] = -u_max
        if event.key == 'right':
            action[0] = u_max
        if event.key == 'up':
            action[1] = u_max
        if event.key == 'down':
            action[1] = -u_max

    def on_key_release(event):
        """ Defines key pressed behavior """
        if event.key == 'left':
            action[0] = 0
        if event.key == 'right':
            action[0] = 0
        if event.key == 'up':
            action[1] = 0
        if event.key == 'down':
            action[1] = 0

    latent_env.figure.canvas.mpl_connect('key_press_event', on_key_press)
    latent_env.figure.canvas.mpl_connect('key_release_event', on_key_release)
    while True:
        obs, _, done = latent_env.step(action)
        latent_env.render()
        # print the classifier output on the latent state (obs)
        obs = torch.tensor(obs).to(device)
        print(f"Action: {action}")
        print(f"Classifier: {classifier(obs)[0,0].item():.2f}")
        # if done:
        #     break
