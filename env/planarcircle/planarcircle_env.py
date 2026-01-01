import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from torch.utils.tensorboard import SummaryWriter
import torch
import os

class Circle(object):
    def __init__(self, radius=1.0, center=(0, 0)):
        self.radius = radius
        self.center = np.array(center)

    def contains(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

    def distance(self, point):
        return np.linalg.norm(point - self.center) - self.radius
    
class Rectangle(object):
    def __init__(self, width=1.0, height=1.0, center=(0, 0)):
        self.width = width
        self.height = height
        self.center = np.array(center)

    def contains(self, point):
        return (abs(point[0] - self.center[0]) <= self.width / 2) and (abs(point[1] - self.center[1]) <= self.height / 2)

    def distance(self, point):
        dx = max(abs(point[0] - self.center[0]) - self.width / 2, 0)
        dy = max(abs(point[1] - self.center[1]) - self.height / 2, 0)
        return np.sqrt(dx**2 + dy**2)

class PlanarCircleEnv(MujocoEnv, gym.utils.EzPickle):
    # this is a gym env from a mujoco xml file
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, 
                #  xml_path=os.path.abspath("envs/assets/planar_circle.xml"), 
                 xml_path=os.path.abspath("env/planarcircle/assets/planar_circle.xml"), 
                 frame_skip=5, 
                 render_mode=None, 
                 weights=None,

                 camera_id=0,
                 camera_width=224,
                 camera_height=224,
                 
                 reset_to_state=None,
                 gmm_classifier:dict=None,
                 log_dir="logs",
                 goal_area='area1',
                 **kwargs):
        self._frame_skip = frame_skip
        observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)

        MujocoEnv.__init__(self,
                           model_path=xml_path,
                           frame_skip=frame_skip,
                           observation_space=observation_space,
                           render_mode=render_mode,
                           camera_id=camera_id,
                           width=camera_width,
                           height=camera_height)
        gym.utils.EzPickle.__init__(self)

        self.areas = {
            'area1': Rectangle(width=1.0, height=1.0, center=(-1.0,-1.0)),
            'area2': Rectangle(width=1.0, height=1.0, center=(3.0,-1.0)),
            'area3': Rectangle(width=1.0, height=1.0, center=(-1.0,3.0)),
            'area4': Rectangle(width=1.0, height=1.0, center=(3.0,3.0)),
            'obstacle': Rectangle(width=1.0, height=1.0, center=(1.0,1.0))
        }
        self.goal_area = goal_area

        # weights has to be a dict
        if weights is None or not isinstance(weights, dict):
            self.weights = {"goal_distance": 0.01,
                            "at_goal": 0.0,
                            "not_at_goal": -1.0,
                            "actuation": 0.0,
                            "obstacle_distance": 0.0,
                            "at_obstacle": 0.0,
                            "not_at_obstacle": 0.0}
        else:
            self.weights = weights

        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

        self.gmm_classifier = gmm_classifier
        self.reset_to_state = reset_to_state

    def reset(self):
        if self.reset_to_state is not None:
            self.set_state(self.reset_to_state[:2], self.reset_to_state[2:4])
            observation = self._get_obs()
            info = {}
            return observation, info
        else:
            return super().reset()

    def step(self, action:np.ndarray)->np.ndarray | np.float64 | bool | bool | dict:
        agent_pos = self._get_obs()[0:2] #agent_pos = self.get_body_com("circle")
        agent_vel = self._get_obs()[2:4] #agent_vel = self.data.body("circle").cvel

        agent_to_goal = agent_pos[:2] - self.areas[self.goal_area].center

        in_goal = self.areas[self.goal_area].contains(agent_pos)
        in_vel_range = np.linalg.norm(agent_vel) < 0.1

        # y = self.render()
        # # get latent state
        # with torch.no_grad():
        #     lstate = self.vae.encoder(self.real_obs_to_vae_input(y))[0]
        # pred_prob = get_probability(lstate.detach().cpu().numpy(),
        #                             self.gmm_classifier['gmm_goal'],
        #                             self.gmm_classifier['gmm_non_goal'],
        #                             self.gmm_classifier['priors'])
        

        done = False
        reward = 0

        # reward = pred_prob
        if not in_goal:
            reward = -1
            
        # if pred_prob > 0.45:
        #     done = True
        #     reward = 20*pred_prob #10.0
        # if in_goal and in_vel_range:
        #     reward = 10
        #     done = True

        self.do_simulation(action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {key: self.areas[key].contains(agent_pos) for key in self.areas.keys()}
        return obs, reward, done, info
    
    def reset_model(self)->np.ndarray:
        # set the circle to a random position in the area of -5+radius to 5-radius and zero velocity
        # if x and y are linear joints
        x0 = np.random.uniform(low=-3, high=5, size=(2,))  # reset position
        dx0 = np.random.uniform(low=0.0, high=0.0, size=(2,))  # reset velocity
        
        self.set_state(x0, dx0)
        return self._get_obs()
    
    def sample_model(self)->np.ndarray:
        # set the circle to a random position in the area of -5+radius to 5-radius
        # and set the velocity to zero (helps with initializing the RNN)
        x0 = np.concatenate([
            np.random.uniform(low=-3, high=5, size=(2,)), 
            np.zeros(5)])
        dx0 = np.concatenate([
            np.random.uniform(low=0.0, high=0.0, size=(2,)),
            np.zeros(4)])
        self.set_state(x0, dx0)
        return self._get_obs()
    
    def real_obs_to_vae_input(self, real_obs):
        # real_obs = [64,64,3] in range [0,255]
        # input should be [3,64,64] in range [0,1]
        vae_input = np.transpose(real_obs, (2, 0, 1)) / 255
        vae_input = torch.tensor(vae_input.copy()).float().unsqueeze(0)
        return vae_input
    
    def render(self, mode=None):
        # assert that mode is equal to the render_mode
        # assert mode == self.render_mode, f"mode {mode} must be equal to render_mode {self.render_mode}"
        return super().render()
    
    def _get_obs(self)->np.ndarray:
        return np.concatenate([
            self.data.qpos.flat[:2],
            self.data.qvel.flat[:2]
        ],dtype=np.float32)

# env = PlanarCircleEnv(render_mode="rgb_array")
# obs, _ = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, reward, done, _, _ = env.step(action)
#     if done:
#         break
#     # # display the rgb array in matplotlib using plt.imshow
#     # import matplotlib.pyplot as plt
#     # plt.imshow(env.render())
#     # plt.axis('off')
#     # plt.show(block=False)
#     # plt.pause(1e-6)

# env.close()