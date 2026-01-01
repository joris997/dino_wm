"""
Generating data from the CarRacing gym environment.
"""
import os
import sys
import matplotlib.pyplot as plt

import argparse
from os.path import join, exists
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.generate_data.misc import sample_brown_policy, sample_ou_policy, sample_switching_ou_policy, \
    sample_biased_white_policy, sample_biased_brown_policy
from env.planarcircle.planarcircle_env import PlanarCircleEnv

def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    # assert exists(data_dir), "The data directory does not exist..."

    env = PlanarCircleEnv(render_mode='rgb_array',
                          camera_id=0, camera_width=224, camera_height=224)
    seq_len = 150

    for i in range(rollouts):
        x = env.reset()[0]
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_brown_policy(env.action_space, seq_len, 0.1)
        elif noise_type == 'ou':
            a_rollout = sample_ou_policy(env.action_space, seq_len, 0.1)
        elif noise_type == 'switching_ou':
            a_rollout = sample_switching_ou_policy(env.action_space, seq_len, 0.1, switch_interval=10)
        elif noise_type == 'none':
            a_rollout = [np.zeros(env.action_space.shape) for _ in range(seq_len)]
        else:
            # for some scenarios we cannot use a fixed action sequence (it is state dependent)
            a_rollout = []

        x_rollout = []
        s_rollout = []
        r_rollout = []
        d_rollout = []
        i_rollout = []

        t = 0
        action = env.action_space.sample()
        B_loc = np.random.uniform(-2, 3, size=2)
        while t < seq_len:
            if noise_type == 'biased_brown':
                action = sample_biased_brown_policy(env.action_space, dt=0.1, state=x, action=action)
                a_rollout += [action]
            elif noise_type == 'A_to_B':
                P, D = 10, 5
                action = np.clip(
                    P * (B_loc - x[:2]) - D * x[2:4],
                    env.action_space.low, env.action_space.high
                )
                a_rollout += [action]
            else:
                action = a_rollout[t]

            if t % 40 == 0:
                B_loc = np.random.uniform(-2, 3, size=2)

            t += 1

            x, r, done, info = env.step(action)
            s = env.render()

            # env.env.viewer.window.dispatch_events()
            # now the action is the action which took the state
            # to its current state. Instead, we want the action
            # which will be taken at the current state.
            x_rollout += [x]
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            i_rollout += [info]
            if done or t == seq_len:
                # # plot the s_rollout fames in a matplotlib plot
                # fig, axs = plt.subplots(10,15, figsize=(15,10))
                # for idx, frame in enumerate(s_rollout):
                #     ax = axs[idx // 15, idx % 15]
                #     ax.imshow(frame)
                #     ax.axis("off")
                # plt.show()
                
                # index with [1:] and [:-1] to ensure that action[i]
                # is the action taken at state [i] leading to state [i+1]
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         states=np.array(x_rollout[:-1]),
                         rewards=np.array(r_rollout[:-1]),
                         actions=np.array(a_rollout[1:]),
                         terminals=np.array(d_rollout[:-1]),
                         infos=np.array(i_rollout[:-1],dtype=object))
                # save video to mp4 format
                video_path = join(data_dir, 'rollout_{}.mp4'.format(i))
                import cv2
                height, width, _ = s_rollout[0].shape
                video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                for frame in s_rollout[1:]:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video.write(frame_bgr)
                video.release()
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts", default=10)
    parser.add_argument('--dir', type=str, help="Where to place rollouts", default='datasets/planarcircle')
    parser.add_argument('--policy', type=str, choices=['white', 'brown', 'ou', 'none',
                                                       'biased_brown', 'switching_ou', 'A_to_B'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)
