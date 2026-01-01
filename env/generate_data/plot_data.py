# just plotting some data from the rollouts in order to get an idea 
# of the data and if it is reasonable
import numpy as np
import matplotlib.pyplot as plt

# load datasets/planarcircle/thread_0/rollout_0.npz
root_dir = 'datasets/planarcircle/A_to_B'
thread_id = 1
rollout_id = 1
data = np.load(f'{root_dir}/thread_{thread_id}/rollout_{rollout_id}.npz')
observations = data['observations']
states = data['states']

# as the duration of the episode is 100, we can plot the first 100 frames
fig, axs = plt.subplots(10,40, figsize=(25, 8))
axs = axs.flatten()
for i in range(len(axs)):
    axs[i].imshow(observations[i])
    axs[i].set_title(f'{i}')
    axs[i].axis('off')
fig.savefig(f'{root_dir}/thread_{thread_id}/rollout_{rollout_id}.png', dpi=300)
plt.show()

