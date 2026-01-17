import numpy as np

# load the numpy file
file_name = "/home/planiacs/gits/dino_wm/datasets/data/planarcircle/A_to_B/train/thread_20/rollout_163.npz"
with open(file_name, "rb") as f:
    data = np.load(f)
# print the keys in the numpy file
print("Keys in the numpy file:", data.keys())