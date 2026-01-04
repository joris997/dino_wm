"""
Encapsulate generate data to make it parallel
"""
import os
from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--rollouts', type=int, help="Total number of rollouts.", default=10_000) # was 10_000
parser.add_argument('--threads', type=int, help="Number of threads", default=24)
parser.add_argument('--rootdir', type=str, help="Directory to store rollout "
                    "directories of each thread", default="datasets/data/planarcircle")
parser.add_argument('--policy', type=str, choices=['brown', 'white', 'ou', 
                                                   'switching_ou','biased_brown','A_to_B','none'],
                    help="Directory to store rollout directories of each thread",
                    default='A_to_B')
parser.add_argument('--add_to_existing', type=str, help="Whether to add to existing data, should be --policy of the other data",
                    default=None)

args = parser.parse_args()

rpt = args.rollouts // args.threads + 1
rootdir = join(args.rootdir, args.policy)
if args.add_to_existing is not None:
    rootdir = join(args.rootdir, args.add_to_existing)
    # find highest thread_{} number in train and val dirs
    existing_threads = []
    for split in ['train', 'val']:
        split_dir = join(rootdir, split)
        for d in os.listdir(split_dir):
            if d.startswith('thread_'):
                existing_threads.append(int(d.split('_')[1]))
    start_thread = max(existing_threads) + 1
else:
    start_thread = 0

def _threaded_generation(i):
    iddir = 'thread_{}'.format(i)
    if i <= int(0.9*(start_thread + args.threads)):
        tdir = join(rootdir, 'train', iddir)
    else:
        tdir = join(rootdir, 'val', iddir)
    makedirs(tdir, exist_ok=True)

    cmd = ["/home/planiacs/miniconda3/envs/rl/bin/python", "-m", "env.generate_data.planarcircle", "--dir",
            tdir, "--rollouts", str(rpt), "--policy", args.policy]
    cmd = " ".join(cmd)
    print(cmd)
    call(cmd, shell=True)
    return True


# now make train and val dirs inside
print(args.rootdir)
makedirs(join(rootdir, 'train'), exist_ok=True)
makedirs(join(rootdir, 'val'), exist_ok=True)
with Pool(args.threads) as p:
    p.map(_threaded_generation, range(start_thread, start_thread + args.threads))

