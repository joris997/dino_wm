"""
Encapsulate generate data to make it parallel
"""
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
args = parser.parse_args()

rpt = args.rollouts // args.threads + 1
rootdir = join(args.rootdir, args.policy)

def _threaded_generation(i):
    iddir = 'thread_{}'.format(i)
    if i <= 22:
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
    p.map(_threaded_generation, range(args.threads))

