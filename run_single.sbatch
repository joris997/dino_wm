#!/usr/bin/env bash
#SBATCH --gres gpu:6
#SBATCH --mem 48GB
#SBATCH --cpus-per-task 32
#SBATCH --time 48:00:00
#SBATCH --constrain "arwen"
#SBATCH --output /Midgard/home/%u/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

PYTHON=/Midgard/home/jorisv/miniforge3/envs/dino_wm/bin/python
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="c9e67cdc7592d355713d3d450dc3b74f925141ce"

cd /Midgard/home/jorisv/gits/dino_wm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/jorisv/miniforge3

$PYTHON ca_train.py env=pusht
