#!/bin/bash

import os
import sys
import numpy as np
import argparse


env_ids = ["MiniGrid-DoorKey-8x8-v0", 
           "MiniGrid-MultiRoom-N6-v0", 
           "MiniGrid-SimpleCrossingS9N3-v0"]

parser = argparse.ArgumentParser()
parser.add_argument("--n-seeds", type=int, default=2,
        help="total number of seeds")
parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")

args = parser.parse_args()

n_seeds = args.n_seeds
seeds = np.random.randint(9999999, size=n_seeds)
print("seed", seeds)

config = {
    "total-timesteps": args.total_timesteps,
    "learning-rate" : 3e-4,
    "ae-dim": 20,
    "ae-batch-size": 32,
    "ae-buffer-size": 50_000,
    "save-sample-AE-reconstruction-every": 5_000,
    "rw-coef": 1e-3,
    "adjacent_norm_coef": 1e-5,
    "window-size-episode": 100,
    "reduce" : "knn",
    "distance-clip": 0.01,
    "reward-scale": 2.,
    "fixed-seed": "True",
    "update-epochs": 4,
    "reduce": "knn",
    "save-final-model": "True"
}

# Train with VAE
for env_id in env_ids:
    for seed in seeds:
        command = (f"python -m src.ppo_vae_cnt_distance_rgb --env-id {env_id} "
            f"--beta 1e-5 --seed {seed} --deterministic-latent True "
        )
        for h, v in config.items():
            command += f" --{h} {v}"
        print(command)
        os.system(command)
        
# Train with AE
for env_id in env_ids:
    for seed in seeds:
        command = (f"python -m src.ppo_ae_cnt_distance_minigrid_rgb --env-id {env_id} "
            f"--beta 0 --seed {seed}  "
        )
        for h, v in config.items():
            command += f" --{h} {v}"
        print(command)
        os.system(command)