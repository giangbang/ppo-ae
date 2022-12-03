#!/bin/bash

import os
import sys
import numpy as np
from distutils.util import strtobool
import argparse


env_ids = ["MiniGrid-DoorKey-8x8-v0",
           "MiniGrid-FourRooms-v0",
           "MiniGrid-SimpleCrossingS9N3-v0"]

parser = argparse.ArgumentParser()
parser.add_argument("--n-seeds", type=int, default=2,
        help="total number of seeds")
parser.add_argument("--total-timesteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
parser.add_argument("--env-indx", type=int, default=0,
        help="run index of the experiments")
# parser.add_argument("--train-vae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
#         help="Training whether AE or VAE.")
parser.add_argument("--use-exp", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Training using exploration bonus or not.")
parser.add_argument("--use-l2", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Training using l2 regularization or not.")
parser.add_argument("--train-ppo-only", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Training a baseline PPO without reconstruction loss")

parser.add_argument("--seed", type=int, default=-1,
        help="seed for training, if this is > 0, will override the n-seeds")


args = parser.parse_args()

if args.seed <= 0:
    n_seeds = args.n_seeds
    seeds = np.random.randint(9999999, size=n_seeds)
else:
    n_seeds = 1
    seeds = [args.seed]
print("seed", seeds)

config = {
    "total-timesteps": args.total_timesteps,
    "learning-rate" : 3e-4,
    "ae-dim": 100,
    "ae-batch-size": 32,
    "ae-buffer-size": 50_000,
    "save-sample-AE-reconstruction-every": 5_000,
    "rw-coef": 1e-2 if args.use_exp else 0,
    "adjacent_norm_coef": 1e-3 if args.use_l2 else 0,
    "window-size-episode": 300,
    "distance-clip": 1,
    "reward-scale": 5,
    "fixed-seed": "True",
    "update-epochs": 3,
    "reduce": "min",
    "save-final-model": "True",
    "ae-warmup-steps": 1000,
    "save-final-buffer": True,
}

env_id = env_ids[args.env_indx]
# Train with VAE
def train_vae():
    for seed in seeds:
        command = (f"python -m src.ppo_vae_cnt_distance_rgb --env-id {env_id} "
            f"--beta 1e-5 --seed {seed} --deterministic-latent True "
        )
        for h, v in config.items():
            command += f" --{h} {v}"
        print(command)
        os.system(command)

# Train with AE
def train_ae():
    for seed in seeds:
        command = (f"python -m src.ppo_ae_cnt_distance_minigrid_rgb --env-id {env_id} "
            f"--beta 0 --seed {seed} --weight-decay 0 "
        )
        for h, v in config.items():
            command += f" --{h} {v}"
        print(command)
        os.system(command)

def train_ppo_only():
    for seed in seeds:
        command = (f"python -m pure_ppo.ppo_minigrid_rgb --env-id {env_id} "
            f" --seed {seed} "
        )
        for h, v in config.items():
            command += f" --{h} {v}"
        print(command)
        os.system(command)

if args.train_ppo_only:
    train_ppo_only()
else:
    # train both ae and vae on the same seed with minigrid
    train_vae()
    train_ae()
