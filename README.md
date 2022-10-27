# PPO-AE

## Installation
```
pip install -r requirements.txt
```

## Code overview

- `ppo_ae_joint_minigrid_rgb.py`: Training a PPO jointly with an Auto Encoder, observation space is image
- `ppo_ae_joint_minigrid.py`: Training a PPO jointly with an Auto Encoder, observation space is observation in minigrid
- `ppo_minigrid.py`: Train PPO on minigrid
- `ppo_minigrid_rgb.py`: Train PPO on minigrid, observation space is image
- `ppo_ae_lsh_counting_rgb.py`: Training a PPO jointly with an Auto Encoder, observation space is image, the rewards are augmented with UCB intrinsic rewards estimated from a hasing count-based method.
- `ppo_ae_separate_minigrid_rgb.py`: Train PPO on a pre-trained Auto-Encoder, data for AE is collected prior to the RL training and by a random agent

## Run
```
python ppo_ae_minigrid.py --env-id MiniGrid-SimpleCrossingS9N1-v0 --exp-name ppo_with_ae --total-timesteps 200_000 --learning-rate 3e-4 --ae-dim 5 --ae-batch-size 256 --ae-env-step 20000 --ae-training-step 40000
```

```
python ppo_minigrid.py --env-id MiniGrid-SimpleCrossingS9N1-v0 --exp-name pure_ppo --total-timesteps 200_000 --learning-rate 3e-4
```

```
python -m tensorboard.main --logdir=./
```
