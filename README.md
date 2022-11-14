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
= `ppo_adv_exploration_fourroom.py`: Train PPO, using adversarial exploration without extrinsic reward, heatmap is logged to visualize the state visitation exploration

## Review code
- [x] `ppo_ae_joint_minigrid_rgb.py`
- [ ] `ppo_ae_joint_minigrid.py`
- [ ] `ppo_minigrid.py`
- [x] `ppo_minigrid_rgb.py`
- [x] `ppo_ae_advesarial_minigrid_rgb.py`

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

## Results
All of the experiments are run with image observation space
run with adversarial exploration
```
python ppo_ae_advesarial_minigrid_rgb.py --env-id MiniGrid-SimpleCrossingS9N1-v0 --exp-name ppo_with_ae_adversarial_exploration \
              --total-timesteps 500_000 --learning-rate 1e-3 --ae-dim 50 \
              --seed 1 --beta 0.0001 --ae-buffer-size 200_000 --save-sample-AE-reconstruction-every 1_000 \
              --adv-rw-coef 1
```
run with pure ppo
```
python ppo_minigrid_rgb.py --env-id MiniGrid-SimpleCrossingS9N1-v0 --exp-name pure_ppo_img 
               --total-timesteps 500_000 --learning-rate 1e-3 
               --seed 1 --ae-dim 50
```
run ppo with AE, jointly trained with policy
```
python ppo_ae_joint_minigrid_rgb.py --env-id MiniGrid-SimpleCrossingS9N1-v0 --exp-name ppo_with_ae \
              --total-timesteps 500_000 --learning-rate 1e-3 --ae-dim 50 \
              --seed 1 --beta 0.0001 --ae-buffer-size 200_000 --save-sample-AE-reconstruction-every 1_000
```
results
![training losses](/img/loss.png "Training losses")
![training rewards](/img/reward.png "Training rewards")
![Legend](/img/legend.png "Legend")