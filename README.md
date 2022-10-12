# PPO-AE

## Installation
```
pip install -r requirements.txt
```

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