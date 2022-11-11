import gymnasium
import gym
from ppo_adv_exploration_fourroom import stateRecording, MovementActionWrapper

env = gymnasium.make("MiniGrid-FourRooms-v0", )
from minigrid.wrappers import ImgObsWrapper,FlatObsWrapper, RGBImgObsWrapper
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)
env = MovementActionWrapper(env)
env.reset()
rc = stateRecording(env)
rc.get_figure_log_scale(10_000)
import matplotlib.pyplot as plt;
plt.show();