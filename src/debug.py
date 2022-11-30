import gymnasium
import gym
from .ppo_ae_cnt_distance_minigrid_rgb import *
from utils.common import *

env = gymnasium.make("MiniGrid-FourRooms-v0", )
from minigrid.wrappers import ImgObsWrapper,FlatObsWrapper, RGBImgObsWrapper
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)
env = MovementActionWrapper(env)
env.reset()
rc = stateRecording(env)
tc = TrajectoryVisualizer(env)

for _ in range(100):
    nexts = env.step(env.action_space.sample())
    tc.add_frame(nexts[0])

# rc.get_figure_log_scale(10_000)
import matplotlib.pyplot as plt;
# plt.show();
tc.plot_trajectory(PixelEncoder(env.observation_space.shape, 10))
plt.show()