import os
import random
import time
from distutils.util import strtobool
import minigrid
import gym
from gym import spaces
import numpy as np
from functools import reduce
import operator

import gym
import gymnasium
import numpy as np


class CustomFlatObsWrapper(gym.core.ObservationWrapper):
    '''
    This is the extended version of the `FlatObsWrapper` from `gym-minigrid`,
    Which only considers the case where the observation contains both `image` and `mission`
    This custom wrapper can work with both cases, i.e whether the `mission` presents or not
    Since `mission` can be discarded when being wrapped with `ImgObsWrapper` for example.
    '''
    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        if isinstance(env.observation_space, spaces.Dict):
            imgSpace = env.observation_space.spaces['image']
        else:
            imgSpace = env.observation_space
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + self.numCharCodes * self.maxStrLen,),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        if isinstance(obs, dict):
            return self._observation(obs)
        return obs.flatten()


    def _observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs
        
class TransposeImageWrapper(gym.ObservationWrapper):
    '''Transpose img dimension before being fed to neural net'''
    def __init__(self, env, op=[2,0,1]):
        super().__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class MovementActionWrapper(gym.core.ActionWrapper):
    """
    For `Minigrid` envs only
    Limit the action space to only take the movement actions, ignoring pickup, drop, toggle and done actions
    Note that this should be used inclusively for several Minigrid, 
    as many other envs requires agent to take additional actions, e.g picking key to open doors
    """
    def __init__(self, env, max_action=3):
        super().__init__(env)
        self.max_action = max_action
        self.action_space = gym.spaces.Discrete(max_action)

    def action(self, action):
        return np.clip(action, a_min=0, a_max=self.max_action)

class EpisodicLifeEnv(gym.Wrapper):
    """
    This file is copied from Stable-baselines3, with some modification to make it
    compatible with new `gym` API.
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

def make_minigrid_rgb_env(env_id, seed, idx, capture_video, run_name, reseed=False, restrict_action:int=None):
    def thunk():
        env = gymnasium.make(env_id)
        from minigrid.wrappers import ImgObsWrapper,FlatObsWrapper, RGBImgObsWrapper
        from gym.wrappers import ResizeObservation
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        if reseed:
            from minigrid.wrappers import ReseedWrapper
            env = ReseedWrapper(env, seeds=[seed])
        if restrict_action is not None:
            env = MovementActionWrapper(env, max_action=restrict_action)

        env.action_space = gym.spaces.Discrete(env.action_space.n)
        env.observation_space = gym.spaces.Box(
            low=np.zeros(shape=env.observation_space.shape,dtype=int),
            high=np.ones(shape=env.observation_space.shape,dtype=int)*255
        )
        env = ResizeObservation(env, 84)
        env = TransposeImageWrapper(env)
        print("obs shape", np.array(env.reset()[0]).shape)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        try:
            env.seed(seed)
        except:
            print("cannot seed the environment")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
    
def make_atari_env(env_id, seed, idx, capture_video, run_name, *args, **kwargs):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = EpisodicLifeEnv(env)
        
        from gym.wrappers.atari_preprocessing import AtariPreprocessing
        env = AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
    
def make_env(env_id, *args, **kwargs):
    if "MiniGrid" in env_id:
        return make_minigrid_rgb_env(env_id, *args, **kwargs)
    elif kwargs["atari"]:
        return make_atari_env(env_id, *args, **kwargs)
    else:
        return gym.make(env_id)

class stateRecording:
    """recording state distributions, for ploting visitation frequency heatmap"""
    def __init__(self, env):
        self.shape = env.grid.height, env.grid.width
        self.count = np.zeros(self.shape, dtype=np.int32)
        self.rewards = np.zeros(self.shape, dtype=float)
        self.extract_mask(env)

    def add_count(self, w, h):
        self.count[h, w] += 1

    def add_count_from_env(self, env):
        self.add_count(*env.agent_pos)
        
    def add_reward(self, w, h, r):
        self.rewards[h, w] += r
    
    def add_reward_from_env(self, env, reward):
        self.add_reward(*env.agent_pos, reward)
        
    def get_figure_log_scale(self, cap_threshold_cnt=10_000):
        """ plot heat map visitation, similar to `get_figure` but on log scale"""
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        cnt = np.clip(self.count+1, 0, cap_threshold_cnt)
        plt.clf()
        plt.jet()
        plt.imshow(cnt, cmap="jet", 
            norm=matplotlib.colors.LogNorm(vmin=1, vmax=cap_threshold_cnt, clip=True))
        cbar=plt.colorbar()
        cbar.set_label('Visitation counts')

        # over lay walls
        plt.imshow(np.zeros_like(cnt, dtype=np.uint8),
                cmap="gray", alpha=self.mask.astype(np.float),
                vmin=0, vmax=1)
        return plt.gcf()

    def get_figure(self, cap_threshold_cnt=5000):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        cnt = np.clip(self.count, 0, cap_threshold_cnt)
        plt.clf()
        plt.jet()
        plt.imshow(cnt, cmap="jet", vmin=0, vmax=cap_threshold_cnt)
        cbar=plt.colorbar()
        lin_spc = np.linspace(0, cap_threshold_cnt, 6).astype(np.int32)
        # cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(lin_spc))
        cbar.update_ticks()
        lin_spc = [str(i) for i in lin_spc]
        lin_spc[-1] = ">"+lin_spc[-1]
        cbar.ax.set_yticklabels(lin_spc)
        cbar.set_label('Visitation counts')

        # over lay walls
        plt.imshow(np.zeros_like(cnt, dtype=np.uint8),
                cmap="gray", alpha=self.mask.astype(np.float),
                vmin=0, vmax=1)
        return plt.gcf()

    def extract_mask(self, env):
        """ Extract walls from grid_env, used for masking wall cells in heatmap """
        self.mask = np.zeros_like(self.count)
        for j in range(env.grid.height):
            for i in range(env.grid.width):
                c = env.grid.get(i, j)
                if c is not None and c.type=="wall":
                    self.mask[j, i]=1
                    
    def save_to(self, file_path):
        with open(file_path, 'wb') as f:
            np.save(f, self.count)
            np.save(f, self.mask)
            
    def load_from(self, file_path):
        with open(file_path, 'rb') as f:
            self.count = np.load(f)
            self.mask = np.load(f)
        self.shape = self.count.shape
        
class TrajectoryVisualizer:
    def __init__(self, env):
        self.frames = []
        self.obs_shape = env.observation_space.shape
        
    def add_frame(self, obs):
        self.frames.append(obs)
        
    def add_traj(self, traj: list):
        self.frames = traj

    def plot_trajectory(self, encoder, device='cpu'):
        import torch
        import numpy as np
        import random
        import matplotlib.pyplot as plt
        
        frames = np.array(self.frames)
        frames = torch.Tensor(frames).to(device).reshape(-1, *self.obs_shape)
        with torch.no_grad():
            encoder.eval()
            embedding = encoder(frames).cpu().numpy().astype(np.float32)
            encoder.train()
        
        if embedding.shape[-1] != 2:
            from sklearn.manifold import TSNE
            embedding = TSNE(n_components=2, 
                   init='random', perplexity=3).fit_transform(embedding)
        
        sample_idx = np.linspace(0, len(self.frames)-1, num=10,dtype=int)
        
        plt.clf()
        plt.jet()
        plt.scatter(embedding[:, 0], embedding[:, 1], 
                c=np.arange(len(embedding)), edgecolors='black', zorder=1)
        
        # from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
        # ax = plt.gca()
        # for indx in sample_idx:
            # patch = self.frames[indx]
            # im = OffsetImage(patch, zoom=.3)
            # im.image.axes = ax
            # ab = AnnotationBbox(im, embedding[indx],
                            # xybox=(random.uniform(-30, 30), random.uniform(-30, 30)),
                            # xycoords='data',
                            # boxcoords="offset points",
                            # pad=0.1,
                            # arrowprops=dict(arrowstyle="->"))

            # ax.add_artist(ab)
        return plt.gcf()

class MiniGridCount:
    """ Count state visitation in MiniGrid, used to derived UCB rewards """
    def __init__(self, envs, hash_size = 20):
        self.envs = envs.envs
        self.cnt = {}
        self.hash_size = hash_size

    def update(self):
        for env in self.envs:
            hash_val = env.hash(self.hash_size)
            self.cnt[hash_val] = self.cnt.get(hash_val, 0) + 1

    def get_cnt(self): 
        cnts = [self.cnt.get(env.hash(self.hash_size), 0) for env in self.envs]
        return np.array(cnts)

def pprint(dict_data):
    '''Pretty print Hyper-parameters'''
    hyper_param_space, value_space = 50, 50
    format_str = "| {:<"+ f"{hyper_param_space}" + "} | {:<"+f"{value_space}"+"}|"
    hbar = '-'*(hyper_param_space + value_space+6)

    print(hbar)
    print(format_str.format('Hyperparams', 'Values'))
    print(hbar)

    for k, v in dict_data.items():
        print(format_str.format(str(k), str(v)))

    print(hbar)
