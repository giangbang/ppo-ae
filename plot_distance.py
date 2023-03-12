import argparse
import gym
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from utils.common import *
from src.ppo_vae_cnt_distance_rgb import PixelEncoder, Agent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--model-weight-path", type=str, default="./",
        help="the path where trained models are")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--ae-dim", type=int, default=50,
        help="number of hidden dim in ae")
    parser.add_argument("--fixed-seed", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Fixed seed when reset env.")

    args, unknown = parser.parse_known_args()
    return args

def get_figure(self, goal_pos=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.clf()
    plt.jet()
    plt.imshow(self.distance_grid, cmap="GnBu", vmin=0)
    cbar=plt.colorbar()
    # cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(lin_spc))
    cbar.update_ticks()
    cbar.set_label('Distance to goal')

    # over lay walls
    plt.imshow(np.zeros_like(self.distance_grid, dtype=np.uint8),
            cmap="gray", alpha=self.mask.astype(float),
            vmin=0, vmax=1)
    # overlay goal
    if goal_pos is not None:
        goal = np.zeros(self.distance_grid.shape + (4,), dtype=np.uint8)
        goal[goal_pos+ (1,)] = 255
        goal[goal_pos+ (3,)] = 255
        plt.imshow(goal)
    return plt.gcf()

stateRecording.get_distance_plot = get_figure

def find_goal(env):
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            c = env.grid.get(i, j)
            if c is not None and c.type == "goal":
                return (j, i)

def randomStart(gym.Wrapper):
    """
    Random start for minigrid fourroom, still keep the layout the same
    """
    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        seed = np.random.randint(low=0)
        from gym.utils import seeding
        # reset seed to a new seed
        self.env._np_random, seed = seeding.np_random(seed)
        self.env.place_agent()
        obs = self.env.gen_obs()
        return obs, {}

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device used:" , device)

    envs = [
        make_env(args.env_id, args.seed + i, i, 0,
        run_name, reseed=args.fixed_seed, atari=False, random_start=True)
        for i in range(1)
    ]
    envs[0] = randomStart(envs[0])
    import gym
    envs = gym.vector.SyncVectorEnv(
        envs
    )

    ae_dim=args.ae_dim

    pprint(vars(args))
    model_path = args.model_weight_path
    checkpoint = torch.load(model_path, map_location=device)
    agent = Agent(envs, obs_shape=ae_dim)
    agent.load_state_dict(checkpoint["agent"])
    encoder = PixelEncoder(envs.single_observation_space.shape, ae_dim)
    encoder.load_state_dict(checkpoint["encoder"])

    # obs = torch.zeros((args.total_timesteps, 1) + envs.single_observation_space.shape).to(device)
    embeddings = torch.zeros((args.total_timesteps, 1, ae_dim)).to(device)
    obs_coord = torch.zeros((args.total_timesteps, 1, 2), dtype=int).to(device)
    goal_obs = []
    goal_embeddings = []
    goal_coord = []

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(1).to(device)

    record_state = stateRecording(envs.envs[0])
    import cv2
    cv2.imwrite("env_obs.png", next_obs.cpu().squeeze().permute([1,2,0]).numpy().astype(np.uint8))

    for global_step in range(args.total_timesteps):
        # obs[global_step] = next_obs # current observation
        obs_coord[global_step] = torch.Tensor(np.array(envs.envs[0].agent_pos).reshape(obs_coord[global_step].shape)).to(device)
        with torch.no_grad():
            next_embedding = encoder.sample(next_obs, deterministic=True)[0]
            action, _, _, _ = agent.get_action_and_value(next_embedding)
            embeddings[global_step] = next_embedding

        next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = np.bitwise_or(terminated, truncated)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
        for i, d in enumerate(done):
            if d and np.sum(reward) > 0.05: # success to reach goal
                # get the position of the real goal
                real_next_obs = info["final_observation"][0]
                with torch.no_grad():
                    real_next_obs = torch.Tensor(real_next_obs).to(device).unsqueeze(0)
                    g_embedding = encoder.sample(real_next_obs, deterministic=True)[0]
                goal_embeddings.append(g_embedding.cpu())
                goal_obs.append(real_next_obs) # observation of the goal position
                goal_coord.append(find_goal(envs.envs[0]))

    print("Number of time reaching goal", len(goal_obs))
    for i in range(1, len(goal_coord), 1):
        assert goal_coord[i] == goal_coord[0]
    # calculate distances
    distances = ((goal_embeddings[0].to(device) - embeddings)**2).sum(dim=-1).sqrt()
    distance_grid = np.zeros(record_state.shape, dtype=np.float32)
    count_grid = np.zeros(record_state.shape, dtype=np.float32) + 1e-5

    for dis, coord in zip(distances, obs_coord):
        # print(dis, coord)
        coord = coord.squeeze()
        distance_grid[coord[0], coord[1]] += dis.item()
        count_grid[coord[0], coord[1]] += 1

    distance_grid_avg = distance_grid / count_grid
    record_state.distance_grid = distance_grid_avg
    record_state.get_distance_plot(goal_coord[0])
    plt.savefig("distance.png")
