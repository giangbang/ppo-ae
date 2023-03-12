import argparse
import gym
import gymnasium
import numpy as np
import torch
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

    args, unknown = parser.parse_known_args()
    return args

def get_figure(self):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.clf()
    plt.jet()
    plt.imshow(self.distance_grid, cmap="GnBu", vmin=0)
    cbar=plt.colorbar()
    # cbar.ax.yaxis.set_major_locator(ticker.FixedLocator(lin_spc))
    cbar.update_ticks()
    lin_spc = [str(i) for i in lin_spc]
    lin_spc[-1] = ">"+lin_spc[-1]
    cbar.ax.set_yticklabels(lin_spc)
    cbar.set_label('Distance to goal')

    # over lay walls
    plt.imshow(np.zeros_like(cnt, dtype=np.uint8),
            cmap="gray", alpha=self.mask.astype(np.float),
            vmin=0, vmax=1)
    return plt.gcf()

stateRecording.get_distance_plot = get_figure

def find_goal(env):
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            c = env.grid.get(i, j)
            if c is not None and c.type == "goal":
                return (j, i)

if __name__ == "__main__":
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device used:" , device)

    envs = [
        make_env(args.env_id, args.seed + i, i, 0,
        run_name, reseed=args.fixed_seed, atari=False)
        for i in range(1)
    ]

    ae_dim=args.ae_dim

    pprint(vars(args))
    model_path = args.model_weight_path
    checkpoint = torch.load(model_path)
    agent = Agent(envs, obs_shape=ae_dim)
    agent.load_state_dict(checkpoint["agent"], map_location=device)
    encoder = PixelEncoder(envs.single_observation_space.shape, ae_dim)
    encoder.load_state_dict(checkpoint["encoder"], map_location=device)

    obs = torch.zeros((args.total_timesteps, 1) + envs.single_observation_space.shape).to(device)
    embeddings = torch.zeros((args.total_timesteps, 1, ae_dim)).to(device)
    obs_coord = torch.zeros((args.total_timesteps, 1) + 2).to(device)
    goal_obs = []
    goal_embeddings = []
    goal_coord = []

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    record_state = stateRecording(envs.envs[0])

    for global_step in range(args.total_timesteps):
        obs[global_step] = next_obs # current observation
        obs_coord[global_step] = np.array(env.agent_pos).reshape(obs_coord[global_step].shape)
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
                real_next_obs = info["terminal_observation"]
                with torch.no_grad():
                    real_next_obs = torch.Tensor(real_next_obs).to(device)
                    g_embedding = encoder.sample(real_next_obs, deterministic=True)[0]
                goal_embeddings.append(g_embedding.cpu())
                goal_obs.append(real_next_obs) # observation of the goal position
                goal_coord.append(find_goal(envs.envs[0]))

    print(len(goal_obs))
    # calculate distances
    distances = ((goal_embeddings[0].to(device) - embeddings)**2).sum(dim=-1).sqrt()
    distance_grid = np.zeros(record_state.shape, dtype=np.float32)
    count_grid = np.zeros(record_state.shape, dtype=int)
    for dis, coord in zip(distances, obs_coord):
        distance_grid[coord[-2:]] += dis
        count_grid[coord[-2:]] += 1

    distance_grid_avg = distance_grid / count_grid
    record_state.distance_grid = distance_grid_avg
    record_state.get_distance_plot()
    plt.savefig("distance.png")
