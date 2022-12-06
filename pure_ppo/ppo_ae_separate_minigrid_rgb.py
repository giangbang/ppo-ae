# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
import sys
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils.common import *


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-final-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save the final model at the end of the training or not")
    parser.add_argument("--reward-scale", type=float, default=10,
            help="scaling factor for extrinsic rewards")

    # auto encoder parameters
    parser.add_argument("--ae-dim", type=int, default=50,
        help="number of hidden dim in ae")
    parser.add_argument("--ae-batch-size", type=int, default=128,
        help="AE batch size")
    parser.add_argument("--ae-training-step", type=int, default=10000,
        help="number of training steps in ae")
    parser.add_argument("--load-buffer", type=str, default="",
        help="Load buffer from expert policy if provided")
    parser.add_argument("--ae-buffer-size", type=int, default=100_000,
        help="buffer size for training ae, recommend less than 200k ")
    parser.add_argument("--beta", type=float, default=0.01,
        help="L2 norm of the latent vectors")

    # visualization of the state distribution
    parser.add_argument("--visualize-states", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Visualize state distribution by heatmaps.")
    parser.add_argument("--whiten-rewards", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Hide rewards signal from agent.")
    parser.add_argument("--fixed-seed", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Fixed seed when reset env.")

    args, unknown = parser.parse_known_args()
    if args.visualize_states:
        # only work with 1 environment
        args.num_envs = 1
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Copy from SAC AE
# https://github.com/denisyarats/pytorch_sac_ae/blob/master/encoder.py#L11
# ===================================

OUT_DIM = {2: 39, 4: 35, 6: 31}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        from torchvision.transforms import Resize
        self.resize = Resize((84, 84), interpolation=0) # Input image is resized to []

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        # self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = self.resize(obs)
        obs = (obs -128)/ 128.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)

        self.outputs['latent'] = h_fc

        return h_fc

class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        obs = torch.tanh(obs)
        self.outputs['obs'] = obs

        return obs


# ===================================

class Agent(nn.Module):
    def __init__(self, envs, obs_shape, hidden_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x.detach()) # gradient stopping of the encoder
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # log command to run this file
    import sys
    cmd = " ".join(sys.argv)
    writer.add_text("terminal", cmd)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device used:" , device)

    # setup AE dimension here
    ae_dim=args.ae_dim
    # setup random timesteps to collect data for training AE (prior to training of PPO)
    # (after AE is trained on random samples, this AE is freezed for training PPO)
    ae_env_step = args.ae_buffer_size
    # number of update of AE
    ae_num_train_step = args.ae_training_step
    ae_batch_size = args.ae_batch_size
    # control the l2 regularization of the latent vectors
    beta=args.beta

    pprint(vars(args))

    # env setup
    envs = [make_env(args.env_id, args.seed + i, i, args.capture_video,
            run_name, reseed=args.fixed_seed) for i in range(args.num_envs)]
    import gym
    envs = gym.vector.SyncVectorEnv(
        envs
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, obs_shape=ae_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(agent)
    encoder, decoder = (
        PixelEncoder(envs.single_observation_space.shape, ae_dim).to(device),
        PixelDecoder(envs.single_observation_space.shape, ae_dim).to(device)
    )
    print(encoder)
    print(decoder)
    encoder_optim = optim.Adam(encoder.parameters(), lr=args.learning_rate, eps=1e-5)
    decoder_optim = optim.Adam(decoder.parameters(), lr=args.learning_rate, eps=1e-5)

    # Collect random transitions for training AE
    if len(args.load_buffer) == 0:
        curr_states, _ = envs.reset()
        ae_dataset = [curr_states]
        for _ in range(ae_env_step//args.num_envs):
            actions = envs.action_space.sample()
            next_obs, reward, terminated, truncated, info = envs.step(actions)
            ae_dataset.append(next_obs)

        ae_dataset = np.array(ae_dataset, dtype=np.uint8)

    else:
        ae_dataset = torch.load(args.load_buffer).numpy()

    # reshape to suitable sizes
    ae_dataset = ae_dataset.reshape((-1,)+ envs.single_observation_space.shape)
    print("Training sizes", ae_dataset.shape)

    for i in range(ae_num_train_step):
        indices = np.random.randint(0, len(ae_dataset), ae_batch_size)
        batch = ae_dataset[indices]
        batch = torch.Tensor(batch).to(device)
        latent = encoder(batch)
        reconstruct = decoder(latent)
        assert encoder.outputs['obs'].shape == reconstruct.shape
        loss = torch.nn.functional.mse_loss(reconstruct, encoder.outputs['obs'])
        writer.add_scalar("AE/loss", loss.item(), i)

        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()
    print("done train ae")
    # save sample reconstruction
    # take random 5 images from buffer
    indices = np.random.randint(0, len(ae_dataset), 5)
    batch = ae_dataset[indices]
    batch = torch.Tensor(batch).to(device)
    latent = encoder(batch)
    reconstruct = decoder(latent)
    reconstruct = (reconstruct *128 + 128).clip(0, 255).cpu()

    # log the reconstruct image
    for i, (o, r) in enumerate(zip(batch.cpu(), reconstruct)):
        writer.add_image('image/AE reconstruction', r.type(torch.uint8), i)
        writer.add_image('image/original', o.type(torch.uint8), i)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, args.ae_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # measure success and reward
    rewards_all = np.zeros(args.num_envs)

    # actual training with PPO
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            # encode the observation with AE
            with torch.no_grad():
                next_obs = encoder(next_obs)
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            rewards_all += np.array(reward).reshape(rewards_all.shape)
            done = np.bitwise_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # log success and rewards
            for i, d in enumerate(done):
                if d:
                    writer.add_scalar("train/rewards", rewards_all[i], global_step)
                    writer.add_scalar("train/success", rewards_all[i] > 0.05, global_step)
                    rewards_all[i] = 0

            # for item in info:
            #     if "episode" in item:
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

        # bootstrap value if not done
        with torch.no_grad():
            next_embedding = encoder(next_obs)
            next_value = agent.get_value(next_embedding).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (args.ae_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
