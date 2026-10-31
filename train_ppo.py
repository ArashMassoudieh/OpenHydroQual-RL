"""
PPO (clipped surrogate, GAE) in plain PyTorch for OHQReservoirEnv.

Each worker process owns its own OHQ instance and working folder, so rollouts
run in parallel through gymnasium's AsyncVectorEnv.

    python train_ppo.py --total-steps 200000 --num-envs 8
"""
import argparse
import os
import time

# One tiny OHQ model per worker process: keep every library single-threaded,
# otherwise N workers x M threads oversubscribe the CPU (set before imports).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from ohq_reservoir_env import OHQReservoirEnv
from ohq_cascade_env import CascadeEnv

ENVS = {"reservoir": OHQReservoirEnv, "cascade": CascadeEnv}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if deterministic:
            return self.actor(obs).clamp(-1, 1).numpy()
        d = self.dist(obs)
        a = d.sample()
        return a.numpy(), d.log_prob(a).sum(-1).numpy(), self.value(obs).numpy()


def make_env(rank, seed, env_name="reservoir"):
    def thunk():
        env = ENVS[env_name]()
        env.reset(seed=seed + rank)
        return env
    return thunk


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    envs = gym.vector.AsyncVectorEnv([make_env(i, args.seed, args.env) for i in range(args.num_envs)],
                                     context="spawn")
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    model = ActorCritic(obs_dim, act_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    N, T = args.num_envs, args.rollout_steps
    obs_buf = np.zeros((T, N, obs_dim), np.float32)
    act_buf = np.zeros((T, N, act_dim), np.float32)
    logp_buf = np.zeros((T, N), np.float32)
    rew_buf = np.zeros((T, N), np.float32)
    done_buf = np.zeros((T, N), np.float32)
    val_buf = np.zeros((T, N), np.float32)

    obs, _ = envs.reset(seed=args.seed)
    ep_ret = np.zeros(N)
    finished_returns = []
    n_updates = args.total_steps // (N * T)
    t_start = time.time()

    for update in range(1, n_updates + 1):
        # ---- collect a rollout of T steps from each env
        for t in range(T):
            a, logp, v = model.act(obs)
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, a, logp, v
            obs, r, term, trunc, _ = envs.step(np.clip(a, -1, 1))
            done = term | trunc
            rew_buf[t], done_buf[t] = r, done
            ep_ret += r
            for i in np.where(done)[0]:
                finished_returns.append(ep_ret[i])
                ep_ret[i] = 0.0
            # AsyncVectorEnv autoresets on the step after done ("next-step"
            # mode): that step's action is ignored and its obs is the reset obs.
            # Those transitions are masked out below via done_buf of t-1.

        # ---- GAE advantages
        with torch.no_grad():
            last_val = model.value(torch.as_tensor(obs)).numpy()
        adv = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * next_val * nonterminal - val_buf[t]
            gae = delta + args.gamma * args.lam * nonterminal * gae
            adv[t] = gae
        ret = adv + val_buf

        # Drop the dummy autoreset transitions (the step right after a done).
        valid = np.ones((T, N), bool)
        valid[1:] = done_buf[:-1] == 0
        b_obs = torch.as_tensor(obs_buf[valid])
        b_act = torch.as_tensor(act_buf[valid])
        b_logp = torch.as_tensor(logp_buf[valid])
        b_adv = torch.as_tensor(adv[valid])
        b_ret = torch.as_tensor(ret[valid])
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ---- PPO update
        n = len(b_obs)
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, args.minibatch):
                idx = perm[start:start + args.minibatch]
                d = model.dist(b_obs[idx])
                logp = d.log_prob(b_act[idx]).sum(-1)
                ratio = (logp - b_logp[idx]).exp()
                pg_loss = -torch.min(ratio * b_adv[idx],
                                     ratio.clamp(1 - args.clip, 1 + args.clip) * b_adv[idx]).mean()
                v_loss = 0.5 * (model.value(b_obs[idx]) - b_ret[idx]).pow(2).mean()
                entropy = d.entropy().sum(-1).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        recent = finished_returns[-20:]
        steps = update * N * T
        print(f"update {update:3d}/{n_updates}  steps {steps:7d}  "
              f"episodes {len(finished_returns):4d}  "
              f"mean return (last 20) {np.mean(recent) if recent else float('nan'):8.3f}  "
              f"std {model.log_std.exp().mean().item():.3f}  "
              f"{steps / (time.time() - t_start):.0f} steps/s", flush=True)

    envs.close()
    torch.save(model.state_dict(), args.out)
    print(f"saved policy to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(ENVS), default="reservoir")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ppo_reservoir.pt")
    train(p.parse_args())
