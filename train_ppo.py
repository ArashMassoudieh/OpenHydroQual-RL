"""
PPO (clipped surrogate, GAE) in plain PyTorch for OHQReservoirEnv.

Each worker process owns its own OHQ instance and working folder, so rollouts
run in parallel through gymnasium's AsyncVectorEnv.

    python train_ppo.py --total-steps 200000 --num-envs 8
"""
import argparse
import importlib
import json
import os
import time

# One tiny OHQ model per worker process: keep every library single-threaded,
# otherwise N workers x M threads oversubscribe the CPU (set before imports).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp

from ohq_reservoir_env import OHQReservoirEnv
from ohq_cascade_env import CascadeEnv
from ohq_pond_env import PondEnv
from ohq_eflows_env import EflowsEnv

ENVS = {"reservoir": OHQReservoirEnv, "cascade": CascadeEnv, "pond": PondEnv, "eflows": EflowsEnv}


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


def rollout_worker(remote, env_name, seed, hidden, env_kwargs=None):
    """Owns one environment. On each request it loads the current policy weights
    and collects T steps on its own, so a slow simulation step delays only this
    worker (with lockstep vectorized environments every worker waited for the
    slowest one, which made rare multi-second OHQ steps stall all of them)."""
    torch.set_num_threads(1)
    env = ENVS[env_name](**(env_kwargs or {}))
    obs, _ = env.reset(seed=seed)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden)
    ep_ret = 0.0
    while True:
        cmd, data = remote.recv()
        if cmd == "close":
            env.close()
            remote.close()
            return
        if cmd == "bc":
            # Behavior cloning data: run a rule-based policy and record what it does.
            rule_path, kwargs, T = data
            module, fn = rule_path.split(":")
            rule = getattr(importlib.import_module(module), fn)
            out = {k: [] for k in ("obs", "act", "rew", "done")}
            for _ in range(T):
                a = np.asarray(rule(env, **kwargs), dtype=np.float32).reshape(-1)
                next_obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                for k, x in zip(out, (obs, a, r, float(done))):
                    out[k].append(x)
                obs = env.reset()[0] if done else next_obs
            remote.send({k: np.array(v, dtype=np.float32) for k, v in out.items()})
            continue
        state_dict, T = data
        model.load_state_dict(state_dict)
        out = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done", "boot")}
        finished = []
        for _ in range(T):
            a, logp, v = model.act(obs[None])
            next_obs, r, term, trunc, _ = env.step(np.clip(a[0], -1, 1))
            done = term or trunc
            # A time limit is not a terminal state: the return continues past it,
            # so record the critic's value of the last state to bootstrap from.
            boot = 0.0
            if trunc and not term:
                with torch.no_grad():
                    boot = float(model.value(torch.as_tensor(next_obs[None], dtype=torch.float32))[0])
            for k, x in zip(out, (obs, a[0], logp[0], v[0], r, float(done), boot)):
                out[k].append(x)
            ep_ret += r
            if done:
                finished.append(ep_ret)
                ep_ret = 0.0
                next_obs, _ = env.reset()
            obs = next_obs
        with torch.no_grad():
            last_val = float(model.value(torch.as_tensor(obs[None], dtype=torch.float32))[0])
        remote.send(({k: np.array(v, dtype=np.float32) for k, v in out.items()}, last_val, finished))


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        b_mean, b_var, b_count = x.mean(), x.var(), x.size
        delta, tot = b_mean - self.mean, self.count + b_count
        self.mean += delta * b_count / tot
        self.var = (self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / tot) / tot
        self.count = tot


def clip_rewards(x, c):
    """Clip normalized rewards to [-c, c]; c <= 0 disables clipping."""
    return np.clip(x, -c, c) if c > 0 else x


def discounted_returns(rew, done, gamma):
    """Monte Carlo returns of one worker's sequence (0 after the last record)."""
    out, g = np.zeros_like(rew), 0.0
    for t in reversed(range(len(rew))):
        g = rew[t] + gamma * g * (1.0 - done[t])
        out[t] = g
    return out


def behavior_cloning(model, remotes, args, ret_rms):
    """Fit the actor to imitate a rule-based policy and the critic to its returns,
    so PPO starts from (at least) the rule's performance."""
    kwargs = json.loads(args.bc_kwargs)
    for r_ in remotes:
        r_.send(("bc", (args.bc_rule, kwargs, args.bc_steps)))
    batches = [r_.recv() for r_ in remotes]
    obs = torch.as_tensor(np.concatenate([b["obs"] for b in batches]))
    act = torch.as_tensor(np.clip(np.concatenate([b["act"] for b in batches]), -1, 1))
    rew = [b["rew"] for b in batches]
    if args.normalize_rewards:              # same scaling PPO will use
        for b, rw in zip(batches, rew):
            acc, accs = 0.0, []
            for x, d in zip(rw, b["done"]):
                acc = acc * args.gamma + x
                accs.append(acc)
                acc *= 1.0 - d
            ret_rms.update(accs)
        rew = [clip_rewards(rw / np.sqrt(ret_rms.var + 1e-8), args.reward_clip) for rw in rew]
    ret = torch.as_tensor(np.concatenate([discounted_returns(rw, b["done"], args.gamma)
                                          for rw, b in zip(rew, batches)]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(obs)
    for epoch in range(args.bc_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, 256):
            idx = perm[start:start + 256]
            loss = (model.actor(obs[idx]) - act[idx]).pow(2).mean() + 0.5 * (model.value(obs[idx]) - ret[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        a_err = (model.actor(obs) - act).abs().mean().item()
    with torch.no_grad():
        model.log_std.fill_(args.bc_log_std)
    print(f"behavior cloning: {n} records from {args.bc_rule}, mean |action error| {a_err:.3f}, "
          f"exploration std reset to {np.exp(args.bc_log_std):.2f}", flush=True)


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env_kwargs = json.loads(args.env_kwargs)
    probe = ENVS[args.env](**env_kwargs)
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.shape[0]
    probe.close()
    model = ActorCritic(obs_dim, act_dim)
    ctx = mp.get_context("spawn")
    remotes, procs = [], []
    for i in range(args.num_envs):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=rollout_worker, args=(child, args.env, args.seed + i, 64, env_kwargs), daemon=True)
        p.start()
        remotes.append(parent)
        procs.append(p)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    N, T = args.num_envs, args.rollout_steps
    obs_buf = np.zeros((T, N, obs_dim), np.float32)
    act_buf = np.zeros((T, N, act_dim), np.float32)
    logp_buf = np.zeros((T, N), np.float32)
    rew_buf = np.zeros((T, N), np.float32)
    done_buf = np.zeros((T, N), np.float32)
    val_buf = np.zeros((T, N), np.float32)
    boot_buf = np.zeros((T, N), np.float32)

    finished_returns = []
    ret_rms, ret_acc = RunningMeanStd(), np.zeros(N)
    if args.bc_rule:
        behavior_cloning(model, remotes, args, ret_rms)
    ckpt = lambda u: args.out.replace(".pt", f"_u{u:04d}.pt")
    if args.save_every:
        torch.save(model.state_dict(), ckpt(0))      # the starting policy (e.g. the cloned rule)
    n_updates = args.total_steps // (N * T)
    t_start = time.time()

    for update in range(1, n_updates + 1):
        # ---- collect a rollout of T steps from each worker (in parallel)
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for r_ in remotes:
            r_.send(("rollout", (weights, T)))
        last_val = np.zeros(N, np.float32)
        for i, r_ in enumerate(remotes):
            batch, last_val[i], finished = r_.recv()
            obs_buf[:, i], act_buf[:, i], logp_buf[:, i] = batch["obs"], batch["act"], batch["logp"]
            val_buf[:, i], rew_buf[:, i], done_buf[:, i] = batch["val"], batch["rew"], batch["done"]
            boot_buf[:, i] = batch["boot"]
            finished_returns.extend(finished)

        raw_rew = rew_buf.copy()
        if args.normalize_rewards:
            # Scale rewards by the running std of the discounted return (and clip):
            # rare, very large penalties (pond overflow) otherwise dominate the critic.
            accs = np.zeros((T, N))
            for t in range(T):
                ret_acc = ret_acc * args.gamma + rew_buf[t]
                accs[t] = ret_acc
                ret_acc = ret_acc * (1.0 - done_buf[t])
            ret_rms.update(accs)
            rew_buf = clip_rewards(rew_buf / np.sqrt(ret_rms.var + 1e-8), args.reward_clip).astype(np.float32)
        # Truncated episodes: add the discounted value of the state after the
        # time limit (already in the critic's, i.e. normalized, units).
        rew_buf = rew_buf + args.gamma * boot_buf

        # ---- GAE advantages
        adv = np.zeros((T, N), np.float32)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * next_val * nonterminal - val_buf[t]
            gae = delta + args.gamma * args.lam * nonterminal * gae
            adv[t] = gae
        ret = adv + val_buf

        valid = np.ones((T, N), bool)   # workers reset immediately: every record is real
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

        if args.save_every and (update % args.save_every == 0 or update == n_updates):
            torch.save(model.state_dict(), ckpt(update))
        recent = finished_returns[-20:]
        steps = update * N * T
        print(f"update {update:3d}/{n_updates}  steps {steps:7d}  "
              f"episodes {len(finished_returns):4d}  "
              f"mean return (last 20) {np.mean(recent) if recent else float('nan'):8.3f}  "
              f"std {model.log_std.exp().mean().item():.3f}  "
              f"reward/step {raw_rew[valid].mean():8.4f}  "
              f"{steps / (time.time() - t_start):.0f} steps/s", flush=True)

    for r_ in remotes:
        r_.send(("close", None))
    for p in procs:
        p.join(timeout=30)
    torch.save(model.state_dict(), args.out)
    print(f"saved policy to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=sorted(ENVS), default="reservoir")
    p.add_argument("--env-kwargs", default="{}", help='JSON keyword arguments for the env, e.g. \'{"scenario": "multi"}\'')
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
    p.add_argument("--normalize-rewards", action="store_true",
                   help="scale rewards by the running std of the discounted return (see --reward-clip)")
    p.add_argument("--reward-clip", type=float, default=10.0,
                   help="clip normalized rewards to +-this value; <= 0 disables clipping")
    p.add_argument("--save-every", type=int, default=0, help="save a checkpoint every N updates (0: off)")
    p.add_argument("--bc-rule", default="", help="module:function of a rule policy to imitate before PPO")
    p.add_argument("--bc-kwargs", default="{}", help="JSON keyword arguments for the rule policy")
    p.add_argument("--bc-steps", type=int, default=2000, help="rule steps per worker for behavior cloning")
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--bc-log-std", type=float, default=-1.2, help="exploration log-std after cloning")
    train(p.parse_args())
