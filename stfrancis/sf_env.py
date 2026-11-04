"""
St. Francis bioretention cell (half footprint, 30.2 mm), perfect forecast:
RL control of the underdrain valve as a correction on the tuned forecast rule.

Simulator: the compiled OHQ kernel of the model with the valve replaced by a
controlled orifice (kernels/rl). Every control interval (1 h) the opening is
set (kernel parameter 'Opening') and the kernel runs to the end of the interval.

Action a in [-1, 1]:  opening = clip(u_rule + a * u_scale, 0, u_max), where
u_rule = sqrt(Ihat/S) if Ihat > I0 else 0 is the tuned rule (a = 0 is the rule).

Reward: minus the increase of J = Excess/Excess_passive + R * Age_Load/Age_passive
over the interval (times reward_scale), so the episode return is -J of the window.
Both objectives are integrals (additive): Excess = int max(Q - Q_lim, 0) dt.

Episodes: windows of episode_days from random starts in the training window
(2020-2023; 2020-2022 with split="train"; 2021-2023 with split="ho2020");
the state at the start is a snapshot from a continuous run of the hourly rule
(snapshots.py), so every episode starts from a consistent, realistic state.
"""
import json
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from common import HERE, KERNELS, SPLITS, SPLIT_DEFS, Kernel, indicator, rule_opening, out_index

RES = os.path.join(HERE, "results")
# perfect-forecast windows (start, end) in days ahead, as mean rain intensity
FORECAST_WINDOWS = [(0, 0.125), (0.125, 0.5), (0.5, 1), (1, 3), (3, 7)]
# observation features (name -> number of values); the default is all of them
FEATURES = {"pond": 1, "agg": 1, "eng": 1, "cb": 1, "outflow": 1, "weir": 1, "u_rule": 1,
            "rain_0_3h": 1, "rain_3_12h": 1, "rain_12_24h": 1, "rain_1_3d": 1, "rain_3_7d": 1,
            "rain_past_24h": 1, "ihat": 1, "season": 2}


class RainWindows:
    def __init__(self, t, c):
        self.t = t
        self.F = np.concatenate([[0.0], np.cumsum(c[1:] * np.diff(t))])   # c[i] holds over (t[i-1], t[i]]

    def mean(self, a, b):
        return float((np.interp(b, self.t, self.F) - np.interp(a, self.t, self.F)) / (b - a))


class StFrancisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, R=0.0, episode_days=30, dt=1 / 24, u_scale=0.004, u_max=0.03,
                 reward_scale=100.0, residual=True, split="full", features=None):
        super().__init__()
        self.R, self.T, self.dt = float(R), float(episode_days), float(dt)
        self.n_steps = int(round(self.T / self.dt))
        self.u_scale, self.u_max, self.reward_scale, self.residual = u_scale, u_max, reward_scale, residual
        sfx = SPLIT_DEFS[split]["sfx"]
        tag = f"R{self.R:g}{sfx}"
        rule = json.load(open(os.path.join(RES, f"rule_{tag}.json")))["params"]
        self.S, self.I0, self.lam = rule["S"], rule["I0"], 1.0 / rule["tau_k"]
        ref = json.load(open(os.path.join(RES, f"passive{sfx}.json")))
        self.w = np.array([1.0 / ref["Excess"], self.R / ref["Age_Load"]])
        snap = np.load(os.path.join(RES, f"snapshots_{tag}.npz"))
        self.snap_t = snap["t"]
        self.snap = {k: snap[k] for k in ("storage", "mass", "limited", "limit_factor")}

        self.k = Kernel(KERNELS["rl"])
        self.ix = out_index(self.k)
        self.obj = [self.k.objectives.index("Excess"), self.k.objectives.index("Age_Load")]
        self.ind = indicator()
        self.rain = RainWindows(self.ind.t, self.ind.c)
        lo, hi = SPLITS[sfx]          # episodes lie inside the training window
        self.starts = self.snap_t[(self.snap_t >= lo) & (self.snap_t <= hi - self.T)]

        self.features = list(features) if features is not None else list(FEATURES)
        unknown = set(self.features) - set(FEATURES)
        if unknown or not self.features:
            raise ValueError(f"features must be a non-empty subset of {list(FEATURES)} (got {unknown or 'none'})")
        n_obs = sum(FEATURES[f] for f in self.features)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------------
    def rule_u(self, t):
        return rule_opening(self.ind.ekr(t, self.lam), self.S, self.I0)

    def opening(self, a):
        base = self.u_rule if self.residual else 0.5 * self.u_max
        return float(np.clip(base + float(a) * (self.u_scale if self.residual else 0.5 * self.u_max),
                             0.0, self.u_max))

    def all_features(self):
        """Every observation feature at the current state (scaled to order 1)."""
        o, ix = self.k.outputs(), self.ix
        x = {"pond": o[ix["pond"]] / 50.0, "agg": o[ix["agg"]] / 30.0, "eng": o[ix["eng"]] / 30.0,
             "cb": o[ix["cb"]] / 1.0, "outflow": np.log1p(max(o[ix["outflow"]], 0.0) / 10.0),
             "weir": np.log1p(max(o[ix["weir"]], 0.0) / 10.0), "u_rule": self.u_rule / self.u_scale,
             "rain_past_24h": np.log1p(max(self.rain.mean(self.t - 1, self.t), 0.0) / 0.01),
             "ihat": np.log1p(self.ind.ekr(self.t, self.lam) / 0.01)}
        for name, (a, b) in zip(("rain_0_3h", "rain_3_12h", "rain_12_24h", "rain_1_3d", "rain_3_7d"), FORECAST_WINDOWS):
            x[name] = np.log1p(max(self.rain.mean(self.t + a, self.t + b), 0.0) / 0.01)
        ph = 2 * np.pi * (self.t - 43831.0) / 365.25
        x["season"] = (np.sin(ph), np.cos(ph))
        return x

    def _obs(self):
        x = self.all_features()
        out = []
        for f in self.features:
            out += list(x[f]) if FEATURES[f] > 1 else [x[f]]
        return np.array(out, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "start" in options:
            t0 = float(options["start"])
        else:
            t0 = float(self.starts[self.np_random.integers(len(self.starts))])
        j = int(np.argmin(np.abs(self.snap_t - t0)))
        t0 = float(self.snap_t[j])
        self.k.initialize(t0)
        self.k.import_state(t0, {k: v[j].copy() for k, v in self.snap.items()})
        self.t0 = self.t = t0
        self.i = 0
        self.u_rule = self.rule_u(self.t)
        self.I = self.k.integral[self.obj].copy()
        return self._obs(), {"t": self.t}

    def step(self, action):
        u = self.opening(np.asarray(action).reshape(-1)[0])
        self.k.set_parameters({"Opening": u})
        self.i += 1
        self.t = self.t0 + self.i * self.dt
        ok = self.k.run_to(self.t)
        I = self.k.integral[self.obj].copy()
        dI, self.I = I - self.I, I
        info = {"t": self.t, "opening": u, "u_rule": self.u_rule, "dExcess": dI[0], "dAge": dI[1]}
        if not ok:
            info["solver_failed"] = True
            return np.zeros(self.observation_space.shape, np.float32), -10.0, True, False, info
        self.u_rule = self.rule_u(self.t)
        reward = -self.reward_scale * float(self.w @ dI)
        return self._obs(), reward, False, self.i >= self.n_steps, info


def zero_policy(env):
    """No correction: with residual=True this is the tuned rule itself."""
    return np.zeros(1, dtype=np.float32)
