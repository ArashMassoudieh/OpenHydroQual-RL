"""
Gymnasium environment: coordinated daily operation of two reservoirs in series
(a cascade), simulated with OpenHydroQual through openhydroqual_py.

    inflow A                       inflow B
       |                              |
  [Reservoir A] --outlet A--> [Reservoir B] --outlet B--> River --+--> flood point
       +-------spillway A------>      +-------spillway B---------> |   (Q_safe, demand)
                                                                    |
                                          tributary (uncontrolled) -+

Storms are regional: each one hits the large, slow upstream catchment (A), the
smaller local catchment (B) and a small, flashy tributary that joins below B.
The agent sets both releases every day. The tributary cannot be controlled, so
the agent must keep flood space where it will be needed, cut releases while
the tributary peaks, and use tributary water to meet the downstream demand.

OHQ model per decision step (one day):
    ReservoirA, ReservoirB : 'Reservoir' blocks with inflow time series
    River, SpillSinkB      : 'fixed_head' sinks; their Storage accumulates B's
                             released and spilled volume
    outletA  (A -> B), outletB (B -> River) : 'Reservoir_link_rule',
                             Q_min=0, Q_max=<action>, S_min=0, S_max=S_dead
    spillwayA (A -> B), spillwayB (B -> SpillSinkB) : 'Reservoir_link_rule'
                             ramping from 0 at S_cap to Q_spill_max at 1.05 S_cap
A's outflow goes into B, not into a sink, so A's released and spilled volumes
are recovered from A's storage at every solver step: OHQ is backward Euler, so
the volume through a link over a step is Q(S at the end of the step) * dt.
The tributary is added to the river flow in Python.

Reward per day (same terms and weights as the single-reservoir problem):
    -(deficit/D)^2 - w_flood*excess/Q_safe - w_spill*(spill_A+spill_B)/Q_safe
with river flow = B release + B spill + tributary, and a terminal penalty for
ending the season below half full in either reservoir. Units: m^3 and days.
"""
import os
import shutil
import tempfile

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ohq_reservoir_env import (OHQBackend, _import_ohq, new_ohq_system,
                               read_outputs_at)


def ramp(S, S_lo, S_hi, Q_hi):
    """Flow of a Reservoir_link_rule with Q_min = 0 (see main_components.json)."""
    return Q_hi * np.clip((S - S_lo) / (S_hi - S_lo), 0.0, 1.0)


class CascadeBackend:
    """Builds and solves the two-reservoir OHQ model for one control interval."""

    def __init__(self, cap_A, cap_B, S_dead, spill_max_A, spill_max_B,
                 solver_dt=0.05, workdir=None):
        self.ohq = _import_ohq()
        self.cap_A, self.cap_B, self.S_dead = cap_A, cap_B, S_dead
        self.spill_max_A, self.spill_max_B = spill_max_A, spill_max_B
        self.dt = solver_dt
        self.workdir = workdir or tempfile.mkdtemp(prefix="ohq_cascade_")
        os.makedirs(self.workdir, exist_ok=True)
        self.inflow_files = {k: os.path.join(self.workdir, f"inflow_{k}.txt") for k in "AB"}
        self.output_file = os.path.join(self.workdir, "output.txt")
        # Output time-stamping convention of the linked OHQLib build.
        self.rows_lag_one_step = OHQBackend(
            1e9, S_dead, 1.0, solver_dt, workdir=os.path.join(self.workdir, "stamping_check")
        ).rows_lag_one_step

    def set_inflow(self, t, q_A, q_B):
        for k, q in (("A", q_A), ("B", q_B)):
            np.savetxt(self.inflow_files[k], np.column_stack([t, q]), fmt="%.4f", delimiter=", ")

    def simulate_day(self, t0, S_A0, S_B0, Q_A, Q_B):
        """Run OHQ over [t0, t0+1]. Returns a dict of end storages and day volumes."""
        s = new_ohq_system(self.ohq, self.workdir, t0, t0 + 1, self.dt, self.output_file)
        s.create_block("ReservoirA", "Reservoir", {"Storage": f"{S_A0}", "inflow": self.inflow_files["A"]})
        s.create_block("ReservoirB", "Reservoir", {"Storage": f"{S_B0}", "inflow": self.inflow_files["B"]})
        s.create_block("River", "fixed_head", {"head": "0", "Storage": "0"})
        s.create_block("SpillSinkB", "fixed_head", {"head": "0", "Storage": "0"})
        outlet = lambda Q: {"Q_min": "0", "Q_max": f"{Q}", "S_min": "0", "S_max": f"{self.S_dead}"}
        spill = lambda cap, Qs: {"Q_min": "0", "Q_max": f"{Qs}", "S_min": f"{cap}", "S_max": f"{1.05 * cap}"}
        s.create_link("outletA", "Reservoir_link_rule", "ReservoirA", "ReservoirB", outlet(Q_A))
        s.create_link("spillwayA", "Reservoir_link_rule", "ReservoirA", "ReservoirB",
                      spill(self.cap_A, self.spill_max_A))
        s.create_link("outletB", "Reservoir_link_rule", "ReservoirB", "River", outlet(Q_B))
        s.create_link("spillwayB", "Reservoir_link_rule", "ReservoirB", "SpillSinkB",
                      spill(self.cap_B, self.spill_max_B))
        s.finalize()
        if not s.solve(True, False):
            raise RuntimeError(f"OHQ solve failed for day starting at t={t0}")
        s.write_outputs()

        S_A1, S_B1, rel_B, spill_B = read_outputs_at(
            self.output_file, t0 + 1,
            ["ReservoirA_Storage", "ReservoirB_Storage", "River_Storage", "SpillSinkB_Storage"],
            self.rows_lag_one_step)
        # A's link volumes: sum of Q(S_end of step) * dt over the day's steps.
        S_A_steps = self._end_of_step_series(t0, "ReservoirA_Storage")
        rel_A = float(np.sum(ramp(S_A_steps, 0.0, self.S_dead, Q_A)) * self.dt)
        spill_A = float(np.sum(ramp(S_A_steps, self.cap_A, 1.05 * self.cap_A, self.spill_max_A)) * self.dt)
        return dict(S_A=S_A1, S_B=S_B1, rel_A=rel_A, spill_A=spill_A, rel_B=rel_B, spill_B=spill_B)

    def _end_of_step_series(self, t0, name):
        """Storage at the end of each solver step in (t0, t0+1]."""
        with open(self.output_file) as f:
            header = [h.strip() for h in f.readline().split(",")]
        data = np.loadtxt(self.output_file, delimiter=",", skiprows=1, ndmin=2)
        c = header.index(name)
        t, v = data[:, c - 1], data[:, c]
        if self.rows_lag_one_step:
            t, v = t[1:], v[:-1]
        grid = t0 + self.dt * np.arange(1, int(round(1 / self.dt)) + 1)
        return np.interp(grid, t, v)

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


def storm_hydrograph(t, t_onset, peak, t_rise, k):
    """Linear rise to `peak` over t_rise, then exponential recession at rate k."""
    tau = t - t_onset
    h = np.zeros_like(t)
    rise = (tau >= 0) & (tau < t_rise)
    fall = tau >= t_rise
    h[rise] = peak * tau[rise] / t_rise
    h[fall] = peak * np.exp(-k * (tau[fall] - t_rise))
    return h


class CascadeEnv(gym.Env):
    metadata = {"render_modes": []}
    STREAMS = ("A", "B", "trib")

    def __init__(self, episode_days=90, cap_A=60_000.0, cap_B=30_000.0, S_dead=200.0,
                 Q_out_A=6_000.0, Q_out_B=8_000.0, Q_safe=4_000.0,
                 spill_max_A=60_000.0, spill_max_B=40_000.0,
                 demand_mean=2_000.0, base_flow=(500.0, 200.0, 300.0), storm_prob=0.05,
                 storm_peaks=(5_000.0, 3_000.0, 4_000.0),
                 forecast_noise=0.2, w_flood=2.0, w_spill=1.0, w_terminal=5.0,
                 solver_dt=0.05, backend=None):
        super().__init__()
        self.episode_days = episode_days
        self.cap_A, self.cap_B = cap_A, cap_B
        self.Q_out_A, self.Q_out_B, self.Q_safe = Q_out_A, Q_out_B, Q_safe
        self.demand_mean, self.base_flow, self.storm_prob = demand_mean, base_flow, storm_prob
        self.storm_peaks = storm_peaks
        self.forecast_noise = forecast_noise
        self.w_flood, self.w_spill, self.w_terminal = w_flood, w_spill, w_terminal
        self.dt = solver_dt
        self.backend = backend or CascadeBackend(cap_A, cap_B, S_dead, spill_max_A, spill_max_B, solver_dt)
        # obs: S_A/cap_A, S_B/cap_B, current inflow of A, B, tributary,
        #      1/2/3-day forecasts of each (9), demand, fraction of season left
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
        # actions in [-1, 1] -> releases in [0, Q_out_A] and [0, Q_out_B]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------ hydrology/forcing
    def _generate_forcing(self):
        rng = self.np_random
        n = self.episode_days + 4
        t = np.arange(0, n + 1e-9, self.dt)
        beta = rng.uniform(0.7, 1.3)
        q = {k: np.full_like(t, b * beta) for k, b in zip(self.STREAMS, self.base_flow)}
        # (peak scale, delay, rise time, recession-rate range) per catchment
        P_A, P_B, P_trib = self.storm_peaks
        response = {"A": (P_A, 0.5, 1.0, (0.2, 0.4)),
                    "B": (P_B, 0.0, 0.5, (0.4, 0.7)),
                    "trib": (P_trib, 0.0, 0.25, (0.7, 1.2))}
        for day in range(n):
            if rng.random() < self.storm_prob:
                onset = day + rng.random()
                intensity = rng.uniform(0.5, 1.5)
                for k, (P, delay, t_r, (k_lo, k_hi)) in response.items():
                    q[k] += storm_hydrograph(t, onset + delay, P * intensity * rng.uniform(0.8, 1.2),
                                             t_r, rng.uniform(k_lo, k_hi))
        self.inflow_t, self.inflow = t, q
        self.backend.set_inflow(t, q["A"], q["B"])

        phase = rng.uniform(0, 2 * np.pi)
        self.demand = self.demand_mean * (1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / self.episode_days + phase))
        self.daily = {k: np.array([v[(t >= d) & (t < d + 1)].mean() for d in range(n)]) for k, v in q.items()}
        lead_sd = self.forecast_noise * np.sqrt([1, 2, 3])
        self.forecast = {k: np.array([[self.daily[k][min(d + j, n - 1)] for j in (1, 2, 3)] for d in range(n)])
                            * np.exp(rng.normal(0, 1, size=(n, 3)) * lead_sd)
                         for k in self.STREAMS}
        # Tributary volume reaching the flood point each day (exact trapezoid).
        self.trib_volume = np.array([np.trapezoid(q["trib"][(t >= d - 1e-9) & (t <= d + 1 + 1e-9)], dx=self.dt)
                                     for d in range(n)])

    def current_inflow(self, k):
        return self.inflow[k][np.searchsorted(self.inflow_t, self.day)]

    def _obs(self):
        d = self.day
        return np.array([
            self.S_A / self.cap_A, self.S_B / self.cap_B,
            *(self.current_inflow(k) / self.Q_safe for k in self.STREAMS),
            *np.concatenate([self.forecast[k][d] for k in self.STREAMS]) / self.Q_safe,
            self.demand[d] / self.Q_safe,
            1.0 - d / self.episode_days,
        ], dtype=np.float32)

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_forcing()
        self.day = 0
        self.S_A = float(self.np_random.uniform(0.3, 0.8) * self.cap_A)
        self.S_B = float(self.np_random.uniform(0.3, 0.8) * self.cap_B)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        Q_A = 0.5 * (a[0] + 1.0) * self.Q_out_A
        Q_B = 0.5 * (a[1] + 1.0) * self.Q_out_B
        d = self.day
        r = self.backend.simulate_day(float(d), self.S_A, self.S_B, Q_A, Q_B)

        D = self.demand[d]
        river = r["rel_B"] + r["spill_B"] + self.trib_volume[d]
        deficit = max(0.0, D - river)
        r_supply = -(deficit / D) ** 2
        r_flood = -self.w_flood * max(0.0, river - self.Q_safe) / self.Q_safe
        r_spill = -self.w_spill * (r["spill_A"] + r["spill_B"]) / self.Q_safe
        reward = r_supply + r_flood + r_spill

        self.S_A, self.S_B = max(r["S_A"], 0.0), max(r["S_B"], 0.0)
        self.day += 1
        terminated = self.day >= self.episode_days
        if terminated:
            reward -= self.w_terminal * 0.5 * (max(0.0, 0.5 * self.cap_A - self.S_A) / self.cap_A
                                               + max(0.0, 0.5 * self.cap_B - self.S_B) / self.cap_B)
        info = dict(day=d, S_A=self.S_A, S_B=self.S_B, Q_A=Q_A, Q_B=Q_B,
                    inflow_A=self.daily["A"][d], inflow_B=self.daily["B"][d], trib=self.trib_volume[d],
                    river=river, demand=D, deficit=deficit, r_supply=r_supply, r_flood=r_flood,
                    r_spill=r_spill, **{k: r[k] for k in ("rel_A", "spill_A", "rel_B", "spill_B")})
        return self._obs(), float(reward), terminated, False, info

    def close(self):
        self.backend.close()


def cascade_rule_policy(env, eta_A=0.6, eta_B=0.5, drawdown_days=1.0):
    """Baseline: coordinated two-threshold rule (flows in m3/d, volumes in m3).
    B releases the part of the demand the tributary does not supply; above
    eta_B*cap_B it also releases the excess storage over drawdown_days, up to the
    river capacity the tributary leaves. A backfills B's demand release (net of
    B's own inflow) while B is below its target; above eta_A*cap_A it also
    releases its excess over drawdown_days, as far as B has room below eta_B."""
    trib, in_B = env.current_inflow("trib"), env.current_inflow("B")
    D = env.demand[env.day]
    Q_supply = max(0.0, D - trib)
    Q_B = Q_supply
    excess_B = env.S_B - eta_B * env.cap_B                     # m3
    if excess_B > 0:
        Q_B = max(Q_B, min(env.Q_safe - trib, Q_B + excess_B / drawdown_days))
    Q_A = max(0.0, Q_supply - in_B) if excess_B < 0 else 0.0
    excess_A = env.S_A - eta_A * env.cap_A                     # m3
    if excess_A > 0:
        room_B = max(0.0, -excess_B) / drawdown_days + Q_B - in_B   # m3/d B can take
        Q_A += min(excess_A / drawdown_days, max(0.0, room_B))
    Q_A, Q_B = min(Q_A, env.Q_out_A), min(max(Q_B, 0.0), env.Q_out_B)
    return np.array([2 * Q_A / env.Q_out_A - 1, 2 * Q_B / env.Q_out_B - 1], dtype=np.float32)
