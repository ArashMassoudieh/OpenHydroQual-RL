"""
Gymnasium environment: daily release control of a single reservoir simulated
with OpenHydroQual (OHQ) through the openhydroqual_py Python bindings.

Toy problem
-----------
A water-supply reservoir receives a low base flow plus random storm
hydrographs. Every day the operator (the RL agent) picks a release rate.

    Inflow(t) --> [ Reservoir ] --outlet (controlled release)--> River
                        |
                        +--spillway (uncontrolled, above S_cap)--> SpillSink

OHQ model built for every decision step (one day):
    Reservoir   : 'Reservoir' block (Reservoir_operation.json), inflow = time series
    River       : 'fixed_head' sink; its Storage accumulates the released volume
    SpillSink   : 'fixed_head' sink; its Storage accumulates the spilled volume
    outlet      : 'Reservoir_link_rule' with Q_min=0, Q_max=<action>, S_min=0,
                  S_max=S_dead -> releases <action> m3/day, tapering to zero
                  as the reservoir empties (storage can never go negative)
    spillway    : 'Reservoir_link_rule' ramping from 0 at S_cap to Q_spill_max
                  at 1.05*S_cap (an emergency overflow weir)

Objectives (reward per day):
    - meet the downstream demand D(t)                 -> -(deficit/D)^2
    - keep the river flow below its safe capacity     -> -w_flood * excess/Q_safe
    - avoid uncontrolled spill (dam safety)           -> -w_spill * spill/Q_safe
    - end the season with water in storage            -> terminal penalty

The simulator is a pluggable backend with one method,
    simulate_day(t0, storage0, release) -> (storage, released_vol, spilled_vol)
so the interpreted OHQ library (OHQBackend) can later be swapped for a
codegen-compiled kernel with its own Python binding without touching the env.

Units: m^3 and days.
"""
import os
import sys
import shutil
import tempfile

import numpy as np
import gymnasium as gym
from gymnasium import spaces

OHQ_PY_PATH = os.environ.get(
    "OHQ_PY_PATH",
    "/home/arash/Projects/OpenHydroQual/PythonBindings/build/Desktop_Qt_6_8_2-Release/")
OHQ_RESOURCES = os.environ.get(
    "OHQ_RESOURCES", "/home/arash/Projects/OpenHydroQual/resources/")


def _import_ohq():
    # Imported lazily so that the parent process of a vectorized env never
    # creates a QCoreApplication before starting workers.
    if OHQ_PY_PATH not in sys.path:
        sys.path.append(OHQ_PY_PATH)
    import openhydroqual_py
    return openhydroqual_py


def new_ohq_system(ohq, workdir, t0, t1, dt, output_file, templates=("Reservoir_operation.json",)):
    """An OHQ System with the main components plus `templates`, set up to run
    [t0, t1] with a fixed step dt, single-threaded, writing to output_file."""
    R = OHQ_RESOURCES
    s = ohq.System()
    s.set_default_template_path(R)
    s.set_working_folder(workdir.rstrip("/") + "/")
    s.set_silent(True)
    s.load_template(R + "main_components.json")
    for tmpl in templates:
        s.add_template(R + tmpl)
    settings = {
        "simulation_start_time": f"{t0}",
        "simulation_end_time": f"{t1}",
        "initial_time_step": f"{dt}",
        # Keep the step at dt0 so the day is resolved finely (OHQLib builds before
        # 3e178fc also overshot t1 by the last, grown step).
        "max_timestep_increase_factor": "1",
        # One tiny model per worker: solver threads only oversubscribe the CPU.
        "n_threads": "1",
        "alloutputfile": output_file,
    }
    for k, v in settings.items():
        # OHQLib builds before commit 300da66 return False even on success.
        if not s.set_system_value(k, v):
            raise ValueError(f"OHQ rejected system setting {k}={v}")
    return s


def read_outputs_at(output_file, t_target, names, rows_lag_one_step):
    """Values of the named output columns at t_target, from an OHQ output file."""
    with open(output_file) as f:
        header = [h.strip() for h in f.readline().split(",")]
    data = np.loadtxt(output_file, delimiter=",", skiprows=1, ndmin=2)
    col = {name: i for i, name in enumerate(header) if name != "t"}
    t = data[:, col[names[0]] - 1]
    if rows_lag_one_step:
        # Row k holds the state at the end of step k, but is stamped with
        # the start time of step k; the end time is the next row's stamp.
        t, data = t[1:], data[:-1]
    return [float(np.interp(t_target, t, data[:, col[n]])) for n in names]


class OHQBackend:
    """Builds and solves the reservoir model with OHQLib for one control interval."""

    def __init__(self, S_cap, S_dead, Q_spill_max, solver_dt=0.05, workdir=None):
        self.ohq = _import_ohq()
        self.S_cap, self.S_dead, self.Q_spill_max = S_cap, S_dead, Q_spill_max
        self.dt = solver_dt
        # Each instance gets its own folder: OHQ writes output.txt, errors.txt,
        # ... into the working folder.
        self.workdir = workdir or tempfile.mkdtemp(prefix="ohq_rl_")
        os.makedirs(self.workdir, exist_ok=True)
        self.inflow_file = os.path.join(self.workdir, "inflow.txt")
        self.output_file = os.path.join(self.workdir, "output.txt")
        self.rows_lag_one_step = self._detect_output_stamping()

    def set_inflow(self, t, q):
        np.savetxt(self.inflow_file, np.column_stack([t, q]), fmt="%.4f", delimiter=", ")

    def simulate_day(self, t0, storage0, release):
        """Run OHQ from t0 to t0+1 day. Returns (storage, released_vol, spilled_vol)."""
        s = new_ohq_system(self.ohq, self.workdir, t0, t0 + 1, self.dt, self.output_file)
        s.create_block("Reservoir", "Reservoir",
                       {"Storage": f"{storage0}", "inflow": self.inflow_file})
        s.create_block("River", "fixed_head", {"head": "0", "Storage": "0"})
        s.create_block("SpillSink", "fixed_head", {"head": "0", "Storage": "0"})
        s.create_link("outlet", "Reservoir_link_rule", "Reservoir", "River",
                      {"Q_min": "0", "Q_max": f"{release}",
                       "S_min": "0", "S_max": f"{self.S_dead}"})
        s.create_link("spillway", "Reservoir_link_rule", "Reservoir", "SpillSink",
                      {"Q_min": "0", "Q_max": f"{self.Q_spill_max}",
                       "S_min": f"{self.S_cap}", "S_max": f"{1.05 * self.S_cap}"})
        s.finalize()
        if not s.solve(True, False):
            raise RuntimeError(f"OHQ solve failed for day starting at t={t0}")
        s.write_outputs()
        return self._read_state_at(t0 + 1)

    def _read_state_at(self, t_target):
        return read_outputs_at(self.output_file, t_target,
                               ["Reservoir_Storage", "River_Storage", "SpillSink_Storage"],
                               self.rows_lag_one_step)

    def _detect_output_stamping(self):
        """Older OHQLib builds stamp each output row with the time at the start
        of the step instead of the end. Run a 1-day test with a known analytical
        answer and find which convention this build uses."""
        Qin, Qout, S0 = 1000.0, 150.0, 1000.0
        self.set_inflow(np.array([0.0, 10.0]), np.array([Qin, Qin]))
        expected = S0 + (Qin - Qout) * 1.0
        for lagged in (False, True):
            self.rows_lag_one_step = lagged
            S_end, _, _ = self.simulate_day(0.0, S0, Qout)
            if abs(S_end - expected) < 1e-4 * expected:
                return lagged
        raise RuntimeError(f"OHQ mass-balance self-check failed: {S_end} vs {expected}")

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


class OHQReservoirEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_days=90, S_cap=50_000.0, S_dead=200.0,
                 Q_release_max=6_000.0, Q_safe=3_000.0, Q_spill_max=50_000.0,
                 demand_mean=1_500.0, base_inflow=600.0, storm_prob=0.06,
                 forecast_noise=0.2, w_flood=2.0, w_spill=1.0, w_terminal=5.0,
                 solver_dt=0.05, backend=None):
        super().__init__()
        self.episode_days = episode_days
        self.S_cap = S_cap
        self.Q_release_max, self.Q_safe = Q_release_max, Q_safe
        self.demand_mean, self.base_inflow, self.storm_prob = demand_mean, base_inflow, storm_prob
        self.forecast_noise = forecast_noise
        self.w_flood, self.w_spill, self.w_terminal = w_flood, w_spill, w_terminal
        self.dt = solver_dt
        self.backend = backend or OHQBackend(S_cap, S_dead, Q_spill_max, solver_dt)

        # obs: storage fraction, inflow now, 1/2/3-day inflow forecasts,
        #      demand today, fraction of season remaining
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        # action in [-1, 1] -> release in [0, Q_release_max]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------ hydrology/forcing
    def _generate_forcing(self):
        rng = self.np_random
        n = self.episode_days + 4  # extra days so forecasts exist at the end
        t = np.arange(0, n + 1e-9, self.dt)
        q = np.full_like(t, self.base_inflow) * rng.uniform(0.7, 1.3)
        for day in range(n):
            if rng.random() < self.storm_prob:
                t_start = day + rng.random()
                peak = rng.uniform(5_000, 15_000)
                t_rise = 0.5
                k = rng.uniform(0.3, 0.7)  # recession constant [1/day]
                tau = t - t_start
                rise = (tau >= 0) & (tau < t_rise)
                fall = tau >= t_rise
                q[rise] += peak * tau[rise] / t_rise
                q[fall] += peak * np.exp(-k * (tau[fall] - t_rise))
        self.inflow_t, self.inflow_q = t, q
        self.backend.set_inflow(t, q)

        phase = rng.uniform(0, 2 * np.pi)
        days = np.arange(n)
        self.demand = self.demand_mean * (1 + 0.3 * np.sin(2 * np.pi * days / self.episode_days + phase))

        # Daily-mean inflow and a noisy forecast of the next 3 days.
        self.daily_inflow = np.array([q[(t >= d) & (t < d + 1)].mean() for d in range(n)])
        noise = np.exp(rng.normal(0, self.forecast_noise, size=(n, 3)) * np.array([1, 2, 3]) ** 0.5)
        self.forecast = np.array([[self.daily_inflow[min(d + j, n - 1)] for j in (1, 2, 3)]
                                  for d in range(n)]) * noise

    def _obs(self):
        d = self.day
        i_now = np.searchsorted(self.inflow_t, d)
        return np.array([
            self.storage / self.S_cap,
            self.inflow_q[i_now] / self.Q_safe,
            *(self.forecast[d] / self.Q_safe),
            self.demand[d] / self.Q_safe,
            1.0 - d / self.episode_days,
        ], dtype=np.float32)

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_forcing()
        self.day = 0
        self.storage = float(self.np_random.uniform(0.3, 0.8) * self.S_cap)
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        release_rate = 0.5 * (a + 1.0) * self.Q_release_max
        d = self.day

        storage, released, spilled = self.backend.simulate_day(float(d), self.storage, release_rate)

        D = self.demand[d]
        deficit = max(0.0, D - released)
        river_flow = released + spilled  # daily volume == mean rate [m3/day]
        flood_excess = max(0.0, river_flow - self.Q_safe)
        r_supply = -(deficit / D) ** 2
        r_flood = -self.w_flood * flood_excess / self.Q_safe
        r_spill = -self.w_spill * spilled / self.Q_safe
        reward = r_supply + r_flood + r_spill

        self.storage = max(storage, 0.0)
        self.day += 1
        terminated = self.day >= self.episode_days
        if terminated:
            reward -= self.w_terminal * max(0.0, 0.5 * self.S_cap - self.storage) / self.S_cap

        info = dict(day=d, storage=self.storage, inflow=self.daily_inflow[d],
                    release=released, spill=spilled, demand=D, deficit=deficit,
                    r_supply=r_supply, r_flood=r_flood, r_spill=r_spill)
        return self._obs(), float(reward), terminated, False, info

    def close(self):
        self.backend.close()


def rule_curve_policy(env, threshold=0.8, drawdown_days=1.0):
    """Baseline: standard operating policy plus a flood-control rule.
    Release the demand D [m3/d]; above threshold*S_cap, also release the excess
    storage [m3] over drawdown_days, up to the safe channel capacity.
    Returns an action in [-1, 1]."""
    S, D = env.storage, env.demand[env.day]
    q = D
    if S > threshold * env.S_cap:
        q = min(env.Q_safe, D + (S - threshold * env.S_cap) / drawdown_days)
    return np.array([2.0 * q / env.Q_release_max - 1.0], dtype=np.float32)
