"""
Gymnasium environment: environmental-flow (e-flow) operation of a large
snowmelt reservoir, modeled after Lake McClure on the Merced River (CA) and
simulated with OpenHydroQual through its Python bindings.

    unimpaired inflow (CDEC MRC, daily)
              |
              v
       [ Reservoir ] --river outlet (action 1)--> River   (e-flows, floods)
          |    |
          |    +--canal outlet (action 2)-----> Canal   (irrigation supply)
          +--spillway (above S_cap)-----------> River

Every day the agent picks the river release and the fraction of today's
irrigation demand that is delivered to the canal. Both outlets pass through
the powerhouse (up to Q_turbine_max), so energy is a by-product of releases.

Four scenarios (after the "DRL for e-flows" outline by Rheinheimer et al.):
    minflow     regulatory minimum instream flow (stylized after the FERC
                licence + Davis-Grunsky schedule) enforced as a hard floor;
                reward = supply + flood
    prescribed  a fixed functional-flow hydrograph per year type enforced as
                a hard floor; reward = supply + flood
    ffreward    no floor; functional flows enter the reward instead;
                reward = supply + flood + functional flows
    multi       ffreward + hydropower
In every scenario the functional-flow score (functional_flows.py) is
reported, so all four can be compared on supply, floods, energy and e-flows.

Year type. The regulatory year type of the current water year is declared
from the forecast of its total unimpaired runoff (observed so far + forecast
of the rest, Bulletin-120 style, noisier early in the year), using the
dry/moderate/wet terciles of the training years. Daily e-flow targets follow
the type declared that day; the year's score uses the May 1 declaration.

Water years run Oct 1 - Sep 30 and Feb 29 is dropped, so each year has 365
daily steps (day 0 = Oct 1). An episode is `episode_years` water years; its
end is a time limit (truncation), not a terminal state, since the reservoir
keeps operating afterwards.

Units: flows in m3/s, storage in hm3 (10^6 m3) in the env; the OHQ model
works in m3 and m3/day.
"""
import os
import shutil
import tempfile

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ohq_reservoir_env import _import_ohq, new_ohq_system, read_outputs_at
import functional_flows as ff

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "merced_daily.npz")
DAY = 86400.0

TRAIN_YEARS = range(1989, 2017)          # water years used to fit targets and train
TEST_STARTS = (2017, 2020, 2023)         # 3-year test windows: WY2017-2025
SCENARIOS = ("minflow", "prescribed", "ffreward", "multi")


class McClureBackend:
    """Builds and solves the one-day reservoir model with OHQLib."""

    def __init__(self, S_cap, S_inactive, Q_spill_max=5000.0, solver_dt=0.1, workdir=None):
        self.ohq = _import_ohq()
        self.S_cap, self.S_inactive = S_cap * 1e6, S_inactive * 1e6       # m3
        self.Q_spill_max = Q_spill_max * DAY                               # m3/day
        self.dt = solver_dt
        self.workdir = workdir or tempfile.mkdtemp(prefix="ohq_eflow_")
        os.makedirs(self.workdir, exist_ok=True)
        self.inflow_file = os.path.join(self.workdir, "inflow.txt")
        self.output_file = os.path.join(self.workdir, "output.txt")

    def simulate_day(self, storage0, inflow, q_river, q_canal):
        """One day from storage0 [hm3] with constant inflow [m3/s] and outlet
        settings [m3/s]. Returns end storage [hm3] and the day-mean river
        outlet, canal and spill flows [m3/s]. The outlets taper to zero over
        the bottom 1% of capacity above the inactive pool, so the reservoir
        never drops below it."""
        np.savetxt(self.inflow_file, [[0.0, inflow * DAY], [2.0, inflow * DAY]],
                   fmt="%.6f", delimiter=", ")
        s = new_ohq_system(self.ohq, self.workdir, 0.0, 1.0, self.dt, self.output_file)
        s.create_block("Reservoir", "Reservoir",
                       {"Storage": f"{storage0 * 1e6}", "inflow": self.inflow_file})
        for sink in ("River", "Canal", "SpillSink"):
            s.create_block(sink, "fixed_head", {"head": "0", "Storage": "0"})
        band = (f"{self.S_inactive}", f"{self.S_inactive + 0.01 * self.S_cap}")
        s.create_link("river_outlet", "Reservoir_link_rule", "Reservoir", "River",
                      {"Q_min": "0", "Q_max": f"{q_river * DAY}", "S_min": band[0], "S_max": band[1]})
        s.create_link("canal_outlet", "Reservoir_link_rule", "Reservoir", "Canal",
                      {"Q_min": "0", "Q_max": f"{q_canal * DAY}", "S_min": band[0], "S_max": band[1]})
        s.create_link("spillway", "Reservoir_link_rule", "Reservoir", "SpillSink",
                      {"Q_min": "0", "Q_max": f"{self.Q_spill_max}",
                       "S_min": f"{self.S_cap}", "S_max": f"{1.01 * self.S_cap}"})
        s.finalize()
        if not s.solve(True, False):
            raise RuntimeError("OHQ solve failed")
        s.write_outputs()
        S, river, canal, spill = read_outputs_at(
            self.output_file, 1.0,
            ["Reservoir_Storage", "River_Storage", "Canal_Storage", "SpillSink_Storage"], False)
        return S / 1e6, river / DAY, canal / DAY, spill / DAY

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


def load_hydrology():
    """Unimpaired inflow by water year {wy: 365 values [m3/s]} and the
    training-year terciles of annual volume [hm3]."""
    d = np.load(DATA)
    years = ff.split_water_years(d["date"], d["fnf"])
    vol = {y: x.sum() * DAY / 1e6 for y, x in years.items()}
    terciles = np.percentile([vol[y] for y in TRAIN_YEARS], [100 / 3, 200 / 3])
    storage = ff.split_water_years(d["date"], d["storage"])
    oct1_storage = {y: float(x[0]) for y, x in storage.items() if not np.isnan(x[0])}
    return years, vol, terciles, oct1_storage


def demand_climatology():
    """Daily irrigation demand [m3/s] for one water year: the mean observed
    Merced diversion (reservoir outflow minus the river near Snelling,
    WY1994-2010, about 640 hm3/yr) for each month, interpolated to a daily
    curve."""
    # Oct ... Sep
    monthly = np.array([11.8, 1.5, 2.4, 1.3, 1.7, 10.3, 18.6, 33.0, 44.2, 50.0, 41.9, 26.9])
    mid = np.cumsum([31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30]) - 15.0
    days = np.arange(365)
    return np.interp(days, np.r_[mid - 365, mid, mid + 365], np.r_[monthly, monthly, monthly])


def min_instream_flow(year_type):
    """Stylized regulatory minimum flow [m3/s] for one water year: 75 cfs in
    late October, 180 cfs Nov-Mar (Davis-Grunsky), 75 cfs Apr-May, 25 cfs
    Jun-mid Oct; dry years get 60% of these."""
    cfs = 0.0283168
    q = np.full(365, 25 * cfs)
    q[15:31] = 75 * cfs
    q[31:182] = 180 * cfs
    q[182:243] = 75 * cfs
    return q * (0.6 if year_type == "dry" else 1.0)


def top_of_conservation(day, S_cap, S_flood=832.0):
    """Flood-control rule curve [hm3]: 350 kAF (432 hm3) of flood space from
    Nov 1 to Mar 1, refilling to full by Jun 1, drawing down from Sep 1."""
    pts_d = [0, 31, 151, 243, 335, 365]
    pts_s = [0.5 * (S_cap + S_flood), S_flood, S_flood, S_cap, S_cap, 0.5 * (S_cap + S_flood)]
    return float(np.interp(day, pts_d, pts_s))


class EflowsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario="ffreward", episode_years=3,
                 S_cap=1264.0, S_inactive=80.0, Q_river_min=0.1, Q_river_max=300.0, Q_safe=170.0,
                 Q_turbine_max=90.0, P_max=94.5, D_peak=50.0,
                 w_supply=5.0, w_flood=1.0, w_ff=5.0, w_hp=0.1,
                 residual=False, rule_kwargs=None, solver_dt=0.1, backend=None):
        super().__init__()
        assert scenario in SCENARIOS, scenario
        # residual=True: the agent's action is a correction added to the action
        # of the tuned rule (sop_rule with rule_kwargs), clipped to [-1, 1].
        # A zero correction reproduces the rule exactly, so training starts from
        # it; the correction has full authority to override it.
        self.residual = residual
        if residual and rule_kwargs is None:
            import json
            params = json.load(open(os.path.join(HERE, "results", "eflows_rule_params.json")))
            rule_kwargs = params["ffreward" if scenario == "multi" else scenario]
        self.rule_kwargs = rule_kwargs or {}
        self.scenario = scenario
        self.episode_years = episode_years
        self.S_cap, self.S_inactive = S_cap, S_inactive
        self.Q_river_min, self.Q_river_max, self.Q_safe = Q_river_min, Q_river_max, Q_safe
        self.Q_turbine_max, self.P_max, self.D_peak = Q_turbine_max, P_max, D_peak
        self.w_supply, self.w_flood, self.w_ff, self.w_hp = w_supply, w_flood, w_ff, w_hp
        self.window_days = {k: int(ff.window_mask(k).sum()) for k in ff.WINDOWS}
        self.backend = backend or McClureBackend(S_cap, S_inactive, solver_dt=solver_dt)

        self.years, self.volume, self.terciles, self.oct1_storage = load_hydrology()
        train_type = {y: self.classify(self.volume[y]) for y in TRAIN_YEARS}
        self.targets = ff.natural_targets({y: self.years[y] for y in TRAIN_YEARS}, train_type, Q_safe)
        self.demand = demand_climatology()
        self.mif = {t: min_instream_flow(t) for t in ff.YEAR_TYPES}
        self.prescribed = {t: ff.prescribed_hydrograph(self.targets[t]) for t in ff.YEAR_TYPES}
        self.train_starts = [y for y in TRAIN_YEARS if y + episode_years - 1 in TRAIN_YEARS]

        # obs: storage, inflow today / 7-day / 30-day mean, season (sin, cos),
        # forecast of the rest of the water year, yesterday's river flow,
        # demand, today's reference river flow (in action units), declared year type (3), pulse progress
        # (fall, peak, spring), flood-control space, spill risk
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
        # actions in [-1, 1]: river release, fraction of demand delivered
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------------ helpers
    def classify(self, volume):
        lo, hi = self.terciles
        return "dry" if volume < lo else "wet" if volume > hi else "moderate"

    # River release on a log scale between Q_river_min and Q_river_max: an
    # action error is then a relative flow error at every flow level.
    def river_from_action(self, a):
        lo, hi = np.log(self.Q_river_min), np.log(self.Q_river_max)
        return float(np.exp(lo + 0.5 * (float(a) + 1.0) * (hi - lo)))

    def action_from_river(self, q):
        lo, hi = np.log(self.Q_river_min), np.log(self.Q_river_max)
        return float(np.clip(2.0 * (np.log(max(q, self.Q_river_min)) - lo) / (hi - lo) - 1.0, -1.0, 1.0))

    @staticmethod
    def fraction_from_action(a):
        return float(np.clip(0.6 * (float(a) + 1.0), 0.0, 1.0))

    @staticmethod
    def action_from_fraction(f):
        return float(np.clip(f, 0.0, 1.0)) / 0.6 - 1.0

    def head(self, S):
        """Gross head [m] vs storage [hm3], about 70 m at the inactive pool and
        140 m at full pool (an approximation of the McClure elevation curve)."""
        x = np.clip((S - self.S_inactive) / (self.S_cap - self.S_inactive), 0.0, 1.0)
        return 70.0 + 70.0 * x ** 0.6

    def power(self, S, q_released):
        """Powerhouse output [MW]: 0.88 efficiency, flow capped at Q_turbine_max."""
        return min(self.P_max, 9.81 * 0.88 * min(q_released, self.Q_turbine_max) * self.head(S) / 1000.0)

    def _forcing(self, start_year):
        n = self.episode_years
        self.wy = [start_year + k for k in range(n)]
        self.inflow = np.concatenate([self.years[y] for y in self.wy])
        # Inflow history for the 7/30-day means at the start of the episode.
        prev = self.years.get(start_year - 1, self.years[start_year])
        self.inflow_hist = np.concatenate([prev[-30:], self.inflow])
        # Bulletin-120-style forecast of the remaining water-year runoff: the
        # true remainder times a log-normal error redrawn every month, with a
        # spread that shrinks as the snowpack becomes known.
        rng = self.np_random
        month_start = np.array([0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335, 365])
        sigma = np.array([0.5, 0.5, 0.5, 0.5, 0.35, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01])
        self.forecast = np.zeros(365 * n)
        self.declared = []
        for k, y in enumerate(self.wy):
            x = self.years[y]
            rest = np.cumsum(x[::-1])[::-1] * DAY / 1e6       # runoff from day d to Sep 30
            seen = np.r_[0.0, np.cumsum(x)[:-1]] * DAY / 1e6
            err = np.zeros(365)
            for m in range(12):
                z = rng.normal()
                err[month_start[m]:month_start[m + 1]] = np.exp(sigma[m] * z - 0.5 * sigma[m] ** 2)
            f = rest * err
            self.forecast[365 * k:365 * (k + 1)] = f
            self.declared.append([self.classify(seen[d] + f[d]) for d in range(365)])

    def _year_type(self, t=None):
        t = self.t if t is None else t
        return self.declared[t // 365][t % 365]

    def _floor(self, t):
        """Hard river floor today [m3/s] in the minflow / prescribed scenarios."""
        d, typ = t % 365, self._year_type(t)
        if self.scenario == "minflow":
            return float(self.mif[typ][d])
        if self.scenario == "prescribed":
            return float(self.prescribed[typ][d])
        return 0.0

    def _reference(self, t):
        """Today's reference river flow shown to the agent [m3/s]: the hard
        floor in minflow/prescribed, the prescribed functional-flow schedule
        of the declared year type otherwise (a hint, not a constraint)."""
        if self.scenario in ("minflow", "prescribed"):
            return self._floor(t)
        return float(self.prescribed[self._year_type(t)][t % 365])

    def _progress(self, key, D):
        return 1.0 if D == 0 else min(1.0, self.pulse_days[key] / D)

    def _obs(self):
        t, d = self.t, self.t % 365
        h = self.inflow_hist
        i = t + 30
        T = self.targets[self._year_type()]
        typ = self._year_type()
        S = self.storage
        return np.array([
            S / self.S_cap,
            np.log1p(h[i] / 10.0),
            np.log1p(h[i - 7:i].mean() / 10.0),
            np.log1p(h[i - 30:i].mean() / 10.0),
            np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365),
            self.forecast[t] / self.S_cap,
            np.log1p(self.q_prev / 10.0),
            self.demand[d] / self.D_peak,
            self.action_from_river(self._reference(t)),
            typ == "dry", typ == "moderate", typ == "wet",
            self._progress("FA", T["D_FA"]),
            self._progress("PK", T["D_PK"]),
            self._progress("SP", T["D_SP"]),
            (top_of_conservation(d, self.S_cap) - S) / self.S_cap,
            (S + self.forecast[t] - self.S_cap) / self.S_cap,
        ], dtype=np.float32)

    # ------------------------------------------------------------- reward
    def ff_reward(self, d, q, T):
        """Daily functional-flow reward for river flow q [m3/s] on water-year
        day d under targets T: w_ff times the day's increment of the
        functional-flow score (functional_flows.component_scores), so that the
        reward summed over a year is w_ff * (sum of component scores) minus a
        constant. Each component is worth w_ff per year (the spring pulse and
        the recession half each):
          baseflows  -w_ff * (1 - c) / N_window per day (c = day credit)
          pulses     +w_ff * increase of min(1, credited days / D)
          recession  -w_ff/2 * (1 - recession credit) / N_window per day
        Pulse progress already paid is never paid again, also when the
        declared year type, and with it D, changes mid-year."""
        r = 0.0
        for key, Q in (("WB", "Q_WB"), ("DS", "Q_DS")):
            if ff.in_window(key, d) and T[Q] > 0:
                r -= self.w_ff * (1.0 - float(ff.credit(q, T[Q]))) / self.window_days[key]
        for key, Q, D, w in (("FA", "Q_FA", "D_FA", 1.0), ("PK", "Q_PK", "D_PK", 1.0), ("SP", "Q_SP", "D_SP", 0.5)):
            if ff.in_window(key, d) and T[D] > 0 and self.pulse_days[key] < T[D]:
                self.pulse_days[key] += min(float(ff.credit(q, T[Q])), T[D] - self.pulse_days[key])
                frac = min(1.0, self.pulse_days[key] / T[D])
                r += w * self.w_ff * max(0.0, frac - self.pulse_paid[key])
                self.pulse_paid[key] = max(self.pulse_paid[key], frac)
        if ff.in_window("REC", d):
            rate = ff.recession_rate(self.q_hist[-8:])
            miss = min(1.0, max(0.0, rate - (1.0 + ff.TOL) * T["ROC_max"]) / T["ROC_max"])
            r -= 0.5 * self.w_ff * miss / self.window_days["REC"]
        return r

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start = options.get("start_year")
        if start is None:
            start = int(self.np_random.choice(self.train_starts))
        self._forcing(start)
        S0 = options.get("storage")
        if S0 is None:
            S0 = self.np_random.uniform(0.3, 0.85) * self.S_cap
        self.storage = float(S0)
        self.t = 0
        self.q_prev = float(self.years.get(start - 1, self.years[start])[-1])
        self.q_hist = [self.q_prev] * 8
        self.pulse_days = {"FA": 0.0, "PK": 0.0, "SP": 0.0}
        self.pulse_paid = {"FA": 0.0, "PK": 0.0, "SP": 0.0}
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if self.residual:
            a = np.clip(sop_rule(self, **self.rule_kwargs) + a, -1.0, 1.0)
        t, d = self.t, self.t % 365
        if d == 0:
            self.pulse_days = {"FA": 0.0, "PK": 0.0, "SP": 0.0}
            self.pulse_paid = {"FA": 0.0, "PK": 0.0, "SP": 0.0}
        typ = self._year_type()
        T = self.targets[typ]
        D = float(self.demand[d])
        floor = self._floor(t)
        q_set = max(self.river_from_action(a[0]), floor)
        q_canal_set = D * self.fraction_from_action(a[1])
        S0 = self.storage

        S, q_out, q_canal, q_spill = self.backend.simulate_day(S0, float(self.inflow[t]), q_set, q_canal_set)
        q_river = q_out + q_spill

        deficit = max(0.0, D - q_canal)
        r_supply = -self.w_supply * (deficit / self.D_peak) ** 2
        r_flood = -self.w_flood * max(0.0, q_river - self.Q_safe) / self.Q_safe
        P = self.power(0.5 * (S0 + S), q_out + q_canal)
        r_hp = self.w_hp * P / self.P_max
        self.q_hist = self.q_hist[-7:] + [q_river]
        r_ff = self.ff_reward(d, q_river, T)
        reward = r_supply + r_flood
        if self.scenario in ("ffreward", "multi"):
            reward += r_ff
        if self.scenario == "multi":
            reward += r_hp

        info = dict(t=t, wy=self.wy[t // 365], day=d, year_type=typ, storage=S, inflow=float(self.inflow[t]),
                    river=q_river, spill=q_spill, canal=q_canal, demand=D, deficit=deficit, floor=floor,
                    power=P, r_supply=r_supply, r_flood=r_flood, r_ff=r_ff, r_hp=r_hp)
        self.storage = S
        self.q_prev = q_river
        self.t += 1
        truncated = self.t >= 365 * self.episode_years
        obs = self._obs() if not truncated else self._obs_safe()
        return obs, float(reward), False, truncated, info

    def _obs_safe(self):
        # Observation after the last day: repeat the last day's forcing.
        self.t -= 1
        o = self._obs()
        self.t += 1
        return o

    def close(self):
        self.backend.close()


# ------------------------------------------------------------ rule policies
def sop_rule(env, hedge_storage=0.35, hedge_fraction=0.75, drawdown_days=10.0, eflow="auto", margin=0.0):
    """Standard operating policy with hedging and flood control.

    River: today's floor (minimum flow or prescribed functional-flow
    hydrograph). In the ffreward/multi scenarios (no floor), eflow='auto'
    follows the prescribed functional-flow hydrograph, so this rule is the
    'prescriptive' operation measured with the functional-flow reward.
    Flood control: storage above the rule curve [hm3] is released over
    drawdown_days [d], up to the safe channel capacity Q_safe [m3/s].
    Canal: full demand, cut to hedge_fraction when storage is below
    hedge_storage * S_cap.
    margin: release (1 + margin) times the e-flow schedule. Used for behavior
    cloning, so that the small errors of the cloned network stay above the
    targets.
    """
    t, d = env.t, env.t % 365
    typ = env._year_type()
    floor = env._floor(t)
    if env.scenario in ("ffreward", "multi") and eflow == "auto":
        floor = float(env.prescribed[typ][d])
    floor *= 1.0 + margin
    excess = env.storage - top_of_conservation(d, env.S_cap)
    q = floor
    if excess > 0:
        q = max(floor, min(env.Q_safe, excess * 1e6 / (drawdown_days * DAY)))
    frac = hedge_fraction if env.storage < hedge_storage * env.S_cap else 1.0
    return np.array([env.action_from_river(q), env.action_from_fraction(frac)], dtype=np.float32)


def zero_policy(env):
    """No correction: with residual=True this is the rule itself (used to
    initialize the residual actor at zero by behavior cloning)."""
    return np.zeros(2, dtype=np.float32)


def make_env(scenario):
    return lambda: EflowsEnv(scenario=scenario)
