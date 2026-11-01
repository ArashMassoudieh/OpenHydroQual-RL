"""
Gymnasium environment: real-time control of a stormwater pond's outlet valve,
simulated with OpenHydroQual (OHQ), driven by observed hourly rainfall from
NOAA (Beltsville, MD, between Baltimore and Washington DC).

    rain --> impervious catchment --> [ Pond ] --valve (agent, opening u)--> outlet
                                           +----overflow weir (crest 2 m)---->

Two competing objectives:
  * peak-flow control: minimize Q_p, the outflow exceeded a fraction p of the
    time (OHQ's 'exceedance' objective);
  * treatment (residence time): minimize the young-water fraction of the
    outflow, measured with OHQ's Laplace age tracer (AgeTracker_laplace): every
    inflow carries tracer at concentration 1, which decays at rate lambda, so the
    pond concentration is C = E[exp(-lambda*age)]. With lambda = 1/day,
    C ~ fraction of 'young' water; the young-water fraction of the outflow is the
    exported tracer load divided by the outflow volume.
Holding water (valve closed) ages it but fills the pond, so the next storm
overflows (high Q_p); opening the valve keeps room but passes young water.

Training reward (per hour): a smooth surrogate for the exceedance quantile
(squared excess of outflow over Q_ref, at every solver step) plus the exported
young-water load. Policies are evaluated on the exact Q_p and young fraction.

The binding cannot create constituents or reactions, so each step's model is
written as an .ohq script and loaded with create_from_script (the age-tracer
construction follows StFrancis_RTC_runs/tools/add_age.py). Units: m, m^3, days.
"""
import os
import shutil
import tempfile

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ohq_reservoir_env import OHQ_RESOURCES, _import_ohq

HERE = os.path.dirname(os.path.abspath(__file__))
RAIN_FILE = os.path.join(HERE, "data", "beltsville_hourly.npz")
# The valve is OHQ's controlled_orifice (resources/Pond_Plugin.json): the orifice
# law with the opening as a property and the regularization h/sqrt(h+eps), which
# equals sqrt(h) for real heads but keeps the slope finite at an empty pond, where
# the plain sqrt law made the implicit solver fail.
HOUR = 1.0 / 24.0


class PondBackend:
    """Builds (from a generated .ohq script) and solves the pond model for one hour."""

    def __init__(self, alpha=250.0, beta=2.0, crest=2.0, A_max=0.03, Cd=0.6,
                 weir_alpha=440_000.0, weir_beta=1.5, lam=1.0, substeps=12, eps_head=0.001,
                 cb_area=4.0, workdir=None):
        self.ohq = _import_ohq()
        self.alpha, self.beta, self.crest = alpha, beta, crest
        self.A_max, self.Cd, self.weir_alpha, self.weir_beta = A_max, Cd, weir_alpha, weir_beta
        self.lam = lam
        self.eps_head = eps_head
        self.cb_area = cb_area
        self.substeps = substeps
        self.dt = HOUR / substeps
        self.workdir = workdir or tempfile.mkdtemp(prefix="ohq_pond_")
        os.makedirs(self.workdir, exist_ok=True)
        self.inflow_file = os.path.join(self.workdir, "inflow.txt")
        self.script_file = os.path.join(self.workdir, "step.ohq")
        self.output_file = os.path.join(self.workdir, "output.txt")

    def set_inflow(self, t_hours, q):
        """Piecewise-constant hourly inflow [m3/d]; q[k] applies on [k, k+1) h."""
        self.q_hourly = np.asarray(q, dtype=float)

    def _write_hour_inflow(self, t0):
        # Only this hour's (constant) inflow: OHQ re-reads the inflow file at every
        # build, and a whole-episode file (52,000 points for 3 years) made each step
        # cost seconds instead of milliseconds.
        q = self.q_hourly[int(round(t0 / HOUR))]
        with open(self.inflow_file, "w") as f:
            f.write(f"{t0 - HOUR:.10f}, {q}\n{t0 + 2 * HOUR:.10f}, {q}\n")

    def outflow(self, depth, u, tail=0.0):
        """Valve (orifice) and overflow weir flow [m3/d] at pond depth [m]; the valve
        sees the head difference to the catch basin's water level `tail` [m] (both
        bottoms are at elevation 0), the weir discharges freely."""
        h = np.maximum(depth - tail, 0.0)
        valve = 86400 * self.Cd * u * self.A_max * np.sqrt(19.6) * h / np.sqrt(h + self.eps_head)
        weir = self.weir_alpha * np.maximum(depth - self.crest, 0.0) ** self.weir_beta
        return valve, weir

    def _script(self, t0, S0, C0, S_cb, C_cb, u):
        R, W = OHQ_RESOURCES, self.workdir + "/"
        a = "Age"
        return "\n".join([
            f"loadtemplate; filename = {R}main_components.json",
            f"addtemplate; filename = {R}Pond_Plugin.json",
            f"addtemplate; filename = {R}mass_transfer.json",
            f"addtemplate; filename = {R}Sewer_system.json",
            f"setvalue; object=system, quantity=simulation_start_time, value={t0!r}",
            f"setvalue; object=system, quantity=simulation_end_time, value={t0 + HOUR!r}",
            f"setvalue; object=system, quantity=initial_time_step, value={self.dt!r}",
            "setvalue; object=system, quantity=max_timestep_increase_factor, value=1",
            # One tiny model per worker: OHQ must run single-threaded in parallel runs.
            "setvalue; object=system, quantity=n_threads, value=1",
            f"setvalue; object=system, quantity=alloutputfile, value={self.output_file}",
            # Laplace age tracer (as StFrancis_RTC_runs/tools/add_age.py): unit
            # inflow concentration, first-order decay at Laplace_param = lambda.
            f"create constituent;type=AgeTracker_laplace,concentration=0,decay_parameter=0,"
            f"diffusion_coefficient=0,external_mass_flow_timeseries=,external_source=,"
            f"inflow_concentration=0,name={a},stoichiometric_constant=0",
            f"create reaction_parameter;type=ReactionParameter,Arrhenius_factor=1,"
            f"base_value={self.lam},name=Laplace_param,reference_temperature=0,temperature=",
            f"create reaction;type=Reaction,{a}:stoichiometric_constant=-1,name=decay,"
            f"rate_expression=(Laplace_param*{a})",
            f"create block;type=Pond,{a}:concentration={C0!r},{a}:external_mass_flow_timeseries=,"
            f"{a}:external_source=,{a}:inflow_concentration=1,Evapotranspiration=,Precipitation=,"
            f"Storage={S0!r},alpha={self.alpha},beta={self.beta},bottom_elevation=0,"
            f"inflow={self.inflow_file},name=Pond",
            # Catch basin (as in the St. Francis model): receives both the valve
            # (underdrain) and the overflow, and discharges through a sewer pipe to
            # the outlet. Exceedance and young-water load are measured on that pipe.
            f"create block;type=Catch basin,{a}:concentration={C_cb!r},{a}:external_mass_flow_timeseries=,"
            f"{a}:external_source=,{a}:inflow_concentration=1,Storage={S_cb!r},area={self.cb_area},"
            f"bottom_elevation=0,inflow=,max_height=1e12,spread_side_slope=0,name=CatchBasin",
            f"create block;type=fixed_head,{a}:concentration=0,{a}:external_mass_flow_timeseries=,"
            f"{a}:external_source=,{a}:inflow_concentration=0,Storage=0,head=-1,name=Outlet",
            f"create link;from=Pond,to=CatchBasin,type=controlled_orifice,{a}:decay_parameter=0,"
            f"head_loss_coefficient={self.Cd},max_area={self.A_max},opening={u!r},"
            f"regularization_head={self.eps_head},name=valve",
            f"create link;from=Pond,to=CatchBasin,type=wier,{a}:decay_parameter=0,"
            f"alpha={self.weir_alpha},beta={self.weir_beta},crest_elevation={self.crest},name=overflow",
            f"create link;from=CatchBasin,to=Outlet,type=Sewer_pipe,{a}:decay_parameter=0,"
            f"ManningCoeff=0.01,diameter=0.3,length=5,start_elevation=0,end_elevation=-0.5,name=outlet_pipe",
            ""])

    def simulate_hour(self, t0, S0, C0, S_cb, C_cb, u):
        """Run one hour. Returns the end state of pond and catch basin, and per
        solver step the outlet-pipe flow and young-water (tracer) load, and the
        pond's overflow."""
        self._write_hour_inflow(t0)
        with open(self.script_file, "w") as f:
            f.write(self._script(t0, S0, C0, S_cb, C_cb, u))
        s = self.ohq.System()
        s.set_default_template_path(OHQ_RESOURCES)
        s.set_working_folder(self.workdir + "/")
        s.set_silent(True)
        if not s.create_from_script(self.script_file, OHQ_RESOURCES + "settings.json"):
            raise RuntimeError("OHQ could not build the pond model")
        if not s.solve(True, False):
            raise RuntimeError(f"OHQ solve failed for the hour starting at t={t0}")
        s.write_outputs()
        with open(self.output_file) as f:
            header = [h.strip() for h in f.readline().split(",")]
        data = np.loadtxt(self.output_file, delimiter=",", skiprows=1, ndmin=2)
        grid = t0 + self.dt * np.arange(1, self.substeps + 1)     # end of each solver step

        def series(name):                 # every output column has its own time column
            c = header.index(name)
            return np.interp(grid, data[:, c - 1], data[:, c])

        depth = series("Pond_depth")
        C_pond = series("Pond_Age:concentration")
        S_cb_n = series("CatchBasin_Storage")
        C_cb_n = series("CatchBasin_Age:concentration")
        valve, weir = self.outflow(depth, u, tail=series("CatchBasin_head"))
        # Volume and young-water load leaving through the outlet pipe this hour,
        # from the catch basin's balance: inflow from the pond (backward Euler:
        # end-of-step pond state) minus the change in storage / tracer mass minus
        # tracer decay in the basin. Exact by construction, unlike integrating
        # OHQ's interpolated rate outputs.
        q_in = valve + weir
        vol = float(np.sum(q_in) * self.dt) - (S_cb_n[-1] - S_cb)
        load = (float(np.sum(q_in * C_pond) * self.dt) - (S_cb_n[-1] * C_cb_n[-1] - S_cb * C_cb)
                - self.lam * float(np.sum(S_cb_n * C_cb_n) * self.dt))
        return dict(S=float(series("Pond_Storage")[-1]), C=float(C_pond[-1]),
                    S_cb=float(S_cb_n[-1]), C_cb=float(C_cb_n[-1]),
                    depth=depth, Q_weir=weir,
                    Q=np.maximum(series("outlet_pipe_flow"), 0.0),   # pipe flow samples, for Q_p
                    volume=max(vol, 0.0), load=max(load, 0.0))

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


class PondEnv(gym.Env):
    metadata = {"render_modes": []}
    FORECAST_HOURS = (1, 3, 6, 12, 24)

    YEAR_HOURS = 8760
    TEST_STARTS = (2020.0, 2023.0)      # two disjoint 3-year test windows

    def __init__(self, split="train", episode_hours=3 * 8760, catchment_area=20_000.0, runoff_coeff=0.9,
                 Q_ref=1_000.0, V_ref=1_000.0, w_peak=1.0, w_young=5.0, w_terminal=0.0,
                 exceed_p=0.001, forecast_noise=0.3, max_missing=0.01, backend=None, rain_file=RAIN_FILE):
        super().__init__()
        self.split = split
        self.episode_hours = episode_hours
        self.catchment_area, self.runoff_coeff = catchment_area, runoff_coeff
        self.Q_ref, self.V_ref = Q_ref, V_ref
        self.w_peak, self.w_young, self.w_terminal = w_peak, w_young, w_terminal
        self.exceed_p = exceed_p
        self.forecast_noise = forecast_noise
        self.backend = backend or PondBackend()
        self.S_crest = self.backend.alpha * self.backend.crest ** self.backend.beta

        rain = np.load(rain_file)
        hours, r = rain["hour_since_1900"], rain["rain_m"]
        year = 1900 + hours / 24 / 365.2425
        # Train on 1980-2019, test on 2020 onward (years never seen in training).
        in_split = (year >= 1980) & (year < 2020) if split == "train" else (year >= 2020)
        # Valid episode starts: windows (plus forecast horizon) inside the split
        # with at most max_missing missing hours, which are then taken as dry.
        need = episode_hours + max(self.FORECAST_HOURS)
        bad = (np.isnan(r) | ~in_split).astype(np.int64)
        outside = np.concatenate([[0], np.cumsum(~in_split)])
        csum = np.concatenate([[0], np.cumsum(bad)])
        starts = np.arange(len(r) - need)
        ok = ((outside[starts + need] - outside[starts]) == 0) & \
             ((csum[starts + need] - csum[starts]) <= max_missing * need)
        self.valid_starts = starts[ok]
        if len(self.valid_starts) == 0:
            raise ValueError(f"no {need}-hour windows with <= {max_missing:.1%} missing in split '{split}'")
        self.year = year
        self.rain = np.nan_to_num(np.where(in_split, r, np.nan), nan=0.0)
        if split == "test":   # first valid start on or after each test-window date
            self.test_starts = [int(self.valid_starts[np.searchsorted(year[self.valid_starts], y0)])
                                for y0 in self.TEST_STARTS]

        # obs: depth/crest, pond tracer C, current inflow, forecast rain over the
        #      next 1/3/6/12/24 h, fraction of the episode left
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        # action a in [-1, 1] -> valve opening u = ((a+1)/2)^2 in [0, 1]: the useful
        # small openings (0-10%) get about a third of the action range instead of 5%
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------- forcing
    def _load_window(self, start):
        n = self.episode_hours + max(self.FORECAST_HOURS)
        self.rain_h = self.rain[start:start + n]                               # m per hour
        self.inflow = self.runoff_coeff * self.rain_h * self.catchment_area * 24  # m3/d
        self.backend.set_inflow(None, self.inflow[:self.episode_hours])
        csum = np.concatenate([[0], np.cumsum(self.rain_h)])
        rng = self.np_random
        self.forecast = np.zeros((self.episode_hours, len(self.FORECAST_HOURS)))
        for j, k in enumerate(self.FORECAST_HOURS):
            true = csum[np.arange(self.episode_hours) + k + 1] - csum[np.arange(self.episode_hours) + 1]
            noise = np.exp(rng.normal(0, self.forecast_noise * np.sqrt(k / 6), self.episode_hours))
            self.forecast[:, j] = true * noise

    def _obs(self):
        h = min(self.hour, self.episode_hours - 1)   # the final observation reuses the last hour
        return np.array([
            self.depth() / self.backend.crest,
            self.C,
            self.inflow[h] / self.Q_ref,
            *(self.forecast[h] / 0.025),          # forecast rain, in units of 25 mm (1 inch)
            1.0 - h / self.episode_hours,
        ], dtype=np.float32)

    def depth(self):
        S, a, b = self.S, self.backend.alpha, self.backend.beta
        return max(S * S / (a * (S + 0.1)), 0.0) ** (1 / b)   # Pond_Plugin.json depth formula

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start = int(self.valid_starts[self.np_random.integers(len(self.valid_starts))])
        if options and "start" in options:
            start = int(options["start"])
        elif options and "test_window" in options:
            start = self.test_starts[int(options["test_window"])]
        self.start = start
        self._load_window(start)
        self.hour = 0
        self.S = float(self.np_random.uniform(0.0, 0.15) * self.S_crest)
        self.C = float(self.np_random.uniform(0.2, 0.8))
        self.S_cb, self.C_cb = 1e-4, 0.0
        self.Q_all, self.load_h, self.volume_h = [], [], []
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        u = opening_from_action(a)
        h = self.hour
        r = self.backend.simulate_hour(h * HOUR, self.S, self.C, self.S_cb, self.C_cb, u)
        dt = self.backend.dt
        # Measured on the catch basin -> outlet pipe: flow and young-water (tracer) load.
        vol, young = r["volume"], r["load"]
        r_peak = -self.w_peak * float(np.sum((np.maximum(r["Q"] - self.Q_ref, 0.0) / self.Q_ref) ** 2) * dt / HOUR)
        r_young = -self.w_young * young / self.V_ref
        reward = r_peak + r_young

        self.Q_all.append(r["Q"])
        self.load_h.append(young)
        self.volume_h.append(vol)
        self.S, self.C = self._snap(r["S"], r["C"], 0.1)
        self.S_cb, self.C_cb = self._snap(r["S_cb"], r["C_cb"], 0.01)
        self.hour += 1
        terminated = self.hour >= self.episode_hours
        if terminated:
            # Water still in the pond and catch basin will leave later at (about)
            # its current age: count it as exported now, so holding water to the end
            # of the window cannot hide young water.
            left = self.S + self.S_cb
            left_load = self.S * self.C + self.S_cb * self.C_cb
            self.load_h[-1] += left_load
            self.volume_h[-1] += left
            reward -= self.w_young * left_load / self.V_ref + self.w_terminal * self.S / self.S_crest
        info = dict(hour=h, u=u, depth=self.depth(), C=self.C, inflow=self.inflow[h],
                    Q_mean=vol / HOUR, Q_max=float(r["Q"].max()), overflow=float(np.sum(r["Q_weir"]) * dt),
                    young=young, volume=vol, rain=self.rain_h[h], r_peak=r_peak, r_young=r_young)
        if terminated:
            info.update(self.episode_metrics())
        return self._obs(), float(reward), terminated, False, info

    @staticmethod
    def _snap(S, C, S_min):
        """Round-off-level storage (an emptied pond or catch basin) and vanishing
        tracer concentrations are set to exactly zero: values like 1e-11 m3 and
        1e-150 made OHQ's Newton iterations crawl (seconds per hour instead of
        milliseconds), while exactly zero solves fast."""
        S = S if S > S_min else 0.0
        C = min(C, 1.0) if (S > 0.0 and C > 1e-9) else 0.0
        return S, C

    def episode_metrics(self):
        """Exact evaluation metrics (5-minute resolution) over the whole episode and
        for each successive year of it."""
        steps = self.backend.substeps

        def metrics(h0, h1):
            Q = np.concatenate(self.Q_all[h0:h1]) if h1 > h0 else np.zeros(1)
            vol, load = float(np.sum(self.volume_h[h0:h1])), float(np.sum(self.load_h[h0:h1]))
            return dict(Q_p=float(np.quantile(Q, 1.0 - self.exceed_p)),
                        young_fraction=load / vol if vol > 0 else 0.0,
                        outflow_volume=vol, Q_peak=float(Q.max()))

        out = metrics(0, len(self.Q_all))
        n_years = len(self.Q_all) // self.YEAR_HOURS
        out["per_year"] = [metrics(k * self.YEAR_HOURS, (k + 1) * self.YEAR_HOURS) for k in range(n_years)]
        return out

    def close(self):
        self.backend.close()


def opening_from_action(a):
    return (0.5 * (a + 1.0)) ** 2


def action_from_opening(u):
    return np.array([2.0 * np.sqrt(np.clip(u, 0.0, 1.0)) - 1.0], dtype=np.float32)


# ------------------------------------------------------------- baselines
def passive_policy(env, u=1.0):
    """Fixed valve opening u (u = 1: conventional dry pond with an open orifice)."""
    return action_from_opening(u)


def forecast_rule_policy(env, u_hold=0.1, u_pre=0.4, rain_threshold=0.015, horizon_index=4, depth_open=1.8):
    """Forecast-informed rule (in the spirit of the St. Francis RTC rule):
    hold water at a small opening u_hold so it ages; when the forecast rain over
    the horizon (default 24 h) exceeds rain_threshold [m], pre-release at opening
    u_pre to make room; open fully if the pond is deeper than depth_open [m]."""
    h = min(env.hour, env.episode_hours - 1)
    u = u_hold
    if env.forecast[h, horizon_index] > rain_threshold:
        u = u_pre
    if env.depth() > depth_open:
        u = 1.0
    return action_from_opening(u)


def evaluation_objective(Q_p, young_fraction, Q_p_passive, young_passive):
    """Scalar used to tune baselines and compare policies (lower is better): Q_p
    and the young-water fraction, each normalized by the passive pond's value
    (valve fully open) under the same rain. The passive pond scores 2."""
    return Q_p / Q_p_passive + young_fraction / young_passive
