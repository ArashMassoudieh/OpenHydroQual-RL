# OpenHydroQual-RL

Closed-loop reinforcement learning (PPO, PyTorch) for operating a reservoir
simulated by OpenHydroQual through the `openhydroqual_py` bindings.
The full write-up, with derivations, verification and results, is in
[report/main.pdf](report/main.pdf).

| file | purpose |
|---|---|
| `ohq_reservoir_env.py` | `OHQBackend` (builds/solves the OHQ model per day), `OHQReservoirEnv` (Gymnasium), rule-curve baseline, shared OHQ helpers |
| `ohq_cascade_env.py` | `CascadeBackend`, `CascadeEnv`: two reservoirs in series + uncontrolled tributary (2-D action), coordinated rule baseline |
| `train_ppo.py` | PPO + GAE in plain PyTorch, parallel rollouts with `AsyncVectorEnv` |
| `evaluate.py` | single reservoir: PPO vs. demand-only / rule-curve / tuned rule-curve on held-out seeds; figures + tables |
| `ohq_pond_env.py` | `PondBackend`, `PondEnv`: stormwater pond with a controllable valve, NOAA hourly rain, Laplace age tracer; Q_p and young-fraction objectives; fixed-opening and forecast-rule baselines |
| `prepare_rain.py` | converts a NOAA HPD v2 station file (here Beltsville, MD, USC00180700) to `data/beltsville_hourly.npz` |
| `evaluate_pond.py` | pond: baseline tuning, PPO vs. baselines on 2020-2025 rain, trade-off plot, tables |
| `evaluate_cascade.py` | cascade: 2-D threshold tuning, PPO vs. rule baselines (parallel), figures + tables |
| `diagnostics.py` | OHQ output-stamping check, mass balance, per-step timing |
| `report/` | LaTeX report (`latexmk -pdf main.tex`) |

```bash
source ~/Projects/ohq_rl_env/bin/activate
python diagnostics.py
python train_ppo.py --total-steps 409600 --num-envs 16 --out results/ppo_reservoir.pt
python evaluate.py --policy results/ppo_reservoir.pt --episodes 30
python train_ppo.py --env cascade --total-steps 1003520 --num-envs 16 --out results/ppo_cascade.pt
python evaluate_cascade.py --policy results/ppo_cascade.pt --episodes 30
python -I prepare_rain.py data/noaa_hpd/USC00180700.csv data/beltsville_hourly.npz   # after downloading the NOAA file
python train_ppo.py --env pond --total-steps 1003520 --num-envs 16 --out results/ppo_pond.pt
python evaluate_pond.py --policy results/ppo_pond.pt --episodes 30
cd report && latexmk -pdf main.tex
```

`OHQ_PY_PATH` and `OHQ_RESOURCES` override the bindings/template locations.
All OHQ models are built through `new_ohq_system()`, which sets `n_threads=1`:
OHQ must run single-threaded whenever models run in parallel.
The simulator sits behind `simulate_day(t0, S0, Q) -> (S1, released, spilled)`
and `set_inflow(t, q)`, so a codegen-compiled backend can be passed in as
`OHQReservoirEnv(backend=...)`.
