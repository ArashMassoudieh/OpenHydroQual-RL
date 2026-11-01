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
| `select_checkpoint.py` | scores training snapshots on the pond objective J (validation windows) and keeps the best |
| `evaluate_pond.py` | pond: baseline tuning, PPO vs. baselines on 2020-2025 rain, trade-off plot, tables |
| `evaluate_cascade.py` | cascade: 2-D threshold tuning, PPO vs. rule baselines (parallel), figures + tables |
| `diagnostics.py` | OHQ output-stamping check, mass balance, per-step timing |
| `ohq_eflows_env.py` | `McClureBackend`, `EflowsEnv`: Lake McClure-like reservoir (Merced R., CA), river + canal releases, four e-flow scenarios (minflow / prescribed / ffreward / multi), optional residual action on top of the tuned rule |
| `functional_flows.py` | CEFF-style functional-flow targets from unimpaired flow, graded year scores, prescribed functional-flow hydrograph |
| `prepare_merced.py` | downloads CDEC/USGS daily data to `data/merced_daily.npz` |
| `plot_eflows_setup.py`, `evaluate_eflows.py` | e-flow setup figures/tables; rule tuning, checkpoint selection, 2017-2025 comparison |
| `report/` | LaTeX report (`latexmk -pdf main.tex`) |
| `report_eflows/` | LaTeX report on the environmental-flow problem |

```bash
source ~/Projects/ohq_rl_env/bin/activate
python diagnostics.py
python train_ppo.py --total-steps 409600 --num-envs 16 --out results/ppo_reservoir.pt
python evaluate.py --policy results/ppo_reservoir.pt --episodes 30
python train_ppo.py --env cascade --total-steps 1003520 --num-envs 16 --out results/ppo_cascade.pt
python evaluate_cascade.py --policy results/ppo_cascade.pt --episodes 30
python -I prepare_rain.py data/noaa_hpd/USC00180700.csv data/beltsville_hourly.npz   # after downloading the NOAA file
python evaluate_pond.py --tune-only                       # tune the pond baselines (3-year windows)
python train_ppo.py --env pond --total-steps 1503232 --num-envs 16 --normalize-rewards --reward-clip 0 \
    --save-every 61 --bc-rule ohq_pond_env:forecast_rule_policy \
    --bc-kwargs '{"u_hold": 0.0, "u_pre": 0.1, "rain_threshold": 0.015}' --out results/ppo_pond.pt
python select_checkpoint.py "results/ppo_pond_u*.pt" --out results/ppo_pond.pt   # pick the snapshot with the best J
python evaluate_pond.py --policy results/ppo_pond.pt
cd report && latexmk -pdf main.tex
# environmental flows (report_eflows/)
python prepare_merced.py && python plot_eflows_setup.py
python evaluate_eflows.py --tune
python train_ppo.py --env eflows --env-kwargs '{"scenario": "ffreward", "residual": true}' \
    --total-steps 3981312 --num-envs 24 --rollout-steps 512 --gamma 0.995 --lr 5e-5 --epochs 5 \
    --normalize-rewards --reward-clip 0 --save-every 18 \
    --bc-rule ohq_eflows_env:zero_policy --bc-log-std -4.6 --out results/ppo_eflows_ffreward.pt
python evaluate_eflows.py --select "results/ppo_eflows_ffreward_u*.pt" --scenario ffreward --env-kwargs '{"residual": true}'
python evaluate_eflows.py --compare
```

`OHQ_PY_PATH` and `OHQ_RESOURCES` override the bindings/template locations.
All OHQ models are built through `new_ohq_system()`, which sets `n_threads=1`:
OHQ must run single-threaded whenever models run in parallel.
The simulator sits behind `simulate_day(t0, S0, Q) -> (S1, released, spilled)`
and `set_inflow(t, q)`, so a codegen-compiled backend can be passed in as
`OHQReservoirEnv(backend=...)`.
