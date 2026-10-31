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
cd report && latexmk -pdf main.tex
```

`OHQ_PY_PATH` and `OHQ_RESOURCES` override the bindings/template locations.
All OHQ models are built through `new_ohq_system()`, which sets `n_threads=1`:
OHQ must run single-threaded whenever models run in parallel.
The simulator sits behind `simulate_day(t0, S0, Q) -> (S1, released, spilled)`
and `set_inflow(t, q)`, so a codegen-compiled backend can be passed in as
`OHQReservoirEnv(backend=...)`.
