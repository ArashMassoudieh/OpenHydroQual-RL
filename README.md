# OpenHydroQual-RL

Closed-loop reinforcement learning (PPO, PyTorch) for operating a reservoir
simulated by OpenHydroQual through the `openhydroqual_py` bindings.
The full write-up, with derivations, verification and results, is in
[report/main.pdf](report/main.pdf).

| file | purpose |
|---|---|
| `ohq_reservoir_env.py` | `OHQBackend` (builds/solves the OHQ model per day), `OHQReservoirEnv` (Gymnasium), rule-curve baseline |
| `train_ppo.py` | PPO + GAE in plain PyTorch, parallel rollouts with `AsyncVectorEnv` |
| `evaluate.py` | PPO vs. demand-only / rule-curve / tuned rule-curve on held-out seeds; figures + tables |
| `diagnostics.py` | OHQ output-stamping check, mass balance, per-step timing |
| `report/` | LaTeX report (`latexmk -pdf main.tex`) |

```bash
source ~/Projects/ohq_rl_env/bin/activate
python diagnostics.py
python train_ppo.py --total-steps 409600 --num-envs 16 --out results/ppo_reservoir.pt
python evaluate.py --policy results/ppo_reservoir.pt --episodes 30
cd report && latexmk -pdf main.tex
```

`OHQ_PY_PATH` and `OHQ_RESOURCES` override the bindings/template locations.
The simulator sits behind `simulate_day(t0, S0, Q) -> (S1, released, spilled)`
and `set_inflow(t, q)`, so a codegen-compiled backend can be passed in as
`OHQReservoirEnv(backend=...)`.
