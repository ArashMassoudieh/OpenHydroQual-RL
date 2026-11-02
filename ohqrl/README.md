# ohqrl: reinforcement learning for any OpenHydroQual model

`ohqrl` turns an OpenHydroQual (OHQ) model into a control problem. You write a
YAML file that says:
- which forcings or settings are controlled (the actions);
- which state variables the agent sees;
- which of the model's own objective functions to minimize.

The OHQ model itself is unchanged: a normal `.ohq` script.

| file | purpose |
|---|---|
| `config.py` | loads and validates the YAML (schema in the module docstring) |
| `env.py` | `OHQControlEnv`, a Gymnasium environment that steps the OHQ model interval by interval; `constant_policy` |
| `evaluate.py` | runs policies (constant controls, PPO checkpoints) on validation/test windows, reports OHQ objective values, selects the best checkpoint |

## How it works

Each episode builds the model once and steps it with the OHQ bindings'
stepping API. The steps are:

1. Call `begin_stepping()`.
2. Then, every control interval:
   1. Write the action to each control: `set_value` for a value property, `set_timeseries` for a time-series property.
   2. Call `advance_to(t + Δt)`. This continues the simulation without re-initializing it.
   3. Read the selected states with `get_values`.
   4. Read each objective function's running integral with `objective_integral`.

**Reward.** The reward for an interval is minus the weighted increase of the
objective integrals over that interval. Summed over an episode, the rewards
are exactly minus the weighted objective that OHQ computes for that window.
The agent is therefore trained on exactly what it is judged by.

**Additive objectives only.** Only objective functions with method `integrate`
are accepted. Formulate anything else as an integral, e.g.
- a peak-flow objective as `(_pos(flow-Qref)*_pos(flow-Qref)/Qref)`;
- a pollutant load as `flow*C.s`.

**Episode end.** The end of an episode is a time limit, not a terminal state.
The trainer bootstraps from the critic's value of the last state.

## Usage

```bash
python examples/pond/make_inputs.py                      # build the example's inflow record
python -m ohqrl.evaluate examples/pond/pond.yaml --windows test --constant "valve=0.1"
python train_ppo.py --env ohq --env-kwargs '{"config": "examples/pond/pond.yaml"}' \
    --total-steps 2000000 --num-envs 24 --normalize-rewards --reward-clip 0 --save-every 50 \
    --bc-rule ohqrl.env:constant_policy --out results/ppo_ohq_pond.pt
python -m ohqrl.evaluate examples/pond/pond.yaml --windows validation \
    --ppo "results/ppo_ohq_pond_u*.pt" --select results/ppo_ohq_pond.pt
python -m ohqrl.evaluate examples/pond/pond.yaml --windows test --ppo results/ppo_ohq_pond.pt --normalize
```

## Requirements on the OHQ side

The stepping API and accessors in `openhydroqual_py`:
- `begin_stepping`, `advance_to`, `end_stepping`
- `get_value`, `get_values`, `set_value`, `set_timeseries`
- `objective_names`, `objective_method`, `objective_weight`, `objective_integral`

`tests/test_stepping.py` checks that a stepped run with the control changed at
every interval reproduces a full `solve()` exactly. It also checks that a value
property changed mid-run takes effect.

Two practical settings, both handled by the environment or the config:
- **Run-length limits.** An episode is one long run. The environment lifts
  OHQ's whole-run limits on matrix inversions and wall-clock time.
- **Snapping.** Near-empty storages left at round-off levels make the solver
  crawl. The optional `snap` list sets tiny states to zero before each interval.
