# St. Francis bioretention cell: PPO vs. the optimized forecast rule and fixed orifice

Underdrain-valve control of the St. Francis cell (half footprint, 30.2 mm), perfect rainfall
forecast, 2020-2023. Write-up: `paper/main.pdf`.

Paper sources and generated results are stored outside Git, under
`/home/arash/Dropbox/OHQ RL paper/OpenHydroQual-RL/stfrancis/` on the
research workstation. Ignored local links named `paper` and `results` preserve
the paths used below. Obtain these artifacts separately when using an existing
policy or rebuilding the write-up; compiled kernels must still be regenerated.

Objectives (integrals, normalized by the passive cell):
`J = Excess/Excess_p + R * Age_Load/Age_p`, Excess = ∫max(Q - 50 m³/d, 0) dt.
R = 0 is case 1 (excess only), R = 1 is case 2 (excess + residence time).

## Setup

- Python venv with numpy, torch, gymnasium, matplotlib (`~/Projects/ohq_rl_env` on Arash's machine;
  the scripts `source` that path, edit it to yours).
- OpenHydroQual master, built with `codegen/build/ohq_generate` (needs `fbf46c8`, the ½ in the
  exponential kernel, and `fc6f532`, the loadSeries leak fix).
- Source models from the forecast-uncertainty study (path in `build_models.py`, `STUDY`):
  `rt_campaign/models/R1_f0.50_half_30.2mm/sigma_0_ts_2` (rule) and
  `rt_campaign/step_check/passive_s0025_half` (passive).

## Pipeline

```bash
python build_models.py --qlim 50            # -> models/{rule,rl,passive}.ohq
# compiled kernels (not in git; machine-specific)
for m in rule rl passive; do
  o=outputs.txt; [ $m = rule ] && o=outputs_rule.txt
  ~/Projects/OpenHydroQual/codegen/build/ohq_generate models/$m.ohq ~/Projects/OpenHydroQual/resources/ \
      kernels/$m SF_$m Storage --project shared --outputs $o
  cmake -S kernels/$m -B kernels/$m/build -DCMAKE_BUILD_TYPE=Release && cmake --build kernels/$m/build
done
python tune_rule.py passive                 # [--split train|ho2020]
python tune_rule.py orifice --R 0 --R 1
python tune_rule.py rule --R 0 --R 1 --gens 35
python snapshots.py --R 1                   # episode start states
cd .. && python train_ppo.py --env stfrancis --env-kwargs '{"R": 1}' ...   # see campaign.sh
python evaluate_sf.py --R 1 --ckpts 'results/ppo_sf_R1_u*.pt'
python make_paper.py; python make_holdout.py
```

`campaign.sh` (whole record + 2023 test) and `campaign_split.sh ho2020` (2020 held out) chain
these steps. `stepwise.py` runs the forward selection of observation features.
Run OHQ models in parallel single-threaded (the env sets this); campaigns were run under
`systemd-run --user --scope -p MemoryMax=20G`.

## Files

| Path | What |
|---|---|
| `common.py`, `kernel.py`, `forecast.py` | kernel wrapper, whole-record runs, data splits, forecast Î |
| `sf_env.py` | gym environment (hourly residual correction on the rule; `features=` selects observations) |
| `results/{rule,orifice}_R{0,1}{,_train,_ho2020}.json`, `passive*.json` | tuned benchmarks (whole record / 2020-22 / 2021-23) |
| `results/eval_*.json/.txt` | scores of every controller and checkpoint |
| `results/logs*/` | training and tuning logs |
| `results/snapshots_*.npz` | episode start states |
| `results/stepwise/` | feature selection (`R1/summary.json`, `stepwise_log.txt`) |
| `results/ppo_sf_*.pt` | PPO policies and checkpoints (not in git, shared separately) |
