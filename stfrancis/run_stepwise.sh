#!/usr/bin/env bash
# Wait for the 2020 hold-out campaign (tuned rule, snapshots, all-feature PPO), then run the
# stepwise feature selection for both cases.
cd "$(dirname "$0")"; source ~/Projects/ohq_rl_env/bin/activate
until grep -q "done" results/campaign_ho2020.out; do sleep 60; done
python stepwise.py --cases 1 --split ho2020 --steps 1000000 --seeds 1 2 --workers 4 --parallel 4
