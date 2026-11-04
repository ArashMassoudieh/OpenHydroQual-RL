#!/usr/bin/env bash
# Full PPO trainings (whole record) and the held-out test (tune/train on 2020-2022, test on 2023).
#   systemd-run --user --collect --scope --unit=sf-campaign ./campaign.sh
#   phase 1: whole-record PPO R0 and R1 (6 workers each)  ||  2020-2022 passive, orifice, rule tuning (4 workers)
#   phase 2: 2020-2022 snapshots and PPO R0 and R1 (6 workers each)
#   phase 3: evaluations (whole record; 2020-2022 selection + 2023 test)
set -u
cd "$(dirname "$0")"; source ~/Projects/ohq_rl_env/bin/activate
L=results/logs; mkdir -p $L
log() { echo "$(date '+%F %T') $*"; }

train() {   # R SPLIT NENVS
  local R=$1 SPLIT=$2 N=$3 sfx=""; [ "$SPLIT" = train ] && sfx=_train
  (cd .. && python train_ppo.py --env stfrancis --env-kwargs "{\"R\": $R, \"split\": \"$SPLIT\"}" \
     --total-steps 3000000 --num-envs $N --rollout-steps 720 --epochs 10 --minibatch 1024 --lr 3e-4 \
     --gamma 0.995 --lam 0.95 --normalize-rewards --bc-rule sf_env:zero_policy --bc-steps 2000 \
     --bc-log-std -1.2 --save-every 20 --seed 1 --out stfrancis/results/ppo_sf_R${R}${sfx}.pt) \
     > $L/train_R${R}${sfx}.txt 2>&1
  log "trained R$R $SPLIT (exit $?)"
}

log "phase 1"
train 0 full 6 & P1=$!
train 1 full 6 & P2=$!
( python tune_rule.py passive --split train > $L/tune_passive_train.txt 2>&1 &&
  python tune_rule.py orifice --R 0 --R 1 --split train --workers 4 > $L/tune_orifice_train.txt 2>&1 &&
  python tune_rule.py rule --R 0 --R 1 --split train --workers 4 --gens 35 > $L/tune_rule_train.txt 2>&1
  log "tuned 2020-2022 (exit $?)" ) & P3=$!
wait $P1 $P2 $P3

log "phase 2"
python snapshots.py --R 0 --split train > $L/snap_R0_train.txt 2>&1
python snapshots.py --R 1 --split train > $L/snap_R1_train.txt 2>&1
train 0 train 6 & P1=$!
python evaluate_sf.py --R 0 --ckpts "results/ppo_sf_R0_u*.pt" --workers 2 --out results/eval_R0.json > results/eval_R0.txt 2>&1 &
train 1 train 6 & P2=$!
python evaluate_sf.py --R 1 --ckpts "results/ppo_sf_R1_u*.pt" --workers 2 --out results/eval_R1.json > results/eval_R1.txt 2>&1 &
wait

log "phase 3"
python evaluate_sf.py --R 0 --split train --ckpts "results/ppo_sf_R0_train_u*.pt" --workers 6 --out results/eval_R0_train.json > results/eval_R0_train.txt 2>&1
python evaluate_sf.py --R 1 --split train --ckpts "results/ppo_sf_R1_train_u*.pt" --workers 6 --out results/eval_R1_train.json > results/eval_R1_train.txt 2>&1
log "done"
