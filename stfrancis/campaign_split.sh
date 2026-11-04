#!/usr/bin/env bash
# One held-out-year experiment: tune the rule and orifice and train PPO on the training
# window of SPLIT, then score every controller on the held-out year.
#   systemd-run --user --collect --scope --unit=sf-ho2020 -p MemoryMax=20G ./campaign_split.sh ho2020
set -u
SPLIT=${1:?usage: ./campaign_split.sh <split: train|ho2020>}
cd "$(dirname "$0")"; source ~/Projects/ohq_rl_env/bin/activate
SFX=$(python -c "from common import SPLIT_DEFS; print(SPLIT_DEFS['$SPLIT']['sfx'])")
L=results/logs_$SPLIT; mkdir -p $L
log() { echo "$(date '+%F %T') $*"; }

train() {   # R NENVS
  local R=$1 N=$2
  (cd .. && python train_ppo.py --env stfrancis --env-kwargs "{\"R\": $R, \"split\": \"$SPLIT\"}" \
     --total-steps 3000000 --num-envs $N --rollout-steps 720 --epochs 10 --minibatch 1024 --lr 3e-4 \
     --gamma 0.995 --lam 0.95 --normalize-rewards --bc-rule sf_env:zero_policy --bc-steps 2000 \
     --bc-log-std -1.2 --save-every 20 --seed 1 --out stfrancis/results/ppo_sf_R${R}${SFX}.pt) \
     > $L/train_R${R}.txt 2>&1
  log "trained R$R (exit $?)"
}

log "tuning on the $SPLIT window"
python tune_rule.py passive --split $SPLIT > $L/tune_passive.txt 2>&1
python tune_rule.py orifice --R 0 --R 1 --split $SPLIT --workers 8 > $L/tune_orifice.txt 2>&1
python tune_rule.py rule --R 0 --split $SPLIT --workers 6 --gens 35 > $L/tune_rule_R0.txt 2>&1 &
python tune_rule.py rule --R 1 --split $SPLIT --workers 6 --gens 35 > $L/tune_rule_R1.txt 2>&1 &
wait; log "tuned"
python snapshots.py --R 0 --split $SPLIT > $L/snap_R0.txt 2>&1
python snapshots.py --R 1 --split $SPLIT > $L/snap_R1.txt 2>&1
train 0 6 & train 1 6 & wait
log "evaluating"
python evaluate_sf.py --R 0 --split $SPLIT --ckpts "results/ppo_sf_R0${SFX}_u*.pt" --workers 6 --out results/eval_R0$SFX.json > results/eval_R0$SFX.txt 2>&1
python evaluate_sf.py --R 1 --split $SPLIT --ckpts "results/ppo_sf_R1${SFX}_u*.pt" --workers 6 --out results/eval_R1$SFX.json > results/eval_R1$SFX.txt 2>&1
log "done"
