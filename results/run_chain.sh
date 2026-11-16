#!/bin/bash
# Resumable pipeline (3 workers, nice 19).  Each stage is skipped if its output exists.
cd /mnt/data/Siddhesh/College/SEM5/CS663/project
PY="nice -n 19 .venv/bin/python"
W=3
log() { echo "$1 $(date +%H:%M)" >> results/chain.log; }
# 0. let an already-running finalize (old 'stop' safeguard) finish, then archive it
while ps -eo pid,args | grep -v grep | grep -q "experiments/finalize_main.py --workers"; do sleep 30; done
if [ ! -f results/ablation/auto_stop.json ] && grep -q "^kappa" results/finalize.log; then
  $PY experiments/archive_auto.py auto_stop
  cp results/kappa.txt results/ablation/kappa_stop.txt
  cp results/ablation/kappa_calibration.json results/ablation/kappa_calibration_stop.json
fi
log "stage archive-stop ok"
[ -f results/ablation/eps_sweep_floor.json ] || $PY experiments/run_ablation.py --workers $W --only epsfloor > results/run_ablation_floor.log 2>&1
log "stage epsfloor ok"
[ -f results/finalize_floor.done ] || { $PY experiments/finalize_main.py --workers $W --skip-refine > results/finalize_floor.log 2>&1 && touch results/finalize_floor.done; }
log "stage finalize-floor ok"
[ -f results/ablation/R_sweep.json ] || $PY experiments/run_ablation.py --workers $W --only R > results/run_ablation_R.log 2>&1
log "stage R ok"
$PY experiments/run_kodak.py --workers $W > results/run_kodak.log 2>&1
log "stage kodak ok"
[ -f results/color/color.json ] || $PY experiments/run_color.py > results/run_color.log 2>&1
log "stage color ok"
$PY experiments/scale_eps.py > results/scale_eps.log 2>&1
$PY experiments/make_figures.py > results/make_figures.log 2>&1
log "stage figures ok -- ALL DONE"
