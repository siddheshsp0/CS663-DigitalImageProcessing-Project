#!/bin/bash
# follow-up: MULTI-autoS (scale-aware eps) after the main chain has finished
cd /mnt/data/Siddhesh/College/SEM5/CS663/project
PY="nice -n 19 .venv/bin/python"
log(){ echo "$1 $(date +%H:%M)" >> results/chain.log; }
while ps -eo pid,args | grep -v grep | grep -q "results/run_chain.sh"; do sleep 30; done
$PY experiments/run_auto_scale.py --workers 3 > results/run_auto_scale.log 2>&1 && log "stage autoS ok"
$PY experiments/make_figures.py > results/make_figures.log 2>&1 && log "stage figures2 ok -- ALL DONE 2"
