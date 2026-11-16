#!/bin/bash
cd /mnt/data/Siddhesh/College/SEM5/CS663/project
PY="nice -n 19 .venv/bin/python"
log(){ echo "$1 $(date +%H:%M)" >> results/chain.log; }
while ps -eo pid,args | grep -v grep | grep -q "results/run_chain3.sh"; do sleep 30; done
[ -f results/ablation/kodak_auto_rules.json ] || $PY experiments/kodak_auto_rules.py --workers 3 > results/kodak_auto_rules.log 2>&1 && log "stage kodak-auto-rules ok"
$PY experiments/make_figures.py > results/make_figures.log 2>&1 && log "stage figures4 ok -- ALL DONE 4"
