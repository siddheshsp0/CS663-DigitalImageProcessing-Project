#!/bin/bash
cd /mnt/data/Siddhesh/College/SEM5/CS663/project
PY="nice -n 19 .venv/bin/python"
log(){ echo "$1 $(date +%H:%M)" >> results/chain.log; }
$PY experiments/retune_kodak_multi.py --workers 3 > results/retune_kodak.log 2>&1 && log "stage kodak-retune ok"
$PY experiments/make_figures.py > results/make_figures.log 2>&1 && log "stage figures3 ok -- ALL DONE 3"
