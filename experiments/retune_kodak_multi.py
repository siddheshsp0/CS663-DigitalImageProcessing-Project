"""
Re-tune oracle MULTI on the Kodak problems with the full NP budget.

The first Kodak run tuned eps with a 2000-NP budget but reported a 20000-NP
run; the small budget cuts lambda-collapse short, so the chosen eps collapsed
in the reported run (fixed in common.runner).  This script redoes only the
MULTI entry of results/kodak/*.json.

Usage:  python experiments/retune_kodak_multi.py [--workers 3]
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import common as C
import numpy as np


def task(path):
    meta = json.load(open(path))
    if meta["MULTI"].get("tuned_full_budget"):
        return os.path.basename(path), None, meta["MULTI"]["metrics"]["RE"]
    P = meta["_problem"]
    u, b, ops = C.make_problem(P["image"], P["blur"], P["delta"])
    run, (lo, hi) = C.runner("MULTI", u, b, ops, P["R"])
    p, x, info, re, nev = C.tune(run, lo, hi)
    old = meta["MULTI"]["metrics"]["RE"]
    meta["MULTI"] = {"param": p, "metrics": C.all_metrics(x, u, b), "n_evals": nev,
                     "time": info["time"], "iters": f"{info['outer_iters']}({info['total_np_iters']})",
                     "lam_norm": float(np.linalg.norm(info["lam"])),
                     "budget_hit": info.get("budget_hit"), "tuned_full_budget": True,
                     "re_budget2000_tuning": old}
    json.dump(meta, open(path, "w"), indent=1, default=float)
    return os.path.basename(path), old, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(C.RESULTS, "kodak", "*.json")))
    # out-of-focus first: those are the slow (collapsing) ones
    paths.sort(key=lambda p: "_oof_" not in p)
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(task, p) for p in paths]):
            n, old, new = f.result()
            print(f"{n:28s} MULTI RE {old} -> {new:.4f}", flush=True)


if __name__ == "__main__":
    main()
