"""
Experiment 3: sensitivity / ablation studies for MULTI.

  (a) eps sweep   : RE and MSSIM as a function of eps, for all 24 problems.
                    Plotted against eps and against eps / sigma^2 to test the
                    hypothesis behind our automatic rule (optimal eps ~ sigma^2).
  (b) R sweep     : neighbourhood size R in {1,3,5,7,9,11,15}; eps re-tuned
                    for every R (delta = 1e-2, both blurs, 4 images).
  (c) convergence : full Algorithm-1 history for 20 outer iterations with
                    Tol = 0 (analogue of the paper's Figure 10).

Usage:  python experiments/run_ablation.py [--workers 10]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C

OUT = os.path.join(C.RESULTS, "ablation")
EPS_GRID = np.logspace(-5, -1, 21)
R_GRID = [1, 3, 5, 9, 15]


def eps_task(args):
    name, blur, delta, safe = args          # safe: False | "stop" | "floor"
    u, b, ops = C.make_problem(name, blur, delta)
    R = C.R_DEFAULT.get(name, 5)
    sig = C.estimate_noise_sigma(b) if safe else None
    mode = safe if isinstance(safe, str) else "stop"
    rows = []
    for eps in EPS_GRID:
        x, lam, info = C.multi(b, ops, eps, R=R, sigma=sig, safeguard=mode)
        m = C.all_metrics(x, u, b)
        rows.append({"eps": eps, "RE": m["RE"], "MSSIM": m["MSSIM"],
                     "ISNR": m["ISNR"], "outer": info["outer_iters"],
                     "np": info["total_np_iters"]})
    return {"image": name, "blur": blur, "delta": delta,
            "sigma_hat": C.estimate_noise_sigma(b),
            "sigma_true": float(np.linalg.norm(b - ops.A(u)) / np.sqrt(u.size)),
            "rows": rows}


def r_task(args):
    name, blur, delta, R = args
    u, b, ops = C.make_problem(name, blur, delta)
    run, (lo, hi) = C.runner("MULTI", u, b, ops, R)
    p, x, info, re, nev = C.tune(run, lo, hi)
    m = C.all_metrics(x, u, b)
    return {"image": name, "blur": blur, "delta": delta, "R": R, "eps": p,
            "RE": m["RE"], "MSSIM": m["MSSIM"], "ISNR": m["ISNR"],
            "time": info["time"]}


def conv_task(args):
    name, blur, delta = args
    u, b, ops = C.make_problem(name, blur, delta)
    R = C.R_DEFAULT.get(name, 5)
    meta = json.load(open(os.path.join(C.RESULTS, "main", f"{name}_{blur}_{delta:g}.json")))
    eps = meta["MULTI"]["param"]
    x, lam, info = C.multi(b, ops, eps, R=R, tol=0.0, maxit=20, u_true=u)
    return {"image": name, "blur": blur, "delta": delta, "eps": eps,
            **{k: info[k] for k in ("lam_norm", "res_norm", "rel_err", "obj",
                                    "pg", "np_iters")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", default="eps,epssafe,epsfloor,R,conv")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    todo = args.only.split(",")
    probs = [(n, bl, d) for n in C.data.MAIN for bl in C.BLURS for d in C.DELTAS]
    with ProcessPoolExecutor(args.workers) as ex:
        if "eps" in todo:
            res = list(ex.map(eps_task, [p + (False,) for p in probs]))
            json.dump(res, open(os.path.join(OUT, "eps_sweep.json"), "w"), indent=1, default=float)
            print("eps sweep done", flush=True)
        if "epssafe" in todo:
            res = list(ex.map(eps_task, [p + ("stop",) for p in probs]))
            json.dump(res, open(os.path.join(OUT, "eps_sweep_safe.json"), "w"), indent=1, default=float)
            print("safe eps sweep done", flush=True)
        if "epsfloor" in todo:
            res = list(ex.map(eps_task, [p + ("floor",) for p in probs]))
            json.dump(res, open(os.path.join(OUT, "eps_sweep_floor.json"), "w"), indent=1, default=float)
            print("floor eps sweep done", flush=True)
        if "R" in todo:
            tasks = [(n, bl, 1e-2, R) for n in C.data.MAIN for bl in C.BLURS for R in R_GRID]
            res = list(ex.map(r_task, tasks))
            json.dump(res, open(os.path.join(OUT, "R_sweep.json"), "w"), indent=1, default=float)
            print("R sweep done", flush=True)
        if "conv" in todo:
            tasks = [(n, "oof", 1e-2) for n in C.data.MAIN]
            res = list(ex.map(conv_task, tasks))
            json.dump(res, open(os.path.join(OUT, "convergence.json"), "w"), indent=1, default=float)
            print("convergence done", flush=True)


if __name__ == "__main__":
    main()
