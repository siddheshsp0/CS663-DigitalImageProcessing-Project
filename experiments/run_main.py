"""
Experiment 1: reproduce the paper's Tables 2-5 protocol on our four test images.

  4 images x 2 blurs x 3 noise levels = 24 problems
  methods: TIKH, TV, TGV, MULTI (param tuned to minimise RE, as in the paper)
           MULTI-auto (eps = kappa * sigma_hat^2, no tuning, no ground truth)

Phase 1  tuned methods (parallel over problems, cached in results/main/)
Phase 2  calibrate kappa = median(eps_opt / sigma_hat^2) over the 24 problems
Phase 3  run MULTI-auto with that kappa

Usage:  python experiments/run_main.py [--workers 10]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import common as C

OUT = os.path.join(C.RESULTS, "main")


def pid(name, blur, delta):
    return f"{name}_{blur}_{delta:g}"


def phase1_task(args):
    name, blur, delta = args
    path = os.path.join(OUT, pid(name, blur, delta))
    if os.path.exists(path + ".json"):
        return pid(name, blur, delta), "cached"
    u, b, ops = C.make_problem(name, blur, delta)
    R = C.R_DEFAULT.get(name, 5)
    res = {}
    for m in ("TIKH", "TV", "TGV", "MULTI"):
        run, (lo, hi) = C.runner(m, u, b, ops, R)
        p, x, info, re, nev = C.tune(run, lo, hi)
        if m == "MULTI":
            # tuning uses a reduced work budget; the reported run uses the
            # paper's full budget (20 outer x 1000 NP iterations)
            x, lam, info = C.multi(b, ops, p, R=R, max_total_np=C.FULL_BUDGET)
            info = dict(info); info["lam"] = lam
        rec = {"param": p, "metrics": C.all_metrics(x, u, b), "u": x,
               "n_evals": nev, "time": info["time"]}
        if m == "MULTI":
            rec["budget_hit"] = info["budget_hit"]
            rec["fallbacks"] = info.get("fallbacks", 0)
            rec["iters"] = f"{info['outer_iters']}({info['total_np_iters']})"
            rec["lam"] = info["lam"]
            rec["lam_norm"] = float(np.linalg.norm(info["lam"]))
        else:
            rec["iters"] = str(info["iters"])
        res[m] = rec
    meta = C.save_problem(path + ".npz", u, b, res)
    meta["_problem"] = {"image": name, "blur": blur, "delta": delta, "R": R,
                        "sigma_hat": C.estimate_noise_sigma(b),
                        "sigma_true": float(np.linalg.norm(b - ops.A(u)) / np.sqrt(u.size)),
                        "blurred": C.all_metrics(b, u, b)}
    json.dump(meta, open(path + ".json", "w"), indent=1, default=float)
    return pid(name, blur, delta), "done"


def phase3_task(args):
    name, blur, delta, kappa = args
    path = os.path.join(OUT, pid(name, blur, delta))
    meta = json.load(open(path + ".json"))
    u, b, ops = C.make_problem(name, blur, delta)
    R = meta["_problem"]["R"]
    sig = C.estimate_noise_sigma(b)
    eps = kappa * sig ** 2
    x, lam, info = C.multi(b, ops, eps, R=R, u_true=u, max_total_np=C.FULL_BUDGET,
                           sigma=sig, safeguard=C.AUTO_SAFEGUARD)
    meta["MULTI-auto"] = {
        "param": eps, "kappa": kappa, "metrics": C.all_metrics(x, u, b),
        "iters": f"{info['outer_iters']}({info['total_np_iters']})",
        "stopped_by_discrepancy": info.get("stopped_by_discrepancy", False),
        "safeguard": C.AUTO_SAFEGUARD,
        "time": info["time"], "lam_norm": float(np.linalg.norm(lam)),
        "history": {k: info[k] for k in ("lam_norm", "res_norm", "rel_err",
                                         "obj", "pg", "np_iters")}}
    z = dict(np.load(path + ".npz"))
    z["u_MULTI-auto"] = x
    z["lam_MULTI-auto"] = lam
    np.savez_compressed(path + ".npz", **z)
    json.dump(meta, open(path + ".json", "w"), indent=1, default=float)
    return pid(name, blur, delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    probs = [(n, bl, d) for n in C.data.MAIN for bl in C.BLURS for d in C.DELTAS]
    # big images first for better load balancing
    probs.sort(key=lambda t: t[0] != "portrait")

    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(phase1_task, p) for p in probs]
        for f in as_completed(futs):
            print("phase1", *f.result(), flush=True)

    # ---- phase 2: kappa calibration
    ratios = []
    for n, bl, d in probs:
        meta = json.load(open(os.path.join(OUT, pid(n, bl, d) + ".json")))
        ratios.append(meta["MULTI"]["param"] / meta["_problem"]["sigma_hat"] ** 2)
    kappa = float(np.median(ratios))
    with open(C.KAPPA_FILE, "w") as f:
        f.write(f"{kappa:.4f}\n")
    print(f"kappa = median(eps_opt/sigma_hat^2) = {kappa:.3f}   "
          f"(range {np.min(ratios):.2f} .. {np.max(ratios):.2f})", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(phase3_task, [(n, bl, d, kappa) for n, bl, d in probs]):
            print("phase3", r, flush=True)


if __name__ == "__main__":
    main()
