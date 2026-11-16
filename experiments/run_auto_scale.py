"""
MULTI-autoS: MULTI-auto with the scale-aware eps of Extension 2.

    eps = c * quantile_q(S(u_pilot)),   u_pilot = discrepancy-principle Tikhonov,

with the noise-floor safeguard and the full NP budget, exactly like MULTI-auto
(which uses eps = kappa * sigma_hat^2).  q and c are taken from
results/ablation/scale_eps.json (the rule with the lowest leave-one-image-out
loss, c fitted on the four main images).  For the main images this is therefore
in-sample, for Kodak it is held out.

Adds a "MULTI-autoS" entry to results/main/*.json (and the image to the .npz)
and to results/kodak/*.json.

Usage:  python experiments/run_auto_scale.py [--workers 3] [--sets main,kodak]
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import common as C
import numpy as np
from multi import scale_eps

KEY = "MULTI-autoS"


def pick_rule():
    S = json.load(open(os.path.join(C.RESULTS, "ablation", "scale_eps.json")))
    rules = {k: v for k, v in S["rules"].items() if k.startswith("q")}
    best = min(rules, key=lambda k: rules[k]["loio_mean"])
    return float(best[1:]), rules[best]["c_all"]


def task(args):
    path, q, c = args
    meta = json.load(open(path))
    P = meta["_problem"]
    u, b, ops = C.make_problem(P["image"], P["blur"], P["delta"])
    R = P["R"]
    sig = C.estimate_noise_sigma(b)
    eps = scale_eps(b, ops, R, q=q, c=c, sigma=sig)
    x, lam, info = C.multi(b, ops, eps, R=R, u_true=u, max_total_np=C.FULL_BUDGET,
                           sigma=sig, safeguard=C.AUTO_SAFEGUARD)
    meta[KEY] = {"param": float(eps), "q": q, "c": c, "metrics": C.all_metrics(x, u, b),
                 "iters": f"{info['outer_iters']}({info['total_np_iters']})",
                 "safeguard": C.AUTO_SAFEGUARD, "time": info["time"],
                 "lam_norm": float(np.linalg.norm(lam))}
    npz = path[:-5] + ".npz"
    if os.path.exists(npz):
        z = dict(np.load(npz))
        z["u_" + KEY] = x
        z["lam_" + KEY] = lam
        np.savez_compressed(npz, **z)
    json.dump(meta, open(path, "w"), indent=1, default=float)
    return os.path.basename(path), meta[KEY]["metrics"]["RE"], meta["MULTI-auto"]["metrics"]["RE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sets", default="main,kodak")
    args = ap.parse_args()
    q, c = pick_rule()
    print(f"scale rule: q={q}, c={c:.4g}", flush=True)
    paths = []
    for s in args.sets.split(","):
        paths += sorted(glob.glob(os.path.join(C.RESULTS, s, "*.json")))
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(task, (p, q, c)) for p in paths]):
            n, re_s, re_a = f.result()
            print(f"{n:32s} autoS {re_s:.4f}  auto {re_a:.4f}", flush=True)


if __name__ == "__main__":
    main()
