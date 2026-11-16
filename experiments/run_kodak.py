"""
Experiment 2: generalisation to 12 unseen Kodak images.

  12 images x 2 blurs x 3 noise levels = 72 problems.
  TIKH / TV / TGV / MULTI are tuned per problem using the ground truth
  (best-case, 'oracle' parameters); MULTI-auto uses the kappa frozen after
  the main experiment and therefore never sees the reference image.

Usage:  python experiments/run_kodak.py [--workers 10]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import common as C

OUT = os.path.join(C.RESULTS, "kodak")


def task(args):
    name, blur, delta, kappa = args
    path = os.path.join(OUT, f"{name}_{blur}_{delta:g}.json")
    if os.path.exists(path):
        return name, blur, delta, "cached"
    u, b, ops = C.make_problem(name, blur, delta)
    res = C.run_all_methods(u, b, ops, R=5, kappa=kappa)
    meta = {m: {k: v for k, v in r.items() if k not in ("u", "lam", "history")}
            for m, r in res.items()}
    meta["_problem"] = {"image": name, "blur": blur, "delta": delta, "R": 5,
                        "sigma_hat": C.estimate_noise_sigma(b),
                        "blurred": C.all_metrics(b, u, b)}
    json.dump(meta, open(path, "w"), indent=1, default=float)
    return name, blur, delta, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--deltas", default="0.01",
                    help="comma-separated noise levels (default: 1%% only, to save compute)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    kappa = C.get_kappa()
    assert os.path.exists(C.KAPPA_FILE), "run run_main.py first (calibrates kappa)"
    print("using frozen kappa =", kappa, flush=True)
    tasks = [(n, bl, d, kappa) for n in C.data.KODAK for bl in C.BLURS for d in [float(x) for x in args.deltas.split(",")]]
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(task, t) for t in tasks]):
            print(*f.result(), flush=True)


if __name__ == "__main__":
    main()
