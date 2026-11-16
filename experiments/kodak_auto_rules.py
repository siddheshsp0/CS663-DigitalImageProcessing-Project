"""
Held-out test of the two automatic eps rules on Kodak at the noise levels where
they differ (delta = 0.5% and 2.5%; at 1% both give similar eps because kappa
is effectively calibrated there).

  MULTI-auto : eps = kappa * sigma_hat^2                 (kappa from results/kappa.txt)
  MULTI-autoS: eps = c * quantile_q(S(u_pilot))           (q, c from scale_eps.json)

Both use the noise-floor safeguard and the full NP budget.  No oracle tuning
(cheap: two MULTI runs per problem).  Writes results/ablation/kodak_auto_rules.json.

Usage:  python experiments/kodak_auto_rules.py [--workers 3]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import common as C
from multi import scale_eps
from run_auto_scale import pick_rule

OUT = os.path.join(C.RESULTS, "ablation", "kodak_auto_rules.json")


def task(args):
    name, blur, delta, kappa, q, c = args
    u, b, ops = C.make_problem(name, blur, delta)
    sig = C.estimate_noise_sigma(b)
    rec = {"image": name, "blur": blur, "delta": delta, "sigma_hat": sig,
           "blurred": C.all_metrics(b, u, b)}
    for key, eps in (("MULTI-auto", kappa * sig ** 2),
                     ("MULTI-autoS", scale_eps(b, ops, 5, q=q, c=c, sigma=sig))):
        x, lam, info = C.multi(b, ops, eps, R=5, u_true=u, max_total_np=C.FULL_BUDGET,
                               sigma=sig, safeguard=C.AUTO_SAFEGUARD)
        rec[key] = {"eps": float(eps), "metrics": C.all_metrics(x, u, b),
                    "iters": f"{info['outer_iters']}({info['total_np_iters']})"}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--deltas", default="0.005,0.025")
    args = ap.parse_args()
    kappa = C.get_kappa()
    q, c = pick_rule()
    tasks = [(n, bl, float(d), kappa, q, c) for n in C.data.KODAK for bl in C.BLURS
             for d in args.deltas.split(",")]
    out = []
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(task, t) for t in tasks]):
            r = f.result()
            out.append(r)
            print(r["image"], r["blur"], r["delta"],
                  "auto %.4f autoS %.4f" % (r["MULTI-auto"]["metrics"]["RE"],
                                            r["MULTI-autoS"]["metrics"]["RE"]), flush=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
