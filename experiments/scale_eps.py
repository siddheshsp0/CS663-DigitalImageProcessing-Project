"""
Extension 2: a scale-aware automatic eps.

The eps sweep shows that the oracle eps of MULTI depends mainly on the image
(its curvature scale) and much less on the noise level, so eps = kappa*sigma^2
cannot be right in general.  Here eps = c * quantile_q(S(u_pilot)), where
u_pilot is a discrepancy-principle Tikhonov restoration (no ground truth).

Evaluation is *offline*, on the RE(eps) curves of the safeguarded (floor) eps sweep
(interpolated in log eps), with leave-one-image-out (LOIO) calibration:
c (and q) are fitted on 3 images and tested on the 4th.  The kappa*sigma^2
rule is evaluated in exactly the same way for a fair comparison.

Usage: python experiments/scale_eps.py    (single process, ~1 min)
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # keep the laptop load low
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import common as C
from multi import curvature_quantiles

ABL = os.path.join(C.RESULTS, "ablation")
QS = (0.5, 0.75, 0.9, 0.95, 0.99)
CGRID = np.logspace(-4, 5, 361)


def loss_at(sweep, eps):
    e = np.array([r["eps"] for r in sweep["rows"]])
    re = np.array([r["RE"] for r in sweep["rows"]])
    v = np.interp(np.log(np.clip(eps, e[0], e[-1])), np.log(e), re)
    return v / re.min() - 1


def fit(probs, feat, sweeps):
    """c minimising mean relative RE loss of eps = c*feat over probs."""
    L = np.array([[loss_at(sweeps[p], c * feat[p]) for c in CGRID] for p in probs])
    i = int(np.argmin(L.mean(0)))
    return CGRID[i], L.mean(0)[i]


def main():
    import sys
    sweep = sys.argv[1] if len(sys.argv) > 1 else "eps_sweep_floor.json"
    safe = {(s["image"], s["blur"], s["delta"]): s
            for s in json.load(open(os.path.join(ABL, sweep)))}
    feats = {}
    for key in safe:
        n, bl, d = key
        u, b, ops = C.make_problem(n, bl, d)
        R = C.R_DEFAULT.get(n, 5)
        sig = C.estimate_noise_sigma(b)
        Q, lam = curvature_quantiles(b, ops, R, sig, QS)
        feats[key] = {"sigma2": sig ** 2, **{f"q{q}": Q[q] for q in QS}, "pilot_lam": lam}
        print(key, " ".join(f"{k}={v:.3g}" for k, v in feats[key].items()), flush=True)

    rules = ["sigma2"] + [f"q{q}" for q in QS]
    images = sorted({k[0] for k in safe})
    out = {"rules": {}, "features": {f"{k[0]}_{k[1]}_{k[2]:g}": v for k, v in feats.items()}}
    for rule in rules:
        f = {k: feats[k][rule] for k in safe}
        c_all, l_all = fit(list(safe), f, safe)
        loio = {}
        for img in images:
            train = [k for k in safe if k[0] != img]
            test = [k for k in safe if k[0] == img]
            c, _ = fit(train, f, safe)
            loio[img] = {"c": float(c), "test_losses": [float(loss_at(safe[k], c * f[k])) for k in test]}
        all_test = np.concatenate([v["test_losses"] for v in loio.values()])
        out["rules"][rule] = {"c_all": float(c_all), "fit_loss_all": float(l_all),
                              "loio": loio, "loio_mean": float(all_test.mean()),
                              "loio_median": float(np.median(all_test)),
                              "loio_max": float(all_test.max())}
        print(f"{rule:8s} c_all={c_all:9.3g} fit={l_all*100:6.2f}%  LOIO mean={all_test.mean()*100:6.2f}% "
              f"median={np.median(all_test)*100:6.2f}% max={all_test.max()*100:7.2f}%", flush=True)
    json.dump(out, open(os.path.join(ABL, "scale_eps.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
