"""Shared experiment utilities: problem generation, parameter tuning, runners."""
import os
import sys
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from operators import Operators, gaussian_psf, disk_psf, degrade  # noqa: E402
from multi import multi, auto_eps, estimate_noise_sigma              # noqa: E402
from baselines import tikhonov, tv_admm, tgv_pdhg                    # noqa: E402
from metrics import all_metrics, rel_err                             # noqa: E402
import data                                                          # noqa: E402

RESULTS = os.path.join(ROOT, "results")
BLURS = {"oof": "Out-of-focus (r=5)", "gauss": "Gaussian (var=2)"}
DELTAS = [2.5e-2, 1e-2, 5e-3]

# neighbourhood size: 5x5 everywhere except the astronomy image (paper: 3x3
# for 'galaxy', whose sparse bright features are only a few pixels wide)
R_DEFAULT = {"satellite": 3}

# MULTI-auto: eps = KAPPA * sigma_hat^2 plus the discrepancy-principle stop.
# KAPPA is calibrated on the 24 main problems (finalize_main.py, minimising the
# mean RE loss on the safeguarded eps sweep) and frozen before the Kodak run.
KAPPA_FILE = os.path.join(RESULTS, "kappa.txt")

# work budget (total NP iterations) for the final reported MULTI runs
FULL_BUDGET = 20000
# lambda-collapse safeguard used by MULTI-auto (see report): "floor" floors the
# noise-energy estimate of the lambda rule at N*sigma_hat^2 ("stop" = discrepancy stop)
AUTO_SAFEGUARD = "floor"


def get_kappa():
    if os.path.exists(KAPPA_FILE):
        return float(open(KAPPA_FILE).read().strip())
    return 30.0


def make_psf(blur):
    return disk_psf(5) if blur == "oof" else gaussian_psf(31, 2.0)


def make_problem(name, blur, delta):
    u = data.load(name)
    ops = Operators(make_psf(blur), u.shape)
    seed = zlib.crc32(f"{name}-{blur}-{delta:g}".encode())
    b = degrade(u, ops, delta, np.random.default_rng(seed))
    return u, b, ops


# --------------------------------------------------------------------------
# 1-D parameter tuning on a log scale (minimise relative error), mimicking
# the paper's "parameters chosen heuristically by minimising RE".
# --------------------------------------------------------------------------
def tune(run, lo, hi, n_coarse=6, n_golden=6, max_expand=3):
    """run(p) -> (u, info, re).  Returns best (p, u, info, re, n_evals)."""
    cache = {}

    def f(lp):
        key = round(lp, 6)
        if key not in cache:
            cache[key] = run(10.0 ** lp)
        return cache[key][2]

    llo, lhi = np.log10(lo), np.log10(hi)
    for _ in range(max_expand + 1):
        grid = np.linspace(llo, lhi, n_coarse)
        vals = [f(x) for x in grid]
        i = int(np.argmin(vals))
        h = grid[1] - grid[0]
        if i == 0:
            llo, lhi = llo - (n_coarse - 2) * h, llo + h
        elif i == n_coarse - 1:
            llo, lhi = lhi - h, lhi + (n_coarse - 2) * h
        else:
            break
    a, c = grid[max(i - 1, 0)], grid[min(i + 1, n_coarse - 1)]
    gr = (np.sqrt(5) - 1) / 2
    x1, x2 = c - gr * (c - a), a + gr * (c - a)
    for _ in range(n_golden):
        if f(x1) < f(x2):
            c, x2 = x2, x1
            x1 = c - gr * (c - a)
        else:
            a, x1 = x1, x2
            x2 = a + gr * (c - a)
    best = min(cache, key=lambda k: cache[k][2])
    u, info, re = cache[best]
    return 10.0 ** best, u, info, re, len(cache)


# --------------------------------------------------------------------------
# Method runners: each returns (u, info, re) for a given parameter
# --------------------------------------------------------------------------
def runner(method, u_true, b, ops, R=5):
    if method == "TIKH":
        def run(p):
            x, info = tikhonov(b, ops, p)
            return x, info, rel_err(x, u_true)
        return run, (1e-4, 1e-1)
    if method == "TV":
        def run(p):
            x, info = tv_admm(b, ops, p)
            return x, info, rel_err(x, u_true)
        return run, (1e-6, 1e-2)
    if method == "TGV":
        def run(p):
            x, info = tgv_pdhg(b, ops, p)
            return x, info, rel_err(x, u_true)
        return run, (1e-6, 1e-2)
    if method == "MULTI":
        def run(p):
            # tune with the same NP budget as the reported run: a smaller budget
            # truncates lambda-collapse and would select an eps that collapses later
            x, lam, info = multi(b, ops, p, R=R, max_total_np=FULL_BUDGET)
            info = dict(info); info["lam"] = lam
            return x, info, rel_err(x, u_true)
        return run, (1e-5, 3e-2)
    raise ValueError(method)


def run_all_methods(u, b, ops, R, methods=("TIKH", "TV", "TGV", "MULTI"),
                    kappa=None):
    """Tune every method; also run automatic MULTI (no tuning).  Returns a
    dict method -> {param, metrics, iters, time, u, (lam, history)}."""
    out = {}
    for m in methods:
        run, (lo, hi) = runner(m, u, b, ops, R)
        p, x, info, re, nev = tune(run, lo, hi)
        rec = {"param": p, "metrics": all_metrics(x, u, b), "u": x,
               "n_evals": nev, "time": info["time"]}
        if m == "MULTI":
            rec["iters"] = f"{info['outer_iters']}({info['total_np_iters']})"
            rec["lam"] = info["lam"]
            rec["lam_norm"] = float(np.linalg.norm(info["lam"]))
        else:
            rec["iters"] = str(info["iters"])
        out[m] = rec
    # automatic MULTI: eps from estimated noise level, no reference image used
    kappa = get_kappa() if kappa is None else kappa
    sig = estimate_noise_sigma(b)
    eps = kappa * sig ** 2
    x, lam, info = multi(b, ops, eps, R=R, u_true=u, max_total_np=FULL_BUDGET,
                         sigma=sig, safeguard=AUTO_SAFEGUARD)
    out["MULTI-auto"] = {"param": eps, "metrics": all_metrics(x, u, b), "u": x,
                         "iters": f"{info['outer_iters']}({info['total_np_iters']})",
                         "time": info["time"], "lam": lam,
                         "lam_norm": float(np.linalg.norm(lam)),
                         "history": {k: info[k] for k in
                                     ("lam_norm", "res_norm", "rel_err", "obj", "pg", "np_iters")},
                         "sigma_hat": estimate_noise_sigma(b)}
    return out


def save_problem(path, u, b, res):
    arrays = {"u_true": u, "b": b}
    meta = {}
    for m, r in res.items():
        arrays[f"u_{m}"] = r["u"]
        if "lam" in r:
            arrays[f"lam_{m}"] = r["lam"]
        meta[m] = {k: v for k, v in r.items() if k not in ("u", "lam")}
    np.savez_compressed(path, **arrays)
    return meta
