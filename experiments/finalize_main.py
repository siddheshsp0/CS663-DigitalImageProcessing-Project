"""
Finalise Experiment 1 after the eps sweeps (run_ablation.py) are available.

 1. Oracle MULTI refinement: the RE(eps) curve of MULTI is not unimodal (see
    the eps sweep), so the local golden-section search can miss the global
    optimum.  For every problem we also refine around the best eps of the
    dense 21-point sweep (full NP budget) and keep the better of the two.
 2. kappa calibration for MULTI-auto: kappa minimising the mean relative RE
    loss  RE(kappa*sigma_hat^2)/min_eps RE  over the 24 problems, read off the
    *safeguarded* eps sweep.  Written to results/kappa.txt.
 3. Re-run MULTI-auto (auto eps + discrepancy stop) with that kappa.

Usage:  python experiments/finalize_main.py [--workers 6]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C
import run_main as RM

ABL = os.path.join(C.RESULTS, "ablation")


def refine_task(args):
    name, blur, delta, eps_grid_best = args
    path = os.path.join(RM.OUT, RM.pid(name, blur, delta))
    meta = json.load(open(path + ".json"))
    u, b, ops = C.make_problem(name, blur, delta)
    R = meta["_problem"]["R"]
    re_old = meta["MULTI"]["metrics"]["RE"]

    def run(p):
        x, lam, info = C.multi(b, ops, p, R=R, max_total_np=C.FULL_BUDGET)
        info = dict(info); info["lam"] = lam
        return x, info, C.rel_err(x, u)
    step = 10 ** 0.2                                  # one sweep grid step
    p, x, info, re, nev = C.tune(run, eps_grid_best / step, eps_grid_best * step,
                                 n_coarse=3, n_golden=5)
    if re < re_old:
        meta["MULTI"].update({
            "param": p, "metrics": C.all_metrics(x, u, b), "time": info["time"],
            "iters": f"{info['outer_iters']}({info['total_np_iters']})",
            "lam_norm": float(np.linalg.norm(info["lam"])),
            "budget_hit": info["budget_hit"], "refined_from_sweep": True,
            "re_golden_only": re_old})
        z = dict(np.load(path + ".npz"))
        z["u_MULTI"] = x
        z["lam_MULTI"] = info["lam"]
        np.savez_compressed(path + ".npz", **z)
        json.dump(meta, open(path + ".json", "w"), indent=1, default=float)
        return RM.pid(name, blur, delta), re_old, re
    return RM.pid(name, blur, delta), re_old, re_old


def calibrate_kappa(sweep):
    kgrid = np.logspace(-1, 5, 121)
    losses = []
    for s in sweep:
        eps = np.array([r["eps"] for r in s["rows"]])
        re = np.array([r["RE"] for r in s["rows"]])
        e = kgrid * s["sigma_hat"] ** 2
        r = np.interp(np.log(e), np.log(eps), re)
        losses.append(r / re.min() - 1)
    losses = np.array(losses)
    mean_loss = losses.mean(0)
    i = int(np.argmin(mean_loss))
    return float(kgrid[i]), kgrid, mean_loss, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-refine", action="store_true")
    ap.add_argument("--sweep", default="eps_sweep_floor.json",
                    help="sweep used to calibrate kappa (must match C.AUTO_SAFEGUARD)")
    args = ap.parse_args()
    plain = json.load(open(os.path.join(ABL, "eps_sweep.json")))
    safe = json.load(open(os.path.join(ABL, args.sweep)))

    # 1. oracle refinement
    tasks = []
    for s in plain:
        i = int(np.argmin([r["RE"] for r in s["rows"]]))
        tasks.append((s["image"], s["blur"], s["delta"], s["rows"][i]["eps"]))
    if not args.skip_refine:
        with ProcessPoolExecutor(args.workers) as ex:
            for pid_, a, b_ in ex.map(refine_task, tasks):
                print(f"refine {pid_:28s} RE {a:.4f} -> {b_:.4f}", flush=True)

    # 2. kappa
    kappa, kgrid, mean_loss, losses = calibrate_kappa(safe)
    with open(C.KAPPA_FILE, "w") as f:
        f.write(f"{kappa:.4f}\n")
    json.dump({"kgrid": kgrid, "mean_loss": mean_loss, "losses": losses,
               "kappa": kappa}, open(os.path.join(ABL, "kappa_calibration.json"), "w"),
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"kappa = {kappa:.2f}  (mean RE loss {mean_loss.min() * 100:.2f}%)", flush=True)

    # 3. MULTI-auto
    probs = [(n, bl, d) for n in C.data.MAIN for bl in C.BLURS for d in C.DELTAS]
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(RM.phase3_task, [(n, bl, d, kappa) for n, bl, d in probs]):
            print("auto", r, flush=True)


if __name__ == "__main__":
    main()
