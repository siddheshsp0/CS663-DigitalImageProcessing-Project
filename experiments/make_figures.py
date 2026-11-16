"""
Generate every figure and LaTeX table of the report from results/.

Usage:  python experiments/make_figures.py
"""
import glob
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage

import common as C

FIG = os.path.join(C.ROOT, "report", "figures")
TAB = os.path.join(C.ROOT, "report", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

METHODS = ["TIKH", "TV", "TGV", "MULTI", "MULTI-auto"]
# fixed categorical colour per method (never re-assigned between figures)
COL = {"MULTI": "#2a78d6", "TGV": "#eb6834", "TV": "#1baf7a",
       "TIKH": "#eda100", "MULTI-auto": "#e87ba4", "MULTI-autoS": "#8a5cc2"}
MARK = {"MULTI": "o", "TGV": "s", "TV": "^", "TIKH": "v", "MULTI-auto": "D", "MULTI-autoS": "P"}
AUTOS = "MULTI-autoS"   # MULTI-auto with the scale-aware eps (run_auto_scale.py)


def with_s(D, ms=None):
    """METHODS plus MULTI-autoS when every problem of D has it."""
    ms = list(METHODS if ms is None else ms)
    return ms + [AUTOS] if D and all(AUTOS in m for m in D.values()) else ms
LABEL = {"satellite": "satellite", "ct_head": "ct\\_head", "cat": "cat",
         "portrait": "portrait"}
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "font.family": "serif",
})


def pid(n, bl, d):
    return f"{n}_{bl}_{d:g}"


def load_main():
    out = {}
    for f in glob.glob(os.path.join(C.RESULTS, "main", "*.json")):
        m = json.load(open(f))
        p = m["_problem"]
        out[(p["image"], p["blur"], p["delta"])] = m
    return out


def npz(n, bl, d):
    return np.load(os.path.join(C.RESULTS, "main", pid(n, bl, d) + ".npz"))


def sci(x, digits=4):
    """LaTeX scientific notation like the paper: 8.2096 x 10^{-2}."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{digits}f}\\times 10^{{{e}}}$"


def imshow(ax, x, title=None, vmin=0, vmax=1, cmap="gray"):
    ax.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=INK)


# ==========================================================================
# Tables
# ==========================================================================
def table_main(M):
    """One LaTeX table per image in the paper's format (Tables 2-5)."""
    for n in C.data.MAIN:
        lines = [r"\begin{tabular}{llllllll}", r"\toprule",
                 r"Blur & $\delta$ & Model & RE & ISNR & MSSIM & Iters & $\lambda$ / $\epsilon$ \\",
                 r"\midrule"]
        for bl in C.BLURS:
            for di, d in enumerate(C.DELTAS):
                m = M[(n, bl, d)]
                best = {k: (min if k == "RE" else max)(
                    m[x]["metrics"][k] for x in METHODS) for k in ("RE", "ISNR", "MSSIM")}
                for j, x in enumerate(METHODS):
                    met = m[x]["metrics"]
                    cells = []
                    for k in ("RE", "ISNR", "MSSIM"):
                        s = f"{met[k]:.4f}" if k != "ISNR" else f"{met[k]:.3f}"
                        cells.append(r"\textbf{" + s + "}" if np.isclose(met[k], best[k]) else s)
                    blur_cell = (r"\multirow{15}{*}{" + C.BLURS[bl].split(" (")[0] + "}") if (di == 0 and j == 0) else ""
                    d_cell = (r"\multirow{5}{*}{" + sci(d, 1).replace(".0", "") + "}") if j == 0 else ""
                    param = sci(m[x]["param"], 2)
                    if x.startswith("MULTI"):
                        param = r"$\epsilon$=" + param
                    lines.append(f"{blur_cell} & {d_cell} & {x} & " + " & ".join(cells)
                                 + f" & {m[x]['iters']} & {param} \\\\")
                if not (bl == "gauss" and d == C.DELTAS[-1]):
                    lines.append(r"\cmidrule{2-8}" if d != C.DELTAS[-1] else r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, f"main_{n}.tex"), "w").write("\n".join(lines))


def table_summary(M, K):
    """Average metrics and win counts for the main set and the Kodak set."""
    rows = []
    for setname, D in ((f"Main ({len(M)})", M), (f"Kodak ({len(K)})", K)):
        if not D:
            continue
        keys = list(D)
        ms = with_s(D)
        for x in ms:
            re = np.mean([D[k][x]["metrics"]["RE"] for k in keys])
            isnr = np.mean([D[k][x]["metrics"]["ISNR"] for k in keys])
            ss = np.mean([D[k][x]["metrics"]["MSSIM"] for k in keys])
            ps = np.mean([D[k][x]["metrics"]["PSNR"] for k in keys])
            # wins among the four tuned methods + auto (best ISNR)
            wins = sum(max(ms, key=lambda y: D[k][y]["metrics"]["ISNR"]) == x for k in keys)
            wins_ss = sum(max(ms, key=lambda y: D[k][y]["metrics"]["MSSIM"]) == x for k in keys)
            t = np.median([D[k][x]["time"] for k in keys])
            rows.append((setname, x, re, isnr, ps, ss, wins, wins_ss, t, len(keys)))
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Set & Model & RE & ISNR (dB) & PSNR (dB) & MSSIM & best ISNR & best MSSIM & time (s) \\",
             r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r[0] != last:
            lines.append(r"\midrule")
        sc = r[0] if r[0] != last else ""
        lines.append(f"{sc} & {r[1]} & {r[2]:.4f} & {r[3]:.3f} & {r[4]:.2f} & {r[5]:.4f} & "
                     f"{r[6]}/{r[9]} & {r[7]}/{r[9]} & {r[8]:.2f} \\\\")
        last = r[0]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "summary.tex"), "w").write("\n".join(lines))
    return rows


# ==========================================================================
# Figures
# ==========================================================================
def fig_testimages():
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for a, n in zip(ax, C.data.MAIN):
        x = C.data.load(n)
        imshow(a, x, f"{n} ({x.shape[0]}$\\times${x.shape[1]})")
    fig.savefig(os.path.join(FIG, "test_images.pdf"))
    plt.close(fig)
    fig, ax = plt.subplots(2, 6, figsize=(7.2, 2.6))
    for a, n in zip(ax.flat, C.data.KODAK):
        imshow(a, C.data.load(n), n)
    fig.savefig(os.path.join(FIG, "kodak_images.pdf"))
    plt.close(fig)
    # PSFs
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.8))
    for a, bl in zip(ax, ("oof", "gauss")):
        p = C.make_psf(bl)
        c = p.shape[0] // 2
        p = p[c - 7:c + 8, c - 7:c + 8]
        imshow(a, p, C.BLURS[bl], vmin=0, vmax=p.max(), cmap="Blues")
    fig.savefig(os.path.join(FIG, "psfs.pdf"))
    plt.close(fig)


def fig_restorations(M, blur="oof", delta=1e-2):
    """Original / blurred / TIKH / TV / TGV / MULTI for every image."""
    cols = ["u_true", "b", "u_TIKH", "u_TV", "u_TGV", "u_MULTI"]
    names = ["Original", "Blurred+noisy", "TIKH", "TV", "TGV", "MULTI"]
    fig, ax = plt.subplots(4, 6, figsize=(7.4, 5.2))
    for i, n in enumerate(C.data.MAIN):
        z = npz(n, blur, delta)
        m = M[(n, blur, delta)]
        for j, (c, t) in enumerate(zip(cols, names)):
            x = z[c]
            if c == "b":
                sub = f"ISNR 0 / SSIM {m['_problem']['blurred']['MSSIM']:.3f}"
            elif c == "u_true":
                sub = n
            else:
                k = c[2:]
                sub = f"{m[k]['metrics']['ISNR']:.2f} dB / {m[k]['metrics']['MSSIM']:.3f}"
            imshow(ax[i, j], np.clip(x, 0, 1), t if i == 0 else None)
            ax[i, j].set_xlabel(sub, fontsize=6.5, color=INK2, labelpad=1)
    fig.subplots_adjust(wspace=0.03, hspace=0.18)
    fig.savefig(os.path.join(FIG, f"restorations_{blur}.pdf"))
    plt.close(fig)


CROPS = {  # (row, col, size) chosen to show smooth areas + fine detail
    "satellite": (70, 70, 110),
    "ct_head": (30, 80, 100),
    "cat": (60, 40, 120),
    "portrait": (110, 150, 220),
}


def fig_zooms(M, blur="oof", delta=1e-2):
    cols = ["u_true", "u_TV", "u_TGV", "u_MULTI", "u_MULTI-auto"]
    names = ["Original", "TV", "TGV", "MULTI", "MULTI-auto"]
    fig, ax = plt.subplots(4, 5, figsize=(7.2, 5.9))
    for i, n in enumerate(C.data.MAIN):
        z = npz(n, blur, delta)
        r, c, s = CROPS[n]
        vmax = 1.0
        if n == "satellite":
            vmax = float(z["u_true"][r:r + s, c:c + s].max())
        for j, (k, t) in enumerate(zip(cols, names)):
            imshow(ax[i, j], np.clip(z[k][r:r + s, c:c + s], 0, 1),
                   t if i == 0 else None, vmax=vmax)
        ax[i, 0].set_ylabel(n, color=INK)
    fig.subplots_adjust(wspace=0.03, hspace=0.04)
    fig.savefig(os.path.join(FIG, f"zooms_{blur}.pdf"))
    plt.close(fig)


def fig_errormaps(M, blur="oof", delta=1e-2):
    cols = ["u_TIKH", "u_TV", "u_TGV", "u_MULTI"]
    fig, ax = plt.subplots(4, 4, figsize=(6.0, 6.2))
    for i, n in enumerate(C.data.MAIN):
        z = npz(n, blur, delta)
        errs = [np.abs(z[k] - z["u_true"]) for k in cols]
        vmax = np.percentile(np.concatenate([e.ravel() for e in errs]), 99.5)
        for j, (k, e) in enumerate(zip(cols, errs)):
            imshow(ax[i, j], e, k[2:] if i == 0 else None, vmin=0, vmax=vmax, cmap="Blues")
        ax[i, 0].set_ylabel(n, color=INK)
    fig.subplots_adjust(wspace=0.03, hspace=0.04)
    fig.savefig(os.path.join(FIG, f"errormaps_{blur}.pdf"))
    plt.close(fig)


def fig_lambda_maps(blur="oof", delta=1e-2):
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.2))
    for a, n in zip(ax, C.data.MAIN):
        lam = npz(n, blur, delta)["lam_MULTI"]
        im = a.imshow(lam, cmap="Blues_r", norm=LogNorm(vmin=lam.min(), vmax=lam.max()))
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(n, color=INK)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
    fig.savefig(os.path.join(FIG, f"lambda_maps_{blur}.pdf"))
    plt.close(fig)


def fig_profile(M, blur="oof", delta=1e-2):
    """1-D line profile through a smooth ramp + an edge to expose staircasing."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for a, (n, row, c0, c1) in zip(axs, (("portrait", 300, 120, 330), ("ct_head", 128, 20, 128))):
        z = npz(n, blur, delta)
        xs = np.arange(c0, c1)
        a.plot(xs, z["u_true"][row, c0:c1], color=INK, lw=1.2, label="Original")
        for k in ("TV", "TGV", "MULTI"):
            a.plot(xs, z["u_" + k][row, c0:c1], color=COL[k], lw=1.4, label=k)
        a.set_title(f"{n}: row {row}", color=INK)
        a.set_xlabel("column"); a.set_ylabel("intensity")
    axs[0].legend(ncol=4, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.25))
    fig.savefig(os.path.join(FIG, f"profiles_{blur}.pdf"))
    plt.close(fig)
    # image showing where the profiles are
    fig, axs = plt.subplots(1, 2, figsize=(3.6, 1.8))
    for a, (n, row, c0, c1) in zip(axs, (("portrait", 300, 120, 330), ("ct_head", 128, 20, 128))):
        imshow(a, C.data.load(n), n)
        a.plot([c0, c1], [row, row], color=COL["TGV"], lw=1.2)
    fig.savefig(os.path.join(FIG, "profiles_location.pdf"))
    plt.close(fig)


def region_masks(u, edge_q=0.85, flat_q=0.5):
    gx = ndimage.sobel(u, axis=1, mode="wrap")
    gy = ndimage.sobel(u, axis=0, mode="wrap")
    g = np.hypot(gx, gy)
    g = ndimage.gaussian_filter(g, 1.0, mode="wrap")
    return g >= np.quantile(g, edge_q), g <= np.quantile(g, flat_q)


def fig_regions(M):
    """RMSE restricted to edge pixels vs flat pixels (all 24 problems,
    expressed relative to TGV)."""
    keys = sorted(M)
    rat = {k: {"edge": [], "flat": []} for k in METHODS}
    abs_ = {k: {"edge": [], "flat": []} for k in METHODS}
    for key in keys:
        z = npz(*key)
        u = z["u_true"]
        E, F = region_masks(u)
        ref = {}
        for reg, mask in (("edge", E), ("flat", F)):
            ref[reg] = np.sqrt(np.mean((z["u_TGV"][mask] - u[mask]) ** 2))
        for k in METHODS:
            for reg, mask in (("edge", E), ("flat", F)):
                e = np.sqrt(np.mean((z["u_" + k][mask] - u[mask]) ** 2))
                rat[k][reg].append(e / ref[reg])
                abs_[k][reg].append(e)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    w = 0.16
    x = np.arange(2)
    for i, k in enumerate(METHODS):
        vals = [np.exp(np.mean(np.log(rat[k][r]))) for r in ("edge", "flat")]
        bars = ax.bar(x + (i - 2) * w, vals, w * 0.88, color=COL[k], label=k)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.5, color=INK2, rotation=90)
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_xticks(x, ["edge pixels (top 15% |∇u|)", "flat pixels (bottom 50% |∇u|)"])
    ax.set_ylabel("RMSE relative to TGV\n(geometric mean, 24 problems)")
    top = max(np.exp(np.mean(np.log(rat[k][r]))) for k in METHODS for r in ("edge", "flat"))
    ax.set_ylim(0.75, max(1.3, top + 0.08))
    ax.legend(ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(os.path.join(FIG, "region_rmse.pdf"))
    plt.close(fig)
    return {k: {r: float(np.exp(np.mean(np.log(rat[k][r])))) for r in ("edge", "flat")} for k in METHODS}


def fig_eps_sweep():
    path = os.path.join(C.RESULTS, "ablation", "eps_sweep.json")
    if not os.path.exists(path):
        return None
    S = json.load(open(path))
    kappa = C.get_kappa()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    dcol = {2.5e-2: "#2a78d6", 1e-2: "#eb6834", 5e-3: "#1baf7a"}
    icol = dict(zip(C.data.MAIN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    stats = []
    for s in S:
        eps = np.array([r["eps"] for r in s["rows"]])
        re = np.array([r["RE"] for r in s["rows"]])
        ls = "-" if s["blur"] == "oof" else "--"
        if s["image"] == "cat":
            axs[0].plot(eps, re, ls, color=dcol[s["delta"]], lw=1.4,
                        label=f"δ={s['delta']:g}, {s['blur']}")
        i = int(np.argmin(re))
        axs[1].plot(s["sigma_hat"] ** 2, eps[i], "o" if s["blur"] == "oof" else "s",
                    color=icol[s["image"]], ms=5, mec="white", mew=0.6)
        e_auto = kappa * s["sigma_hat"] ** 2
        re_auto = np.interp(np.log(e_auto), np.log(eps), re)
        stats.append({"image": s["image"], "blur": s["blur"], "delta": s["delta"],
                      "eps_opt": eps[i], "ratio_opt": eps[i] / s["sigma_hat"] ** 2,
                      "loss_auto": re_auto / re.min() - 1,
                      "sigma_hat": s["sigma_hat"], "sigma_true": s["sigma_true"]})
    axs[0].set_xscale("log"); axs[0].set_xlabel("ε"); axs[0].set_ylabel("RE")
    axs[0].set_title("cat: RE vs ε (plain MULTI)", color=INK)
    axs[0].legend(fontsize=5.5, loc="upper left")
    s2 = np.logspace(-6, -3, 10)
    axs[1].plot(s2, kappa * s2, color=INK2, lw=0.9, ls="--")
    axs[1].text(1.2e-6, kappa * 1.2e-6 * 2.5, f"ε = {kappa:.0f} σ̂²", fontsize=7, color=INK2, rotation=20)
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("estimated noise variance σ̂²"); axs[1].set_ylabel("oracle ε (sweep)")
    axs[1].set_title("best ε per problem (24)", color=INK)
    for n, c in icol.items():
        axs[1].plot([], [], "o", color=c, label=n)
    axs[1].plot([], [], "o", color=INK2, label="out-of-focus")
    axs[1].plot([], [], "s", color=INK2, label="Gaussian")
    axs[1].legend(fontsize=5.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eps_sweep.pdf"))
    plt.close(fig)
    return stats


def fig_lambda_collapse():
    """Semi-convergence of plain MULTI-auto (no stopping rule): once the
    residual drops below the noise level, lambda shrinks, the next solve fits
    more noise, the residual drops further ... (positive feedback)."""
    path = os.path.join(C.RESULTS, "ablation", "auto_plain_kappa78.json")
    if not os.path.exists(path):
        return None
    P = json.load(open(path))
    cases = [("cat_oof_0.005", "#2a78d6"), ("portrait_oof_0.005", "#eb6834"),
             ("ct_head_oof_0.005", "#1baf7a")]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.3))
    out = {}
    for n, c in cases:
        if n not in P:
            continue
        h = P[n]["history"]
        k = np.arange(1, len(h["rel_err"]) + 1)
        noise = np.sqrt(P[n]["N"]) * P[n]["sigma_hat"]
        res = np.array(h["res_norm"][-len(k):])
        lab = n.replace("_0.005", "").replace("_", " ")
        axs[0].plot(k, h["rel_err"], color=c, label=lab, lw=1.5)
        axs[1].plot(k, res / noise, color=c, lw=1.5)
        lam = np.array(h["lam_norm"])
        axs[2].plot(np.arange(len(lam)), lam / lam[0], color=c, lw=1.5)
        # first iteration where the discrepancy principle would stop
        hit = np.where(res <= noise)[0]
        if len(hit):
            j = hit[0]
            axs[0].plot(k[j], h["rel_err"][j], "*", color=c, ms=11, mec=INK, mew=0.7, zorder=5)
        out[n] = {"re_min": float(min(h["rel_err"])), "re_final": float(h["rel_err"][-1]),
                  "k_min": int(np.argmin(h["rel_err"]) + 1),
                  "k_disc": int(hit[0] + 1) if len(hit) else None}
    axs[0].set_title("RE vs outer iteration", color=INK); axs[0].set_ylabel("RE")
    axs[0].plot([], [], "*", color="white", mec=INK, ms=9, label="discrepancy stop")
    axs[0].legend(fontsize=6.5)
    axs[1].axhline(1.0, color=INK2, lw=0.8, ls="--")
    axs[1].text(9, 1.08, "noise level √N σ̂", fontsize=6.5, color=INK2, va="bottom")
    axs[1].set_title("‖Au−b‖ / (√N σ̂)", color=INK)
    axs[2].set_yscale("log"); axs[2].set_title("‖λ⁽ᵏ⁾‖ / ‖λ⁽⁰⁾‖", color=INK)
    for a in axs:
        a.set_xlabel("outer iteration k")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lambda_collapse.pdf"))
    plt.close(fig)
    return out


def fig_eps_safe_and_kappa():
    """(a) RE(eps) for plain MULTI and the two safeguards on the collapse case,
    (b) RE(eps)/min_plain for the floor safeguard on all 24 problems,
    (c) kappa calibration curve on the floor sweep."""
    ld = lambda f: {(s["image"], s["blur"], s["delta"]): s
                    for s in json.load(open(os.path.join(C.RESULTS, "ablation", f)))}
    try:
        plain, stop, floor = ld("eps_sweep.json"), ld("eps_sweep_safe.json"), ld("eps_sweep_floor.json")
    except FileNotFoundError:
        return None
    pk = os.path.join(C.RESULTS, "ablation", "kappa_calibration.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.4))
    icol = dict(zip(C.data.MAIN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    curve = lambda s: (np.array([r["eps"] for r in s["rows"]]), np.array([r["RE"] for r in s["rows"]]))
    stats = []
    for key in sorted(plain):
        eps, rp = curve(plain[key]); _, rs = curve(stop[key]); _, rf = curve(floor[key])
        if key == ("cat", "oof", 5e-3):
            axs[0].plot(eps, rp, color=COL["MULTI"], lw=1.5, label="plain (Tol only)")
            axs[0].plot(eps, rs, color=INK2, lw=1.2, ls=":", label="discrepancy stop")
            axs[0].plot(eps, rf, color=COL["MULTI-auto"], lw=1.6, ls="--", label="noise floor (ours)")
        axs[1].plot(eps / plain[key]["sigma_hat"] ** 2, rf / rp.min(), color=icol[key[0]], lw=0.9, alpha=0.75)
        stats.append({"problem": pid(*key), "min_plain": rp.min(), "min_stop": rs.min(),
                      "min_floor": rf.min(), "worst_plain": rp.max() / rp.min(),
                      "worst_stop": rs.max() / rp.min(), "worst_floor": rf.max() / rp.min()})
    axs[0].set_xscale("log"); axs[0].set_xlabel("ε"); axs[0].set_ylabel("RE")
    axs[0].set_title("cat, out-of-focus, δ=0.005", color=INK); axs[0].legend(fontsize=6)
    axs[1].axhline(1.0, color=INK2, lw=0.8)
    axs[1].set_xscale("log"); axs[1].set_ylim(0.95, 1.4)
    axs[1].set_xlabel("ε / σ̂²"); axs[1].set_ylabel("RE / min RE of plain MULTI")
    axs[1].set_title("noise-floor safeguard, 24 problems", color=INK)
    for n, c in icol.items():
        axs[1].plot([], [], color=c, label=n)
    axs[1].legend(fontsize=5.5, loc="upper left")
    if os.path.exists(pk):
        K = json.load(open(pk))
        kg = np.array(K["kgrid"]); L = np.array(K["losses"])
        axs[2].plot(kg, 100 * np.median(L, 0), color=INK2, lw=1.2, ls="--", label="median")
        axs[2].plot(kg, 100 * np.array(K["mean_loss"]), color=COL["MULTI-auto"], lw=1.8, label="mean")
        axs[2].axvline(K["kappa"], color=INK2, lw=0.8)
        axs[1].axvline(K["kappa"], color=INK2, lw=0.8)
        axs[2].set_xscale("log"); axs[2].set_yscale("log")
        axs[2].text(K["kappa"] * 1.15, 0.6 * axs[2].get_ylim()[1], f"κ={K['kappa']:.0f}",
                    fontsize=7, color=INK2)
        axs[2].set_xlabel("κ"); axs[2].set_ylabel("RE loss vs oracle ε (%)")
        axs[2].set_title("κ calibration (floor sweep)", color=INK); axs[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "eps_safe.pdf"))
    plt.close(fig)
    return stats


def table_auto_variants(M):
    """MULTI-auto with no safeguard / discrepancy stop / noise floor (main set)."""
    srcs = [("none (Tol only)", "auto_plain_kappa78.json"), ("discrepancy stop", "auto_stop.json")]
    rows = []
    for lab, f in srcs:
        p = os.path.join(C.RESULTS, "ablation", f)
        if not os.path.exists(p):
            continue
        A = json.load(open(p))
        isn = {k: v["metrics"]["ISNR"] for k, v in A.items()}
        ss = {k: v["metrics"]["MSSIM"] for k, v in A.items()}
        kap = next(iter(A.values())).get("kappa", float("nan"))
        rows.append((lab, kap, isn, ss))
    A = {pid(*k): m["MULTI-auto"] for k, m in M.items()}
    if all(v.get("safeguard") == "floor" for v in A.values()):
        rows.append(("noise floor (ours)", next(iter(A.values()))["kappa"],
                     {k: v["metrics"]["ISNR"] for k, v in A.items()},
                     {k: v["metrics"]["MSSIM"] for k, v in A.items()}))
    if AUTOS in with_s(M):
        S = {pid(*k): m[AUTOS] for k, m in M.items()}
        rows.append((r"floor + scale-aware $\epsilon$ (autoS)", float("nan"),
                     {k: v["metrics"]["ISNR"] for k, v in S.items()},
                     {k: v["metrics"]["MSSIM"] for k, v in S.items()}))
    orc = {pid(*k): m["MULTI"]["metrics"]["ISNR"] for k, m in M.items()}
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Variant & $\kappa$ & ISNR & worst ISNR & MSSIM & loss (dB) & within 0.5\,dB \\",
             r"\midrule"]
    out = {}
    for lab, kap, isn, ss in rows:
        ks = sorted(isn)
        loss = [orc[k] - isn[k] for k in ks]
        ks_ = "--" if np.isnan(kap) else f"{kap:.0f}"
        lines.append(f"{lab} & {ks_} & {np.mean(list(isn.values())):.2f} & {min(isn.values()):.2f} & "
                     f"{np.mean(list(ss.values())):.3f} & {np.mean(loss):.2f} & {sum(l <= 0.5 for l in loss)}/{len(ks)} \\\\")
        out[lab] = {"mean": np.mean(list(isn.values())), "worst": min(isn.values()), "loss": np.mean(loss),
                    "within": int(sum(l <= 0.5 for l in loss))}
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "auto_variants.tex"), "w").write("\n".join(lines))
    return out


def fig_R_sweep():
    path = os.path.join(C.RESULTS, "ablation", "R_sweep.json")
    if not os.path.exists(path):
        return None
    S = json.load(open(path))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    icol = dict(zip(C.data.MAIN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    for a, bl in zip(axs, C.BLURS):
        for n in C.data.MAIN:
            rows = sorted([r for r in S if r["image"] == n and r["blur"] == bl], key=lambda r: r["R"])
            R = [r["R"] for r in rows]
            re = np.array([r["RE"] for r in rows])
            a.plot(R, re / re.min(), "-o", color=icol[n], label=n, lw=1.6, ms=4)
        a.set_title(f"{C.BLURS[bl]}, δ=1e-2", color=INK)
        a.set_xlabel("neighbourhood size R"); a.set_xticks([1, 3, 5, 9, 15])
    axs[0].set_ylabel("RE / min over R")
    axs[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "R_sweep.pdf"))
    plt.close(fig)
    return S


def fig_convergence():
    path = os.path.join(C.RESULTS, "ablation", "convergence.json")
    if not os.path.exists(path):
        return
    S = json.load(open(path))
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.0))
    icol = dict(zip(C.data.MAIN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    for s in S:
        c = icol[s["image"]]
        k = np.arange(1, len(s["rel_err"]) + 1)
        axs[0].plot(np.arange(len(s["lam_norm"])), np.array(s["lam_norm"]) / s["lam_norm"][0], color=c, lw=1.4, label=s["image"])
        axs[1].plot(k, np.array(s["rel_err"]) / min(s["rel_err"]), color=c, lw=1.4)
        axs[2].plot(k, np.array(s["res_norm"]) / s["res_norm"][-1], color=c, lw=1.4)
        axs[3].plot(k, s["np_iters"], color=c, lw=1.4)
    axs[0].set_title("‖λ⁽ᵏ⁾‖ / ‖λ⁽⁰⁾‖", color=INK); axs[0].set_yscale("log")
    axs[1].set_title("RE / min RE", color=INK)
    axs[2].set_title("‖Au−b‖ / final", color=INK)
    axs[3].set_title("NP iterations", color=INK)
    for a in axs:
        a.set_xlabel("outer iteration k")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.pdf"))
    plt.close(fig)


def load_kodak():
    out = {}
    for f in glob.glob(os.path.join(C.RESULTS, "kodak", "*.json")):
        m = json.load(open(f))
        p = m["_problem"]
        out[(p["image"], p["blur"], p["delta"])] = m
    return out


def fig_kodak(K):
    if not K:
        return None
    keys = sorted(K)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    meths = with_s(K, ["TIKH", "TV", "MULTI", "MULTI-auto"])
    for a, metric, unit in ((axs[0], "ISNR", "dB"), (axs[1], "MSSIM", "")):
        data_ = []
        for k in meths:
            data_.append([K[key][k]["metrics"][metric] - K[key]["TGV"]["metrics"][metric] for key in keys])
        bp = a.boxplot(data_, patch_artist=True, widths=0.55, showfliers=True,
                       medianprops=dict(color=INK, lw=1.2),
                       flierprops=dict(marker=".", ms=3, markeredgecolor=INK2))
        for patch, k in zip(bp["boxes"], meths):
            patch.set_facecolor(COL[k]); patch.set_edgecolor(INK2); patch.set_alpha(0.85)
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xticks(range(1, len(meths) + 1), [m.replace("MULTI-", "") for m in meths])
        a.set_ylabel(f"{metric} − {metric}(TGV) {('(' + unit + ')') if unit else ''}")
        a.set_title(f"Kodak, {len(keys)} problems: {metric} gain over TGV", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "kodak_gain.pdf"))
    plt.close(fig)
    # per-noise-level breakdown table
    km = with_s(K)
    lines = [r"\begin{tabular}{ll" + "r" * len(km) + "}", r"\toprule",
             r"Blur & $\delta$ & " + " & ".join(km) + r" \\", r"\midrule"]
    for bl in C.BLURS:
        for d in C.DELTAS:
            ks = [k for k in keys if k[1] == bl and k[2] == d]
            if not ks:
                continue
            vals = {x: np.mean([K[k][x]["metrics"]["ISNR"] for k in ks]) for x in km}
            best = max(vals.values())
            cells = [(r"\textbf{%.3f}" % v) if np.isclose(v, best) else "%.3f" % v
                     for v in (vals[x] for x in km)]
            lines.append(f"{C.BLURS[bl].split(' (')[0]} & {d:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "kodak_isnr.tex"), "w").write("\n".join(lines))
    return True


def write_numbers(M, K, reg, collapse, safe_stats=None, variants=None):
    """Every number quoted in the report text, as LaTeX macros
    (report/numbers.tex), so the prose never goes stale."""
    out = {}
    tuned = ["TIKH", "TV", "TGV", "MULTI"]
    tag = {"TIKH": "TIKH", "TV": "TV", "TGV": "TGV", "MULTI": "MULTI", "MULTI-auto": "Auto",
           AUTOS: "AutoS"}

    def isnr(D, k, x):
        return D[k][x]["metrics"]["ISNR"]

    def ssim(D, k, x):
        return D[k][x]["metrics"]["MSSIM"]

    def best_of(D, k, ms, f=isnr):
        return max(ms, key=lambda y: f(D, k, y))

    for setname, D in (("Main", M), ("Kodak", K)):
        if not D:
            continue
        keys = sorted(D)
        out[f"n{setname}N"] = len(keys)
        for x in with_s(D):
            out[f"nISNR{setname}{tag[x]}"] = f"{np.mean([isnr(D, k, x) for k in keys]):.2f}"
            out[f"nSSIM{setname}{tag[x]}"] = f"{np.mean([ssim(D, k, x) for k in keys]):.3f}"
            if x in tuned:
                out[f"nWins{setname}{tag[x]}"] = sum(best_of(D, k, tuned) == x for k in keys)
                out[f"nWinsSSIM{setname}{tag[x]}"] = sum(best_of(D, k, tuned, ssim) == x for k in keys)
        for bl, bn in (("gauss", "Gauss"), ("oof", "Oof")):
            ks = [k for k in keys if k[1] == bl]
            out[f"n{setname}GainTGV{bn}"] = f"{np.mean([isnr(D, k, 'MULTI') - isnr(D, k, 'TGV') for k in ks]):+.2f}"
            out[f"n{setname}Wins{bn}"] = sum(best_of(D, k, tuned) == "MULTI" for k in ks)
            out[f"n{setname}N{bn}"] = len(ks)
            out[f"n{setname}AutoGainTGV{bn}"] = f"{np.mean([isnr(D, k, 'MULTI-auto') - isnr(D, k, 'TGV') for k in ks]):+.2f}"
        out[f"n{setname}MultiBeatsTikh"] = sum(isnr(D, k, "MULTI") > isnr(D, k, "TIKH") for k in keys)
        out[f"n{setname}MultiTikhGain"] = f"{np.mean([isnr(D, k, 'MULTI') - isnr(D, k, 'TIKH') for k in keys]):.2f}"
        out[f"n{setname}MultiBeatsTGV"] = sum(isnr(D, k, "MULTI") > isnr(D, k, "TGV") for k in keys)
        out[f"n{setname}AutoBeatsTGV"] = sum(isnr(D, k, "MULTI-auto") > isnr(D, k, "TGV") for k in keys)
        out[f"n{setname}AutoBeatsTV"] = sum(isnr(D, k, "MULTI-auto") > isnr(D, k, "TV") for k in keys)
        out[f"n{setname}AutoBeatsTikh"] = sum(isnr(D, k, "MULTI-auto") > isnr(D, k, "TIKH") for k in keys)
        al = [isnr(D, k, "MULTI") - isnr(D, k, "MULTI-auto") for k in keys]
        out[f"n{setname}AutoLoss"] = f"{np.mean(al):.2f}"
        out[f"n{setname}AutoLossMedian"] = f"{np.median(al):.2f}"
        out[f"n{setname}AutoWithinHalf"] = sum(a <= 0.5 for a in al)
        rel = lambda x: [D[k][x]["metrics"]["RE"] / D[k]["MULTI"]["metrics"]["RE"] - 1 for k in keys]
        out[f"n{setname}AutoRelLoss"] = f"{100 * np.mean(rel('MULTI-auto')):.1f}"
        out[f"n{setname}AutoRelLossMax"] = f"{100 * np.max(rel('MULTI-auto')):.0f}"
        if AUTOS in with_s(D):
            sl = [isnr(D, k, "MULTI") - isnr(D, k, AUTOS) for k in keys]
            out[f"n{setname}AutoSLoss"] = f"{np.mean(sl):.2f}"
            out[f"n{setname}AutoSWithinHalf"] = sum(a <= 0.5 for a in sl)
            out[f"n{setname}AutoSRelLoss"] = f"{100 * np.mean(rel(AUTOS)):.1f}"
            out[f"n{setname}AutoSRelLossMax"] = f"{100 * np.max(rel(AUTOS)):.0f}"
            out[f"n{setname}AutoSBeatsTGV"] = sum(isnr(D, k, AUTOS) > isnr(D, k, "TGV") for k in keys)
            out[f"n{setname}AutoSBeatsAuto"] = sum(isnr(D, k, AUTOS) > isnr(D, k, "MULTI-auto") for k in keys)
            for bl, bn in (("gauss", "Gauss"), ("oof", "Oof")):
                ks = [k for k in keys if k[1] == bl]
                out[f"n{setname}AutoSGainTGV{bn}"] = f"{np.mean([isnr(D, k, AUTOS) - isnr(D, k, 'TGV') for k in ks]):+.2f}"
        for x in with_s(D):
            out[f"nTime{setname}{tag[x]}"] = f"{np.median([D[k][x]['time'] for k in keys]):.1f}"
    for n in C.data.MAIN:
        ks = [k for k in M if k[0] == n]
        out[f"nGainTGV{n.replace('_', '').capitalize()}"] = f"{np.mean([isnr(M, k, 'MULTI') - isnr(M, k, 'TGV') for k in ks]):+.2f}"
    refined = [k for k in M if M[k]["MULTI"].get("refined_from_sweep")]
    out["nRefined"] = len(refined)
    for x in METHODS:
        out[f"nEdge{tag[x]}"] = f"{reg[x]['edge']:.3f}"
        out[f"nFlat{tag[x]}"] = f"{reg[x]['flat']:.3f}"
    if collapse and "cat_oof_0.005" in collapse:
        c = collapse["cat_oof_0.005"]
        out.update({"nCollReMin": f"{c['re_min']:.4f}", "nCollReFinal": f"{c['re_final']:.4f}",
                    "nCollKmin": c["k_min"], "nCollKdisc": c["k_disc"]})
    disc = [k for k in M if M[k]["MULTI-auto"].get("stopped_by_discrepancy")]
    out["nAutoDiscStops"] = len(disc)
    pk = os.path.join(C.RESULTS, "ablation", "kappa_calibration.json")
    if os.path.exists(pk):
        Kc = json.load(open(pk))
        out["nKappa"] = f"{Kc['kappa']:.0f}"
        out["nKappaLoss"] = f"{100 * min(Kc['mean_loss']):.1f}"
    else:
        out["nKappa"] = f"{C.get_kappa():.0f}"
    pe = os.path.join(C.RESULTS, "ablation", "eps_sweep.json")
    if os.path.exists(pe):
        S = json.load(open(pe))
        r_ns, r_s, cv = [], [], {}
        for s in S:
            re = [r["RE"] for r in s["rows"]]
            i = int(np.argmin(re))
            r = s["rows"][i]["eps"] / s["sigma_hat"] ** 2
            (r_s if s["image"] == "satellite" else r_ns).append(r)
            cv.setdefault(s["image"], []).append(np.log10(s["rows"][i]["eps"]))
        out["nRatioLo"] = f"{min(r_ns):.0f}"; out["nRatioHi"] = f"{max(r_ns):.0f}"
        out["nRatioSatLo"] = f"{min(r_s):,.0f}".replace(",", "{,}")
        out["nRatioSatHi"] = f"{max(r_s):,.0f}".replace(",", "{,}")
    pr = os.path.join(C.RESULTS, "ablation", "R_sweep.json")
    if os.path.exists(pr):
        S = json.load(open(pr))
        l5 = []
        for n in C.data.MAIN:
            for bl in C.BLURS:
                rows = [r for r in S if r["image"] == n and r["blur"] == bl]
                if not rows:
                    continue
                m = min(r["RE"] for r in rows)
                r5 = [r["RE"] for r in rows if r["R"] == 5]
                r1 = [r["RE"] for r in rows if r["R"] == 1]
                if r5:
                    l5.append(r5[0] / m - 1)
                if r1:
                    out.setdefault("_r1", []).append(r1[0] / m - 1)
                r15 = [r["RE"] for r in rows if r["R"] == 15]
                if r15:
                    out.setdefault("_r15", []).append(r15[0] / m - 1)
        out["nRfiveLoss"] = f"{100 * np.mean(l5):.1f}"
        out["nRfiveLossMax"] = f"{100 * np.max(l5):.1f}"
        r1 = out.pop("_r1", [])
        if r1:
            out["nRoneLoss"] = f"{100 * np.mean(r1):.1f}"
            out["nRoneLossMax"] = f"{100 * np.max(r1):.0f}"
        r15 = out.pop("_r15", [])
        if r15:
            out["nRfifteenLoss"] = f"{100 * np.mean(r15):.1f}"
            out["nRfifteenLossMax"] = f"{100 * np.max(r15):.0f}"
    pc = os.path.join(C.RESULTS, "color", "color.json")
    if os.path.exists(pc):
        for r in json.load(open(pc)):
            t = {"kodim23": "Parrots", "kodim05": "Bikes"}.get(r["image"], "X")
            for x, xn in (("TGV", "TGV"), ("MULTI-auto", "Auto"), ("blurred", "Blur")):
                out[f"nCol{t}{xn}PSNR"] = f"{r[x]['PSNR']:.2f}"
                out[f"nCol{t}{xn}SSIM"] = f"{r[x]['SSIM']:.3f}"
    ps = os.path.join(C.RESULTS, "ablation", "scale_eps.json")
    if os.path.exists(ps):
        R = json.load(open(ps))["rules"]
        best = min((k for k in R if k != "sigma2"), key=lambda k: R[k]["loio_mean"])
        out["nScaleBest"] = best.replace("q", "")
        for k, nm in (("sigma2", "Sig"), (best, "Best")):
            out[f"nLoio{nm}Mean"] = f"{100 * R[k]['loio_mean']:.1f}"
            out[f"nLoio{nm}Median"] = f"{100 * R[k]['loio_median']:.1f}"
            out[f"nLoio{nm}Max"] = f"{100 * R[k]['loio_max']:.0f}"
            for img, v in R[k]["loio"].items():
                out[f"nLoio{nm}{img.replace('_', '').capitalize()}"] = f"{100 * np.mean(v['test_losses']):.1f}"
        out["nScaleC"] = f"{R[best]['c_all']:.2g}"
    pk2 = os.path.join(C.RESULTS, "ablation", "kodak_auto_rules.json")
    if os.path.exists(pk2):
        KR = json.load(open(pk2))
        for d, dn in ((0.005, "Lo"), (0.025, "Hi")):
            rs = [r for r in KR if np.isclose(r["delta"], d)]
            if not rs:
                continue
            ia = [r["MULTI-auto"]["metrics"]["ISNR"] for r in rs]
            is_ = [r["MULTI-autoS"]["metrics"]["ISNR"] for r in rs]
            rel = [r["MULTI-autoS"]["metrics"]["RE"] / r["MULTI-auto"]["metrics"]["RE"] - 1 for r in rs]
            out[f"nKR{dn}N"] = len(rs)
            out[f"nKR{dn}Auto"] = f"{np.mean(ia):.2f}"
            out[f"nKR{dn}AutoS"] = f"{np.mean(is_):.2f}"
            out[f"nKR{dn}Gain"] = f"{np.mean(np.subtract(is_, ia)):+.2f}"
            out[f"nKR{dn}Wins"] = int(sum(np.greater(is_, ia)))
            out[f"nKR{dn}BestGain"] = f"{np.max(np.subtract(is_, ia)):.2f}"
            out[f"nKR{dn}WorstGain"] = f"{np.min(np.subtract(is_, ia)):+.2f}"
            out[f"nKR{dn}RelRE"] = f"{100 * np.mean(rel):+.1f}"
    if safe_stats:
        g = lambda k: np.array([r[k] for r in safe_stats])
        out["nStopMinLoss"] = f"{100 * np.mean(g('min_stop') / g('min_plain') - 1):.1f}"
        out["nFloorMinLoss"] = f"{100 * np.mean(g('min_floor') / g('min_plain') - 1):+.1f}"
        out["nFloorBetter"] = int(np.sum(g("min_floor") <= g("min_plain") * 1.005))
        out["nWorstPlain"] = f"{np.median(g('worst_plain')):.2f}"
        out["nWorstStop"] = f"{np.median(g('worst_stop')):.2f}"
        out["nWorstFloor"] = f"{np.median(g('worst_floor')):.2f}"
    if variants:
        for lab, nm in (("none (Tol only)", "None"), ("discrepancy stop", "Stop"), ("noise floor (ours)", "Floor"),
                        (r"floor + scale-aware $\epsilon$ (autoS)", "Scale")):
            if lab in variants:
                out[f"nVar{nm}Mean"] = f"{variants[lab]['mean']:.2f}"
                out[f"nVar{nm}Worst"] = f"{variants[lab]['worst']:.2f}"
                out[f"nVar{nm}Loss"] = f"{variants[lab]['loss']:.2f}"
                out[f"nVar{nm}Within"] = variants[lab]["within"]
    # emit; a macro with no data yet prints a visible marker
    import re as _re
    lines = ["% generated by experiments/make_figures.py -- do not edit"]
    for k, v in out.items():
        assert _re.fullmatch(r"n[A-Za-z]+", k), k
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    open(os.path.join(C.ROOT, "report", "numbers.tex"), "w").write("\n".join(lines) + "\n")
    return out


def main():
    M = load_main()
    K = load_kodak()
    fig_testimages()
    table_main(M)
    rows = table_summary(M, K)
    for bl in C.BLURS:
        fig_restorations(M, bl)
        fig_zooms(M, bl)
        fig_errormaps(M, bl)
        fig_lambda_maps(bl)
        fig_profile(M, bl)
    reg = fig_regions(M)
    eps_stats = fig_eps_sweep()
    collapse = fig_lambda_collapse()
    safe_stats = fig_eps_safe_and_kappa()
    variants = table_auto_variants(M)
    fig_R_sweep()
    fig_convergence()
    fig_kodak(K)
    # numbers quoted in the report
    nums = write_numbers(M, K, reg, collapse, safe_stats, variants)
    summary = {"regions": reg, "summary_rows": rows, "collapse": collapse,
               "eps_safe": safe_stats, "auto_variants": variants, "numbers": nums}
    if eps_stats:
        summary["eps"] = eps_stats
    json.dump(summary, open(os.path.join(C.RESULTS, "report_numbers.json"), "w"),
              indent=1, default=float)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
