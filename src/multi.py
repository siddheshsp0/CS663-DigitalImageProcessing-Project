"""
MULTI: automatic pixel-wise multi-penalty image restoration
(Bortolotti, Landi & Zama, J. Imaging 2023, 9, 249).

    u* = argmin_{u >= 0}  1/2 ||A u - b||^2 + 1/2 sum_i lambda_i (L u)_i^2
    lambda_i = ||A u - b||^2 / ( N (S_i + eps) ),   S_i = max_{j in N_i} (L u)_j^2

Outer loop  : Algorithm 1 of the paper (fixed-point on lambda).
Inner solver: modified Newton Projection (NP) method, eqs. (9)-(14), with the
              BCCB Hessian approximation H = A^T A + mean(lambda) L^T L, which
              is inverted with FFTs.
"""
import time
import numpy as np
from scipy.ndimage import maximum_filter



# --------------------------------------------------------------------------
# Regularisation parameters  (eqs. 7, 8 and Algorithm 1 steps 4-5)
# --------------------------------------------------------------------------
def local_curvature(u, ops, R):
    """S_i = max over the R x R neighbourhood of pixel i of (L u)^2."""
    Lu2 = ops.L(u) ** 2
    if R <= 1:
        return Lu2
    return maximum_filter(Lu2, size=R, mode="wrap")


def update_lambda(u, b, ops, eps, R, floor=0.0):
    res2 = max(np.sum((ops.A(u) - b) ** 2), floor)
    S = local_curvature(u, ops, R)
    return res2 / (ops.N * (S + eps))


# --------------------------------------------------------------------------
# Objective of the inner problem
# --------------------------------------------------------------------------
def objective(u, b, lam, ops):
    r = ops.A(u) - b
    Lu = ops.L(u)
    J = 0.5 * np.sum(r * r) + 0.5 * np.sum(lam * Lu * Lu)
    return J, r, Lu


def gradient(r, Lu, lam, ops):
    return ops.At(r) + ops.Lt(lam * Lu)


# --------------------------------------------------------------------------
# Newton Projection inner solver
# --------------------------------------------------------------------------
def newton_projection(u0, b, lam, ops, tol=1e-5, maxit=1000, eps_bar=1e-3,
                      beta=0.5, eta=1e-4, max_backtrack=20, pg_tol=None):
    """Solve  min_{u>=0} 1/2||Au-b||^2 + 1/2 sum lam_i (Lu)_i^2.

    Returns (u, info) where info has 'iters' and per-iteration histories.
    """
    if pg_tol is None:
        pg_tol = tol
    mu = float(np.mean(lam))
    Hhat = ops.A2 + mu * ops.L2             # eigenvalues of BCCB approximation
    Hhat = np.maximum(Hhat, 1e-14)          # safety (only DC can be tiny)

    u = np.maximum(u0, 0.0)
    J, r, Lu = objective(u, b, lam, ops)
    g = gradient(r, Lu, lam, ops)
    pg0 = None
    hist_J, hist_pg = [J], []
    it = 0
    n_fallback = 0
    for it in range(1, maxit + 1):
        # projected gradient  u - [u - g]^+ : measure of stationarity
        pgv = u - np.maximum(u - g, 0.0)
        w = np.linalg.norm(pgv)
        if pg0 is None:
            pg0 = max(w, 1e-300)
        hist_pg.append(w)
        if w / pg0 < pg_tol:
            break

        # active set  I(u) = { i : 0 <= u_i <= eps_k and g_i > 0 }
        eps_k = min(eps_bar, w)
        I = (u <= eps_k) & (g > 0)
        gI = np.where(I, 0.0, g)                       # reduced gradient (9)
        d = ops.Fi(ops.F(gI) / Hhat)                   # H d = g_I       (11)
        p = np.where(I, g, d)                          # search dir.     (13)

        free_dec = np.sum(g[~I] * p[~I])
        if free_dec <= 0:                              # safeguard: fall back
            p = g                                      # to projected gradient
            free_dec = np.sum(g[~I] * g[~I])
            n_fallback += 1

        # Armijo rule along the projection arc                            (14)
        t = 1.0
        accepted = False
        for _ in range(max_backtrack):
            ut = np.maximum(u - t * p, 0.0)
            Jt, rt, Lut = objective(ut, b, lam, ops)
            rhs = eta * (t * free_dec + np.sum(g[I] * (u[I] - ut[I])))
            if J - Jt >= rhs:
                accepted = True
                break
            t *= beta
        if not accepted:                               # no progress possible
            break

        step = np.linalg.norm(ut - u) / max(np.linalg.norm(u), 1e-300)
        u, J, r, Lu = ut, Jt, rt, Lut
        g = gradient(r, Lu, lam, ops)
        hist_J.append(J)
        if step < tol:
            break

    return u, {"iters": it, "J": hist_J, "pg": hist_pg, "fallbacks": n_fallback}


# --------------------------------------------------------------------------
# Outer loop: Algorithm 1
# --------------------------------------------------------------------------
def multi(b, ops, eps, R=5, tol=1e-1, maxit=20, np_tol=1e-5, np_maxit=1000,
          u_true=None, verbose=False, max_total_np=2000, sigma=None, tau=1.0,
          safeguard="stop"):
    """Run MULTI.  Returns (u, lam, info).

    max_total_np caps the total number of NP iterations over all outer
    iterations (a work budget; only reached when the lambda-update collapses
    towards lambda -> 0, see report).

    sigma (optional, our extension): noise standard deviation.  If given, the
    outer loop also stops as soon as the discrepancy principle
    ||A u - b|| <= tau * sqrt(N) * sigma holds.  This prevents the
    'lambda collapse': once the residual drops below the noise level the
    lambda rule (which uses ||Au-b||^2/N as a noise-variance estimate)
    under-estimates lambda, which lowers the residual further, and so on.

    safeguard="floor" (alternative fix): instead of stopping, the noise-energy
    estimate in the lambda rule is floored at the noise level,
    lambda_i = max(||Au-b||^2, tau^2 N sigma^2) / (N (S_i + eps)),
    so the feedback loop cannot start but the iteration keeps adapting S."""
    floor = 0.0
    if sigma is not None and safeguard == "floor":
        floor = (tau * sigma) ** 2 * ops.N
    t0 = time.time()
    u = np.maximum(b.copy(), 0.0)                          # u^(0) = b
    lam = update_lambda(u, b, ops, eps, R, floor)          # eqs. (7)-(8)
    info = {"np_iters": [], "lam_norm": [np.linalg.norm(lam)],
            "res_norm": [], "rel_err": [], "obj": [], "pg": []}
    k = 0
    used = 0
    info["budget_hit"] = False
    for k in range(1, maxit + 1):
        budget = min(np_maxit, max_total_np - used)
        if budget <= 0:
            info["budget_hit"] = True
            k -= 1
            break
        u, npinfo = newton_projection(u, b, lam, ops, tol=np_tol, maxit=budget)
        used += npinfo["iters"]
        lam_new = update_lambda(u, b, ops, eps, R, floor)
        info["np_iters"].append(npinfo["iters"])
        info["fallbacks"] = info.get("fallbacks", 0) + npinfo["fallbacks"]
        info["lam_norm"].append(np.linalg.norm(lam_new))
        info["res_norm"].append(np.linalg.norm(ops.A(u) - b))
        info["obj"].append(npinfo["J"][-1])
        info["pg"].append(npinfo["pg"][-1] if npinfo["pg"] else 0.0)
        if u_true is not None:
            info["rel_err"].append(np.linalg.norm(u - u_true) / np.linalg.norm(u_true))
        if verbose:
            msg = f"  outer {k:2d}: NP iters {npinfo['iters']:4d}  ||lam|| {info['lam_norm'][-1]:.4e}"
            if u_true is not None:
                msg += f"  RE {info['rel_err'][-1]:.4e}"
            print(msg)
        done = np.linalg.norm(lam_new - lam) <= tol * np.linalg.norm(lam_new)
        if (sigma is not None and safeguard == "stop"
                and info["res_norm"][-1] <= tau * np.sqrt(ops.N) * sigma):
            done = True
            info["stopped_by_discrepancy"] = True
        lam = lam_new
        if done:
            break
    info["outer_iters"] = k
    info["total_np_iters"] = int(np.sum(info["np_iters"]))
    info["time"] = time.time() - t0
    return u, lam, info


# --------------------------------------------------------------------------
# Automatic choice of eps (our extension, see report Section 5)
# --------------------------------------------------------------------------
def estimate_noise_sigma(b):
    """Robust noise std estimate (Donoho-Johnstone MAD) from the finest-scale
    diagonal Haar wavelet coefficients of the observed image."""
    h, w = (b.shape[0] // 2) * 2, (b.shape[1] // 2) * 2
    x = b[:h, :w]
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2.0
    return np.median(np.abs(hh)) / 0.6745


def auto_eps(b, kappa=30.0):
    """eps = kappa * sigma_hat^2.

    Rationale: for i.i.d. noise of variance sigma^2 the 5-point Laplacian has
    variance 20 sigma^2, so eps should be measured in units of sigma^2: it is
    the curvature level below which a pixel is treated as 'flat'."""
    s = estimate_noise_sigma(b)
    return kappa * s ** 2


# --------------------------------------------------------------------------
# Scale-aware automatic eps (our second extension)
# --------------------------------------------------------------------------
def pilot_tikhonov(b, ops, sigma, tau=1.0, iters=40):
    """Unconstrained Laplacian-Tikhonov restoration with lambda chosen by the
    discrepancy principle ||A u - b|| = tau sqrt(N) sigma.  Closed form in the
    Fourier domain (BCCB), bisection on log(lambda): costs ~40 FFT solves."""
    B = ops.F(b)
    target = tau * np.sqrt(ops.N) * sigma
    lo, hi = -12.0, 3.0                           # log10(lambda) bracket
    u = b
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lam = 10.0 ** mid
        U = np.conj(ops.Ahat) * B / (ops.A2 + lam * ops.L2 + 1e-14)
        u = ops.Fi(U)
        r = np.linalg.norm(ops.A(u) - b)
        if r > target:
            hi = mid                              # too smooth: decrease lambda
        else:
            lo = mid
    return u, 10.0 ** (0.5 * (lo + hi))


def curvature_quantiles(b, ops, R, sigma=None, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    """Quantiles of S_i = max_{RxR}(L u_pilot)^2 for a discrepancy-principle
    Tikhonov pilot u_pilot.  These describe the curvature scale of the image
    itself (not of the noise), which is what eps has to separate."""
    sigma = estimate_noise_sigma(b) if sigma is None else sigma
    u, lam = pilot_tikhonov(b, ops, sigma)
    S = local_curvature(u, ops, R)
    return {q: float(np.quantile(S, q)) for q in qs}, lam


def scale_eps(b, ops, R, q=0.9, c=1.0, sigma=None):
    """eps = c * quantile_q(S(u_pilot))."""
    Q, _ = curvature_quantiles(b, ops, R, sigma, qs=(q,))
    return c * Q[q]
