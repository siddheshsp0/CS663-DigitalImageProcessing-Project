"""
Single-parameter baselines, all with the non-negativity constraint u >= 0 and
periodic boundary conditions (same setting as MULTI):

  TIKH : min 1/2||Au-b||^2 + lam/2 ||L u||^2           (Newton Projection)
  TV   : min 1/2||Au-b||^2 + lam TV(u)                 (ADMM)
  TGV  : min 1/2||Au-b||^2 + lam (||grad u - v||_1 + 2 ||E v||_1)   (PDHG)

TIKH uses the *same* Laplacian penalty as MULTI, so MULTI with a constant
lambda reduces exactly to TIKH.  This gives a clean ablation of what the
pixel-wise parameters contribute.
"""
import time
import numpy as np

from multi import newton_projection


# --------------------------------------------------------------------------
def tikhonov(b, ops, lam, tol=1e-5, maxit=300):
    t0 = time.time()
    lamv = np.full(b.shape, lam)
    u, info = newton_projection(b, b, lamv, ops, tol=tol, maxit=maxit)
    return u, {"iters": info["iters"], "time": time.time() - t0}


# --------------------------------------------------------------------------
def tv_admm(b, ops, lam, rho=None, maxit=400, tol=1e-5):
    """Isotropic TV with non-negativity, ADMM with splittings
    z = grad u  and  w = u  (w >= 0)."""
    t0 = time.time()
    if rho is None:
        rho = max(10.0 * lam, 1e-4)
    rho_w = rho
    Atb_hat = np.conj(ops.Ahat) * ops.F(b)
    denom = ops.A2 + rho * ops.D2 + rho_w

    u = np.maximum(b.copy(), 0)
    zx, zy = ops.grad(u)
    yx = np.zeros_like(u); yy = np.zeros_like(u)
    w = u.copy(); yw = np.zeros_like(u)
    it = 0
    for it in range(1, maxit + 1):
        u_old = u
        rhs = Atb_hat + rho * ops.F(ops.gradT(zx - yx, zy - yy)) + rho_w * ops.F(w - yw)
        u = ops.Fi(rhs / denom)
        gx, gy = ops.grad(u)
        # z-update: isotropic shrinkage
        vx, vy = gx + yx, gy + yy
        mag = np.sqrt(vx ** 2 + vy ** 2)
        shrink = np.maximum(1 - (lam / rho) / np.maximum(mag, 1e-12), 0)
        zx, zy = shrink * vx, shrink * vy
        # w-update: projection onto u >= 0
        w = np.maximum(u + yw, 0)
        # dual updates
        yx += gx - zx; yy += gy - zy; yw += u - w
        if np.linalg.norm(u - u_old) < tol * np.linalg.norm(u):
            break
    return np.maximum(w, 0), {"iters": it, "time": time.time() - t0}


# --------------------------------------------------------------------------
def _sym_grad(vx, vy):
    """Symmetrised gradient E v (backward differences), returns 3 comps
    (xx, yy, xy)."""
    exx = vx - np.roll(vx, 1, axis=1)
    eyy = vy - np.roll(vy, 1, axis=0)
    exy = 0.5 * ((vx - np.roll(vx, 1, axis=0)) + (vy - np.roll(vy, 1, axis=1)))
    return exx, eyy, exy


def _sym_gradT(qxx, qyy, qxy):
    """Adjoint of _sym_grad."""
    vx = (qxx - np.roll(qxx, -1, axis=1)) + 0.5 * (qxy - np.roll(qxy, -1, axis=0))
    vy = (qyy - np.roll(qyy, -1, axis=0)) + 0.5 * (qxy - np.roll(qxy, -1, axis=1))
    return vx, vy


def tgv_pdhg(b, ops, lam, ratio=2.0, maxit=2000, tol=2e-6, step_ratio=30.0,
             return_v=False):
    """Second-order TGV with non-negativity, solved with the primal-dual
    hybrid gradient (Chambolle-Pock) method.  The data term is dualised so
    that every proximal map is closed-form:

        min_{u>=0, v}  F1(Au) + a1||grad u - v||_{2,1} + a0||E v||_{F,1}
        F1(z) = 1/2||z - b||^2,   a1 = lam,  a0 = ratio * lam
    """
    t0 = time.time()
    a1, a0 = lam, ratio * lam
    Lnorm2 = 1.0 + 12.0                     # ||K||^2 upper bound (A, grad, E)
    tau = 0.99 * step_ratio / np.sqrt(Lnorm2)
    sigma = 0.99 / (step_ratio * np.sqrt(Lnorm2))

    u = np.maximum(b.copy(), 0); vx = np.zeros_like(u); vy = np.zeros_like(u)
    ub, vbx, vby = u.copy(), vx.copy(), vy.copy()
    y = np.zeros_like(u)                                      # dual of A
    px = np.zeros_like(u); py = np.zeros_like(u)              # dual of grad u - v
    qxx = np.zeros_like(u); qyy = np.zeros_like(u); qxy = np.zeros_like(u)
    it = 0
    for it in range(1, maxit + 1):
        # ---- dual ascent
        y = (y + sigma * ops.A(ub) - sigma * b) / (1 + sigma)
        gx, gy = ops.grad(ub)
        px += sigma * (gx - vbx); py += sigma * (gy - vby)
        n = np.maximum(1.0, np.sqrt(px ** 2 + py ** 2) / a1)
        px /= n; py /= n
        exx, eyy, exy = _sym_grad(vbx, vby)
        qxx += sigma * exx; qyy += sigma * eyy; qxy += sigma * exy
        # dual norm of ||(exx,eyy,exy)||_F = sqrt(exx^2+eyy^2+2exy^2) w.r.t. the
        # plain inner product used by _sym_gradT
        n = np.maximum(1.0, np.sqrt(qxx ** 2 + qyy ** 2 + 0.5 * qxy ** 2) / a0)
        qxx /= n; qyy /= n; qxy /= n
        # ---- primal descent
        u_old, vx_old, vy_old = u, vx, vy
        u = np.maximum(u - tau * (ops.At(y) + ops.gradT(px, py)), 0)
        sx, sy = _sym_gradT(qxx, qyy, qxy)
        vx = vx - tau * (-px + sx)
        vy = vy - tau * (-py + sy)
        # ---- extrapolation
        ub = 2 * u - u_old; vbx = 2 * vx - vx_old; vby = 2 * vy - vy_old
        if it > 50 and np.linalg.norm(u - u_old) < tol * np.linalg.norm(u):
            break
    info = {"iters": it, "time": time.time() - t0}
    if return_v:
        info["v"] = (vx, vy)
    return u, info


def tgv_objective(u, v, b, ops, lam, ratio=2.0):
    vx, vy = v
    gx, gy = ops.grad(u)
    exx, eyy, exy = _sym_grad(vx, vy)
    return (0.5 * np.sum((ops.A(u) - b) ** 2)
            + lam * np.sum(np.sqrt((gx - vx) ** 2 + (gy - vy) ** 2))
            + ratio * lam * np.sum(np.sqrt(exx ** 2 + eyy ** 2 + 2 * exy ** 2)))
