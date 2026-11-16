"""
Linear operators used throughout the project.

All operators assume PERIODIC boundary conditions, so every convolution is a
BCCB (block-circulant with circulant blocks) matrix and is diagonalised by the
2-D DFT.  This is exactly the setting used in the paper (Section 2, eq. 12),
and it is what makes the Hessian approximation H = A^T A + mu L^T L invertible
in O(N log N) with FFTs.
"""
import numpy as np
import scipy.fft as sfft


# --------------------------------------------------------------------------
# Point spread functions
# --------------------------------------------------------------------------
def gaussian_psf(size=31, var=2.0):
    """Isotropic Gaussian PSF with the given variance (paper: variance 2).
    Equivalent of RestoreTools' psfGauss(size, sqrt(var))."""
    r = np.arange(size) - size // 2
    X, Y = np.meshgrid(r, r)
    psf = np.exp(-(X ** 2 + Y ** 2) / (2.0 * var))
    return psf / psf.sum()


def disk_psf(radius=5, supersample=33):
    """Out-of-focus (pill-box) PSF of the given radius, with anti-aliased
    boundary pixels, i.e. the equivalent of MATLAB's fspecial('disk', radius).
    Each pixel's weight is the fraction of its area inside the circle,
    estimated by supersampling."""
    size = 2 * radius + 1
    s = supersample
    off = (np.arange(s) + 0.5) / s - 0.5          # sub-pixel offsets
    psf = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            yy = (i - radius) + off[:, None]
            xx = (j - radius) + off[None, :]
            psf[i, j] = np.mean(xx ** 2 + yy ** 2 <= radius ** 2)
    return psf / psf.sum()


def psf2otf(psf, shape):
    """Zero-pad a (small, centred) PSF to `shape`, circularly shift its centre
    to pixel (0,0) and return its 2-D real DFT (the eigenvalues of the BCCB
    matrix; only the non-redundant half-spectrum is stored since all our
    images and kernels are real)."""
    pad = np.zeros(shape)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return sfft.rfft2(pad)


# --------------------------------------------------------------------------
# FFT helpers
# --------------------------------------------------------------------------
def fft2(x):
    """Real-input 2-D FFT (half spectrum)."""
    return sfft.rfft2(x)


def ifft2r(X, shape):
    """Inverse of fft2 for a real image of the given shape."""
    return sfft.irfft2(X, s=shape)


# --------------------------------------------------------------------------
# Operator bundle
# --------------------------------------------------------------------------
LAPLACIAN_KERNEL = np.array([[0., 1., 0.],
                             [1., -4., 1.],
                             [0., 1., 0.]])


class Operators:
    """Blur A, discrete Laplacian L and forward-difference gradient D, all
    periodic, together with their DFT eigenvalues."""

    def __init__(self, psf, shape):
        self.shape = shape
        self.N = shape[0] * shape[1]
        self.Ahat = psf2otf(psf, shape)               # eigenvalues of A
        self.A2 = np.abs(self.Ahat) ** 2              # eigenvalues of A^T A
        self.Lhat = np.real(psf2otf(LAPLACIAN_KERNEL, shape))  # real, <= 0
        self.L2 = self.Lhat ** 2                      # eigenvalues of L^T L
        # forward differences (periodic):  (Dx u)_ij = u_i,j+1 - u_ij
        dx = np.zeros(shape); dx[0, 0] = -1; dx[0, -1] = 1
        dy = np.zeros(shape); dy[0, 0] = -1; dy[-1, 0] = 1
        self.Dxhat = sfft.rfft2(dx)
        self.Dyhat = sfft.rfft2(dy)
        self.D2 = np.abs(self.Dxhat) ** 2 + np.abs(self.Dyhat) ** 2  # -Laplacian eigs

    # spectral helpers ------------------------------------------------------
    def F(self, x):
        return fft2(x)

    def Fi(self, X):
        return ifft2r(X, self.shape)

    # blur ------------------------------------------------------------------
    def A(self, u):
        return ifft2r(self.Ahat * fft2(u), self.shape)

    def At(self, r):
        return ifft2r(np.conj(self.Ahat) * fft2(r), self.shape)

    # Laplacian (symmetric, L^T = L) --------------------------------------
    def L(self, u):
        # direct stencil is faster than an FFT for a 3x3 kernel
        return (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
                np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4.0 * u)

    Lt = L

    # gradient / divergence -------------------------------------------------
    @staticmethod
    def grad(u):
        gx = np.roll(u, -1, axis=1) - u
        gy = np.roll(u, -1, axis=0) - u
        return gx, gy

    @staticmethod
    def gradT(px, py):
        """Adjoint of grad (= minus the backward-difference divergence)."""
        return (np.roll(px, 1, axis=1) - px) + (np.roll(py, 1, axis=0) - py)


# --------------------------------------------------------------------------
# Degradation model  b = A u + e,   ||e|| / ||A u|| = delta
# --------------------------------------------------------------------------
def degrade(u, ops, delta, rng):
    Au = ops.A(u)
    n = rng.standard_normal(u.shape)
    e = delta * np.linalg.norm(Au) * n / np.linalg.norm(n)
    return Au + e
