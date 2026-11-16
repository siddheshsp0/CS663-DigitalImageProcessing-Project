"""Quality measures used in the paper (RE, ISNR, MSSIM) plus PSNR."""
import numpy as np
from skimage.metrics import structural_similarity


def rel_err(u_hat, u):
    return np.linalg.norm(u_hat - u) / np.linalg.norm(u)


def isnr(u_hat, u, b):
    """ISNR = 20 log10( ||b - u|| / ||u_hat - u|| )  (dB)."""
    return 20 * np.log10(np.linalg.norm(b - u) / np.linalg.norm(u_hat - u))


def mssim(u_hat, u):
    rng = float(u.max() - u.min())
    return structural_similarity(u_hat, u, data_range=rng, gaussian_weights=True,
                                 sigma=1.5, use_sample_covariance=False)


def psnr(u_hat, u):
    peak = float(u.max())
    return 10 * np.log10(peak ** 2 / np.mean((u_hat - u) ** 2))


def all_metrics(u_hat, u, b):
    return {"RE": rel_err(u_hat, u), "ISNR": isnr(u_hat, u, b),
            "MSSIM": mssim(u_hat, u), "PSNR": psnr(u_hat, u)}
