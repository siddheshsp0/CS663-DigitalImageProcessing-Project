"""
Extension: colour images.  MULTI-auto applied channel-by-channel (R, G, B),
each channel with its own noise estimate and lambda map, compared with
channel-wise TGV whose single lambda is tuned on the colour RE (oracle).

Usage:  python experiments/run_color.py
"""
import json
import os
import zlib

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

import common as C

OUT = os.path.join(C.RESULTS, "color")
FIG = os.path.join(C.ROOT, "report", "figures")
IMAGES = ["kodim23", "kodim05"]


def load_rgb(name, size=256):
    x = np.asarray(Image.open(os.path.join(C.data.RAW, name + ".png")), dtype=float) / 255
    m = min(x.shape[:2])
    i, j = (x.shape[0] - m) // 2, (x.shape[1] - m) // 2
    x = x[i:i + m, j:j + m]
    im = Image.fromarray(np.round(x * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255


def cssim(a, b):
    return structural_similarity(a, b, channel_axis=2, data_range=1.0,
                                 gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    kappa = C.get_kappa()
    blur, delta = "oof", 1e-2
    res = []
    fig, ax = plt.subplots(len(IMAGES), 4, figsize=(7.2, 1.9 * len(IMAGES)))
    for r, name in enumerate(IMAGES):
        u = load_rgb(name)
        ops = C.Operators(C.make_psf(blur), u.shape[:2])
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        b = np.stack([C.degrade(u[..., c], ops, delta, rng) for c in range(3)], -1)
        # MULTI-auto, per channel
        xm = []
        for c in range(3):
            sig = C.estimate_noise_sigma(b[..., c])
            xm.append(C.multi(b[..., c], ops, kappa * sig ** 2, R=5, sigma=sig,
                              max_total_np=C.FULL_BUDGET, safeguard=C.AUTO_SAFEGUARD)[0])
        xm = np.stack(xm, -1)

        # TGV, per channel, one lambda tuned on colour RE
        def run(p):
            x = np.stack([C.tgv_pdhg(b[..., c], ops, p)[0] for c in range(3)], -1)
            return x, {}, C.rel_err(x, u)
        p, xt, _, _, _ = C.tune(run, 1e-6, 1e-2, n_coarse=5, n_golden=5)
        rec = {"image": name, "tgv_lambda": p}
        for k, x in (("blurred", b), ("TGV", xt), ("MULTI-auto", xm)):
            rec[k] = {"PSNR": 10 * np.log10(1 / np.mean((np.clip(x, 0, 1) - u) ** 2)),
                      "SSIM": cssim(np.clip(x, 0, 1), u), "RE": C.rel_err(x, u)}
        res.append(rec)
        print(json.dumps(rec, default=float), flush=True)
        for j, (t, x) in enumerate((("Original", u), ("Blurred+noisy", b),
                                     ("TGV (oracle λ)", xt), ("MULTI-auto", xm))):
            a = ax[r, j]
            a.imshow(np.clip(x, 0, 1)); a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(t, fontsize=9)
            if j > 0:
                k = {1: "blurred", 2: "TGV", 3: "MULTI-auto"}[j]
                a.set_xlabel(f"{rec[k]['PSNR']:.2f} dB / SSIM {rec[k]['SSIM']:.3f}", fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.12)
    fig.savefig(os.path.join(FIG, "color.pdf"), bbox_inches="tight", dpi=200)
    json.dump(res, open(os.path.join(OUT, "color.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
