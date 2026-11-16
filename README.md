# CS663 Project — Automatic Pixel-Wise Multi-Penalty Regularization for Image Deblurring and Denoising

Team members:
* Kushagra Kushwaha (24b2191)
* Siddhesh Sangram Phadke (24b4502)

A from-scratch Python implementation and empirical study of **MULTI**:

> V. Bortolotti, G. Landi, F. Zama, *An Automatic Pixel-Wise Multi-Penalty Approach to Image Restoration*,
> J. Imaging 2023, 9, 249. https://doi.org/10.3390/jimaging9110249

MULTI gives every pixel its own regularization weight
`λ_i = ||Au−b||² / (N (max_{R×R}(Lu)² + ε))`. Flat regions get smoothed strongly and edges are preserved.
The model is solved by a fixed-point loop on λ. Each step uses a Newton Projection method whose
Hessian is approximated by a BCCB matrix and inverted with FFTs.

The report is in [`report/report.pdf`](report/report.pdf).

## Layout

```
src/                     library (no experiment logic)
  operators.py           PSFs, periodic BCCB operators (blur, Laplacian, gradient) via FFT, degradation
  multi.py               MULTI (Algorithm 1) + modified Newton Projection solver,
                         noise-floor safeguard against λ-collapse, automatic ε rules (our extensions)
  baselines.py           TIKH (Laplacian Tikhonov, NP), TV (ADMM), TGV² (primal-dual / Chambolle–Pock)
  metrics.py             RE, ISNR, MSSIM, PSNR
  data.py                builds data/images/*.png from data/raw (run once)
experiments/
  common.py              problem generation, log-scale golden-section parameter tuning
  run_main.py            Exp. 1: 4 images × 2 blurs × 3 noise levels, all methods (oracle-tuned)
  run_ablation.py        Exp. 3: ε sweeps (plain / safeguarded), R sweep, convergence histories
  finalize_main.py       refines oracle ε from the dense sweep, calibrates κ, re-runs MULTI-auto
  scale_eps.py           scale-aware ε rule, leave-one-image-out evaluation (offline)
  run_auto_scale.py      MULTI-autoS (scale-aware ε) on the main and Kodak problems
  run_kodak.py           Exp. 2: 12 unseen Kodak images, frozen κ (δ = 1% by default)
  retune_kodak_multi.py  re-tunes oracle MULTI on Kodak with the full NP budget (bug fix, see below)
  kodak_auto_rules.py    MULTI-auto vs MULTI-autoS on Kodak at δ = 0.5% and 2.5% (held out)
  archive_auto.py        saves MULTI-auto results of one safeguard variant before re-running
  run_color.py           Extension: colour images (channel-wise)
  make_figures.py        every figure / LaTeX table / number macro in the report
data/raw                 original downloads (Kodak PNGs, RestoreTools satellite.mat)
data/images              the grayscale test images actually used
results/                 JSON metrics + .npz restorations; run_chain.sh = resumable pipeline
report/                  LaTeX source, figures, tables, numbers.tex, build.sh, report.pdf
```

## Reproducing

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install numpy scipy matplotlib pillow scikit-image

PY="nice -n 19 .venv/bin/python"        # all scripts take --workers (we used 3 to keep a laptop cool)
$PY src/data.py                          # build test images
$PY experiments/run_main.py              # Exp. 1 (oracle tuning of all methods)
$PY experiments/run_ablation.py --only eps,conv
$PY experiments/run_ablation.py --only epssafe,epsfloor
$PY experiments/finalize_main.py         # oracle refinement + kappa + MULTI-auto (noise floor)
$PY experiments/run_ablation.py --only R
$PY experiments/run_kodak.py
$PY experiments/run_color.py
$PY experiments/scale_eps.py
$PY experiments/run_auto_scale.py
$PY experiments/kodak_auto_rules.py
$PY experiments/make_figures.py
report/build.sh                          # pdflatex + bibtex -> report/report.pdf
```

`results/run_chain*.sh` run the post-`run_main` stages in order and skip the ones already done.
Note: the first Kodak run tuned oracle MULTI with a 2000-NP budget but reported a 20000-NP run,
which lets λ-collapse finish; `common.runner` now tunes with the full budget, and
`retune_kodak_multi.py` redoes those entries (a fresh `run_kodak.py` no longer needs it).
Total compute is a few hours of single-threaded work per stage; MULTI on the satellite image
is the slowest case (the non-negativity constraint is active on 90% of its pixels).

Minimal usage of the library:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from operators import Operators, disk_psf, degrade
from multi import multi, estimate_noise_sigma

u = ...                                   # grayscale image in [0, 1]
ops = Operators(disk_psf(5), u.shape)     # out-of-focus blur, periodic BCs
b = degrade(u, ops, 1e-2, np.random.default_rng(0))   # 1% Gaussian noise
sig = estimate_noise_sigma(b)             # MAD noise estimate, no ground truth
kappa = float(open("results/kappa.txt").read())
x, lam, info = multi(b, ops, eps=kappa * sig**2, R=5, sigma=sig,   # MULTI-auto
                     safeguard="floor",   # noise floor in the λ rule (prevents λ-collapse)
                     max_total_np=20000)
# MULTI-autoS: eps from the curvature scale of a cheap pilot restoration
from multi import scale_eps
eps_s = scale_eps(b, ops, 5, q=0.5, c=7.5, sigma=sig)
```

## Data

The paper's test images (galaxy, mri, leopard, elaine) are not freely redistributable, so we
replaced each one with a freely available image of the same kind:

| paper     | ours        | source |
|-----------|-------------|--------|
| galaxy    | `satellite` | RestoreTools test data (Nagy et al.), https://www.cs.emory.edu/~nagy/RestoreTools/ |
| mri       | `ct_head`   | scikit-image `data.brain()` slice 5 |
| leopard   | `cat`       | scikit-image `data.chelsea()` (fur + whiskers) |
| elaine    | `portrait`  | Kodak `kodim04`, 512×512 crop, https://r0k.us/graphics/kodak/ |

The generalization set has 12 further Kodak images (grayscale, 256×256).

## Main findings

See the report. In short: MULTI beats Tikhonov with the same penalty almost everywhere and is
best under Gaussian blur, while TGV wins under out-of-focus blur. The automatic λ update can
*collapse* (semi-convergence); flooring the residual in the λ rule at the MAD noise level fixes it
(a discrepancy-principle stop also prevents it but halts adaptation too early). The best ε depends
on the image content much more than on the noise level, so a scale-aware ε (MULTI-autoS) beats
ε = κσ̂².
