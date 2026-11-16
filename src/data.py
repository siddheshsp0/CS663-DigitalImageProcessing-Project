"""Test images.  Run this file once to build data/images/*.png from data/raw.

The paper's images (galaxy, mri, leopard, elaine) are not publicly
redistributable, so each one is replaced by an image of the same *type*:

  galaxy  -> satellite   (RestoreTools astronomy test image, dark background,
                          thin bright structures)
  mri     -> ct_head     (scikit-image 'brain' CT slice: medical image with
                          large flat regions and sharp anatomical edges)
  leopard -> cat         (scikit-image 'chelsea': fur texture + whiskers)
  elaine  -> portrait    (Kodak kodim04: portrait of a woman with a hat, 512x512)

plus a generalisation set of 12 Kodak images (grayscale, 256x256).
All images are scaled to [0, 1].
"""
import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
IMG = os.path.join(ROOT, "data", "images")

MAIN = ["satellite", "ct_head", "cat", "portrait"]
KODAK = ["kodim01", "kodim02", "kodim03", "kodim05", "kodim07", "kodim08",
         "kodim13", "kodim14", "kodim15", "kodim19", "kodim20", "kodim23"]


def _to_gray(rgb):
    rgb = np.asarray(rgb, dtype=float) / 255.0
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _center_crop(x, h, w):
    H, W = x.shape[:2]
    i, j = (H - h) // 2, (W - w) // 2
    return x[i:i + h, j:j + w]


def _resize(x, size):
    im = Image.fromarray((np.clip(x, 0, 1) * 65535).astype(np.uint16))
    im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 65535.0


def _save(name, x):
    os.makedirs(IMG, exist_ok=True)
    x = np.clip(x, 0, 1)
    Image.fromarray(np.round(x * 255).astype(np.uint8)).save(os.path.join(IMG, name + ".png"))


def build():
    import scipy.io as sio
    from skimage import data as skdata
    # satellite (256x256, float) -------------------------------------------
    s = sio.loadmat(os.path.join(RAW, "satellite.mat"))["x_true"].astype(float)
    _save("satellite", s / s.max())
    # CT head slice (256x256) ------------------------------------------------
    ct = skdata.brain()[5].astype(float)
    _save("ct_head", ct / ct.max())
    # cat: 300x451 -> centre crop 300x300 -> 256x256 -------------------------
    cat = _to_gray(skdata.chelsea())
    _save("cat", _resize(_center_crop(cat, 300, 300), 256))
    # portrait: kodim04 (768x512 portrait) -> 512x512 crop around the face ---
    k4 = _to_gray(Image.open(os.path.join(RAW, "kodim04.png")))
    _save("portrait", k4[60:572, :])
    # Kodak generalisation set: centre square crop, resized to 256x256 -------
    for k in KODAK:
        x = _to_gray(Image.open(os.path.join(RAW, k + ".png")))
        m = min(x.shape)
        _save(k, _resize(_center_crop(x, m, m), 256))


def load(name):
    x = np.asarray(Image.open(os.path.join(IMG, name + ".png")), dtype=float)
    return x / 255.0


if __name__ == "__main__":
    build()
    print("images written to", IMG)
