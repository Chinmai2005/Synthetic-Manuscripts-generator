"""Small numpy/OpenCV helpers for procedural textures."""
import cv2
import numpy as np


def fbm(rng, h, w, sx, sy=None, octaves=4, persistence=0.5):
    """Fractal value noise normalised to [0, 1]. sx/sy = base feature size in px."""
    sy = sy or sx
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for _ in range(octaves):
        gw = max(2, int(np.ceil(w / max(sx, 1.5))) + 2)
        gh = max(2, int(np.ceil(h / max(sy, 1.5))) + 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
        sx, sy = sx / 2, sy / 2
        if min(sx, sy) < 1.5:
            break
    out /= tot
    lo, hi = float(out.min()), float(out.max())
    return (out - lo) / (hi - lo + 1e-6)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def inside_distance(mask):
    """Distance (px) from each inside pixel to the nearest outside pixel."""
    m = (mask > 0.5).astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5)


def lerp_color(c0, c1, t):
    c0 = np.asarray(c0, np.float32)
    c1 = np.asarray(c1, np.float32)
    return c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]
