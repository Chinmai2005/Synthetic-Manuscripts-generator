"""Ink artefacts: bleeding, faded strokes, dry-pen skips, smudges, blots, show-through."""
import cv2
import numpy as np

from .noise import blur, fbm, smoothstep


def U(rng, rg):
    return float(rng.uniform(rg[0], rg[1]))


def apply_ink_artifacts(rng, layers, rect, size, acfg, kind, n_labels):
    H, W = layers.alpha.shape
    a = layers.alpha
    k = size / 50.0
    x0, y0, x1, y1 = [int(v) for v in rect]

    # per-line density variation (some lines are written with a drier pen)
    lut = np.ones(256, np.float32)
    lut[1:n_labels + 1] = rng.uniform(0.80, 1.0, n_labels)
    a *= lut[layers.label]

    # faded strokes: low-frequency patches where the ink has almost gone
    fade = fbm(rng, H, W, size * 7, octaves=3)
    patches = smoothstep(rng.uniform(0.58, 0.70), 0.85, fade)
    a *= 1 - patches * U(rng, acfg["fade_strength"])

    # dry-pen skips / flaking ink
    d = fbm(rng, H, W, max(size * 0.33, 3), octaves=2)
    a *= 1 - 0.65 * smoothstep(0.74, 0.9, d) * U(rng, acfg["dropout"])

    # smudges: directional smear of existing ink
    ys, xs = np.nonzero(a > 0.5)
    for _ in range(int(rng.integers(acfg["smudges"][0], acfg["smudges"][1] + 1))):
        if xs.size == 0:
            break
        i = int(rng.integers(xs.size))
        cx, cy = int(xs[i]), int(ys[i])
        L = max(3, int(rng.uniform(0.6, 1.6) * size))
        th = rng.uniform(-0.5, 0.5) + (np.pi / 2 if rng.random() < 0.3 else 0)
        ker = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
        cv2.line(ker, (L, L), (int(L + np.cos(th) * L), int(L + np.sin(th) * L)), 1.0, 1)
        ker /= ker.sum()
        r = int(rng.uniform(1.0, 2.4) * size)
        xa, ya, xb, yb = max(cx - r, 0), max(cy - r, 0), min(cx + r, W), min(cy + r, H)
        if xb - xa < 4 or yb - ya < 4:
            continue
        sub = a[ya:yb, xa:xb]
        sm = cv2.filter2D(sub, -1, ker)
        yy, xx = np.mgrid[ya:yb, xa:xb]
        wgt = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2.0) ** 2))).astype(np.float32)
        a[ya:yb, xa:xb] = np.maximum(sub, np.clip(sm * 1.4, 0, 1) * wgt * rng.uniform(0.35, 0.7))

    # blots / splatter, kept inside the writing area
    for _ in range(int(rng.integers(acfg["blots"][0], acfg["blots"][1] + 1))):
        cx, cy = int(rng.uniform(x0, x1)), int(rng.uniform(y0, y1))
        cv2.circle(a, (cx, cy), max(1, int(rng.uniform(1.5, 5) * k)), float(rng.uniform(0.5, 0.9)), -1, cv2.LINE_AA)

    # ink bleeding into the fibres (feathered halo, stronger along fibres)
    s = acfg["bleed_sigma"] * k * rng.uniform(0.7, 1.4)
    absorb = 0.55 + 0.9 * fbm(rng, H, W, 5, 2.0, octaves=2)
    halo = blur(a, s) * absorb * U(rng, acfg["bleed_strength"]) * (1.2 if kind == "paper" else 0.8)
    a[:] = np.maximum(a, np.clip(halo, 0, 1))
    np.clip(a, 0, 1, out=a)
    return layers


def show_through(rng, alpha, strength):
    """Faint mirrored ink from the reverse side."""
    return blur(np.ascontiguousarray(alpha[:, ::-1]), 2.5) * strength
