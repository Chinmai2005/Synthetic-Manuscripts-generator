"""Physical deformation (warp, curl, folds, skew), lighting and scan simulation."""
import cv2
import numpy as np

from .backgrounds import make_backdrop
from .noise import blur, fbm, smoothstep


def U(rng, rg):
    return float(rng.uniform(rg[0], rg[1]))


def deform_scene(rng, rgb, alpha, label, kind, dcfg, backdrops):
    H, W = alpha.shape
    b = int(U(rng, dcfg["border"]) * min(W, H) + 0.012 * max(W, H))
    Hs, Ws = H + 2 * b, W + 2 * b

    # ---- global skew / perspective (scanner or camera tilt) -------------------
    cx, cy = b + W / 2, b + H / 2
    ang = np.radians(U(rng, dcfg["rotation_deg"]))
    src = np.float32([[b, b], [b + W, b], [b + W, b + H], [b, b + H]])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    dst = (src - [cx, cy]) @ rot.T + [cx, cy]
    dst = (dst + rng.normal(0, dcfg["perspective"] * min(W, H), dst.shape)).astype(np.float32)
    Minv = np.linalg.inv(cv2.getPerspectiveTransform(src, dst))
    gx, gy = np.meshgrid(np.arange(Ws, dtype=np.float32), np.arange(Hs, dtype=np.float32))
    den = Minv[2, 0] * gx + Minv[2, 1] * gy + Minv[2, 2]
    px = (Minv[0, 0] * gx + Minv[0, 1] * gy + Minv[0, 2]) / den - b   # page-frame coords
    py = (Minv[1, 0] * gx + Minv[1, 1] * gy + Minv[1, 2]) / den - b

    dx = np.zeros((Hs, Ws), np.float32)
    dy = np.zeros((Hs, Ws), np.float32)
    shade = np.ones((Hs, Ws), np.float32)

    # ---- smooth surface warp ---------------------------------------------------
    if kind == "palm_leaf":
        A = U(rng, dcfg["palm_bend_px"])
        h = fbm(rng, Hs, Ws, Ws / 1.4, Hs * 4.0, octaves=2) - 0.5
        dy += h * 2 * A
        dx += (fbm(rng, Hs, Ws, Ws / 1.8, Hs * 3.0, octaves=2) - 0.5) * 2 * A * 0.3
        hh = h * 2 * A
    else:
        A = U(rng, dcfg["warp_px"])
        fx = (fbm(rng, Hs, Ws, max(Ws, Hs) / 1.5, octaves=3) - 0.5) * 2 * A
        fy = (fbm(rng, Hs, Ws, max(Ws, Hs) / 1.5, octaves=3) - 0.5) * 2 * A
        dx += fx
        dy += fy
        hh = fy + 0.6 * fx
    gyh, gxh = np.gradient(blur(hh, 6))
    shade += -1.6 * (gxh + gyh) * 0.7

    # ---- folds (paper) ---------------------------------------------------------
    n_folds = int(rng.integers(dcfg["folds"][0], dcfg["folds"][1] + 1)) if kind == "paper" else 0
    for _ in range(n_folds):
        th = (0 if rng.random() < 0.5 else np.pi / 2) + rng.normal(0, 0.22)
        nx, ny = -np.sin(th), np.cos(th)
        tx, ty = np.cos(th), np.sin(th)
        p0x, p0y = rng.uniform(0.25, 0.75) * Ws, rng.uniform(0.25, 0.75) * Hs
        s = (gx - p0x) * nx + (gy - p0y) * ny
        t = (gx - p0x) * tx + (gy - p0y) * ty
        L = rng.uniform(0.5, 1.2) * (Ws if abs(tx) > 0.5 else Hs)
        win = 1 - smoothstep(0.35 * L, 0.5 * L, np.abs(t))
        wf = rng.uniform(5, 14)
        amp = rng.uniform(1.5, 5.0) * win
        step = np.tanh(s / wf) * amp
        dx += nx * step
        dy += ny * step
        crease = np.exp(-(s / (0.45 * wf)) ** 2) * win
        shade += -0.15 * crease * rng.uniform(0.5, 1.0) + 0.05 * np.tanh(s / (3 * wf)) * win

    # ---- edge / corner curl ------------------------------------------------------
    if rng.random() < dcfg["curl_prob"]:
        c = U(rng, dcfg["curl_px"])
        if kind == "palm_leaf" or rng.random() < 0.55:
            edge = int(rng.integers(4))
            zw = rng.uniform(0.05, 0.12) * (min(W, H) if edge in (2, 3) or kind == "paper" else H)
            d = [px, W - px, py, H - py][edge]
            p = np.clip(1 - d / zw, 0, 1) ** 2
            out = [(-1, 0), (1, 0), (0, -1), (0, 1)][edge]
            dx += out[0] * c * p
            dy += out[1] * c * p
            shade += 0.16 * p - 0.14 * np.exp(-((d - 0.85 * zw) / (0.12 * zw)) ** 2) * (d > 0)
        else:
            cxn, cyn = [(0, 0), (W, 0), (W, H), (0, H)][int(rng.integers(4))]
            dd = np.hypot(px - cxn, py - cyn)
            zw = rng.uniform(0.08, 0.2) * min(W, H)
            p = np.clip(1 - dd / zw, 0, 1) ** 2
            ux, uy = (cxn - px) / (dd + 1e-3), (cyn - py) / (dd + 1e-3)
            dx += ux * c * 1.2 * p
            dy += uy * c * 1.2 * p
            shade += 0.18 * p - 0.15 * np.exp(-((dd - 0.8 * zw) / (0.12 * zw)) ** 2)

    mx = (px + dx).astype(np.float32)
    my = (py + dy).astype(np.float32)
    rgb_w = cv2.remap(rgb.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a_w = cv2.remap(alpha.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lab_w = cv2.remap(label, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    rgb_w *= np.clip(shade, 0.55, 1.35)[..., None]
    # thin darker rim = thickness of the leaf / paper edge
    rim = np.clip(a_w - cv2.erode(a_w, np.ones((3, 3), np.uint8)), 0, 1)
    rgb_w *= (1 - 0.35 * rim)[..., None]

    # ---- backdrop + shadow ---------------------------------------------------------
    bd = make_backdrop(rng, Ws, Hs, backdrops)
    off = (int(rng.integers(3, 9)), int(rng.integers(5, 14)))
    sh = np.roll(a_w, (off[1], off[0]), (0, 1))
    shadow = blur(sh, rng.uniform(8, 16)) * rng.uniform(0.35, 0.6)
    img = bd * (1 - shadow[..., None])
    img = img * (1 - a_w[..., None]) + rgb_w * a_w[..., None]
    return img, lab_w, a_w


def scan_effects(rng, img, scfg):
    H, W = img.shape[:2]
    # uneven illumination + vignette
    th = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = ((xx / W - 0.5) * np.cos(th) + (yy / H - 0.5) * np.sin(th)) * rng.uniform(-0.18, 0.18)
    r = np.hypot(xx / W - 0.5, yy / H - 0.5)
    vig = 1 - U(rng, scfg["vignette"]) * (r / 0.7) ** 2
    img = img * (1 + grad)[..., None] * vig[..., None]
    img *= rng.uniform(0.96, 1.04, 3)[None, None]            # colour cast
    sig = U(rng, scfg["blur"])
    img = blur(img.astype(np.float32), sig)
    img += rng.normal(0, U(rng, scfg["noise"]), img.shape).astype(np.float32)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < scfg["jpeg_round_trip"]:
        q = int(rng.integers(62, 90))
        ok, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out)
