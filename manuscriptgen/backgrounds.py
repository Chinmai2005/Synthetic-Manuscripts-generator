"""Procedural manuscript backgrounds: aged handmade paper and palm leaf.

Every function returns a flat (un-deformed) folio:
    rgb   float32 HxWx3 in 0..255
    alpha float32 HxW   (1 = material, 0 = outside the folio / holes)
    holes list of (cx, cy, r) string-holes / worm-holes that text must avoid
"""
import cv2
import numpy as np

from .noise import blur, fbm, inside_distance, lerp_color, smoothstep


def pick(rng, items):
    return items[int(rng.integers(len(items)))]


def uni(rng, lo, hi):
    return float(rng.uniform(lo, hi))


# ----------------------------------------------------------------------------
def folio_mask(rng, W, H, radius, rough, wave=0.0):
    """Anti-aliased rounded-rectangle mask with a rough / deckled edge."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if wave > 0:  # long edges are never perfectly straight
        wx = fbm(rng, 1, W, W / 3.0, 1.0, octaves=2)[0]
        wy = fbm(rng, H, 1, H / 2.0, 1.0, octaves=2)[:, 0]
        yy = yy + (wx[None, :] - 0.5) * 2 * wave
        xx = xx + (wy[:, None] - 0.5) * 2 * wave * 0.6
    cx, cy = W / 2, H / 2
    qx = np.abs(xx - cx) - (W / 2 - 2 - radius)
    qy = np.abs(yy - cy) - (H / 2 - 2 - radius)
    sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius
    n = fbm(rng, H, W, 9, octaves=3) - 0.5
    n2 = fbm(rng, H, W, 2.5, octaves=2) - 0.5
    sdf = sdf + n * rough * 2 + n2 * rough * 0.8
    return np.clip(0.5 - sdf, 0, 1).astype(np.float32)


def add_stains(rng, rgb, W, H, aging, tone):
    """Foxing spots, water-rings and soot blotches."""
    n_fox = int(rng.integers(10, 60) * (0.4 + aging))
    spots = np.zeros((H, W), np.float32)
    for _ in range(n_fox):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(spots, (x, y), int(rng.integers(1, 7)), uni(rng, 0.3, 1.0), -1)
    spots = blur(spots, 1.6) * (0.4 + 0.6 * fbm(rng, H, W, 6))
    for _ in range(int(rng.integers(0, 3) * (0.5 + aging))):
        cx, cy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
        r0 = uni(rng, 0.06, 0.22) * min(W, H)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.hypot(xx - cx, yy - cy) + (fbm(rng, H, W, r0 / 2.0) - 0.5) * r0 * 0.5
        ring = np.exp(-((d - r0) / (0.03 * r0 + 2)) ** 2) * 0.55 + (d < r0) * 0.12
        spots = np.maximum(spots, ring.astype(np.float32) * uni(rng, 0.5, 1.0))
    stain_col = np.array(tone, np.float32) * 0.55
    rgb = rgb * (1 - 0.55 * spots[..., None] * aging) + stain_col[None, None] * (0.55 * spots[..., None] * aging)
    return rgb


# ----------------------------------------------------------------------------
def make_paper(rng, W, H, mcfg):
    pal = pick(rng, mcfg["palettes"])
    light, dark = np.array(pal["light"], np.float32), np.array(pal["dark"], np.float32)
    aging = uni(rng, *mcfg.get("aging", [0.3, 1.0]))

    t = fbm(rng, H, W, max(W, H) / 2.5, octaves=4)
    t = np.clip(0.15 + 0.55 * t * aging + 0.12 * (fbm(rng, H, W, 40, octaves=3) - 0.5), 0, 1)
    rgb = lerp_color(light, dark, t)

    # fibres
    fib = np.full((H, W), 0.5, np.float32)
    for _ in range(int(W * H / 2200)):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a, l = rng.uniform(0, np.pi), rng.uniform(5, 28)
        p2 = (int(x + np.cos(a) * l), int(y + np.sin(a) * l))
        cv2.line(fib, (int(x), int(y)), p2, float(rng.choice([0.35, 0.65])), 1, cv2.LINE_AA)
    fib = blur(fib, 0.7)
    rgb *= (1 + (fib - 0.5) * 0.28)[..., None]

    # grain + laid lines
    grain = fbm(rng, H, W, 2.0, octaves=2) - 0.5
    rgb *= (1 + grain * 0.10)[..., None]
    if rng.random() < 0.55:
        period = uni(rng, 5.5, 9.0)
        xs = np.arange(W, dtype=np.float32)
        laid = np.sin(2 * np.pi * xs / period + (fbm(rng, 1, W, 60, 1, 2)[0] * 3)) * 0.5 + 0.5
        rgb *= (1 - 0.035 * laid)[None, :, None]

    rgb = add_stains(rng, rgb, W, H, aging, dark)

    alpha = folio_mask(rng, W, H, radius=uni(rng, 4, 18), rough=uni(rng, 0.5, 2.0) * (0.5 + aging), wave=uni(rng, 0.5, 2.5))

    # edge wear: darker, browner border
    d = inside_distance(alpha)
    edge = np.exp(-d / (0.018 * min(W, H) + 1)) * (0.25 + 0.5 * aging)
    edge *= 0.6 + 0.8 * fbm(rng, H, W, 30)
    rgb = rgb * (1 - edge[..., None] * 0.55) + dark[None, None] * 0.45 * edge[..., None]

    # worm holes (tiny holes with dark rim)
    holes = []
    for _ in range(int(rng.integers(0, 4) * aging * 3)):
        cx, cy, r = int(rng.integers(0.1 * W, 0.9 * W)), int(rng.integers(0.1 * H, 0.9 * H)), int(rng.integers(2, 5))
        hole = np.zeros((H, W), np.float32)
        cv2.circle(hole, (cx, cy), r, 1.0, -1)
        rim = blur(hole, 2.0)
        alpha = alpha * (1 - (hole > 0.5)) 
        rgb *= (1 - 0.4 * rim[..., None])
        holes.append((cx, cy, r + 6))
    return {"rgb": rgb, "alpha": alpha, "holes": holes, "kind": "paper", "palette": pal["name"], "aging": aging,
            "tone": dark}


# ----------------------------------------------------------------------------
def make_palm_leaf(rng, W, H, mcfg):
    pal = pick(rng, mcfg["palettes"])
    light, dark = np.array(pal["light"], np.float32), np.array(pal["dark"], np.float32)
    aging = uni(rng, *mcfg.get("aging", [0.3, 1.0]))

    streak = 0.6 * fbm(rng, H, W, 220, 3.0, octaves=3) + 0.4 * fbm(rng, H, W, 90, 1.6, octaves=3)
    blotch = fbm(rng, H, W, max(W, H) / 5, H / 1.5, octaves=4)
    mottle = fbm(rng, H, W, 60, 14, octaves=3)
    t = np.clip(0.08 + 0.42 * streak + 0.42 * blotch * (0.5 + aging) + 0.25 * (mottle - 0.5), 0, 1)
    rgb = lerp_color(light, dark, t)

    # fine longitudinal veins
    veins = np.full((H, W), 0.5, np.float32)
    for _ in range(int(rng.integers(40, 110))):
        y0 = rng.uniform(0, H)
        xs = np.linspace(0, W, 24)
        ys = y0 + np.cumsum(rng.normal(0, 0.35, xs.size))
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(veins, [pts], False, float(rng.uniform(0.2, 0.42)), 1, cv2.LINE_AA)
    veins = blur(veins, 0.6)
    rgb *= (1 + (veins - 0.5) * 0.55)[..., None]
    rgb *= (1 + (fbm(rng, H, W, 1.8, 1.8, octaves=2) - 0.5) * 0.12)[..., None]
    # scratches and soot-darkened patches
    sc = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(6, 30) * (0.5 + aging))):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        L = rng.uniform(30, 260)
        cv2.line(sc, (int(x), int(y)), (int(x + L), int(y + rng.normal(0, 4))), float(rng.uniform(0.3, 1)), 1, cv2.LINE_AA)
    rgb *= (1 - 0.22 * blur(sc, 0.7))[..., None]
    soot = smoothstep(0.62, 0.86, fbm(rng, H, W, W / 6, H / 1.2, octaves=3)) * aging
    rgb *= (1 - 0.14 * soot)[..., None]

    # cross splits near the edges
    cracks = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(3, 14))):
        x = int(rng.integers(0, W))
        top = rng.random() < 0.5
        y0, y1 = (0, int(rng.uniform(6, 0.12 * H))) if top else (H - 1, H - 1 - int(rng.uniform(6, 0.12 * H)))
        cv2.line(cracks, (x, y0), (x + int(rng.integers(-4, 5)), y1), 1.0, 1, cv2.LINE_AA)
    rgb *= (1 - 0.45 * blur(cracks, 0.8))[..., None]

    rgb = add_stains(rng, rgb, W, H, aging * 0.8, dark)

    alpha = folio_mask(rng, W, H, radius=uni(rng, 6, 22), rough=uni(rng, 1.0, 3.0), wave=uni(rng, 1.0, 5.0))
    d = inside_distance(alpha)
    edge = np.exp(-d / (0.03 * H + 1)) * (0.3 + 0.4 * aging)
    rgb = rgb * (1 - edge[..., None] * 0.5) + dark[None, None] * 0.35 * edge[..., None]

    # string holes
    holes = []
    n_holes = int(rng.choice([1, 2], p=[0.55, 0.45]))
    xs = [rng.uniform(0.32, 0.45)] if n_holes == 1 else [rng.uniform(0.14, 0.2), rng.uniform(0.78, 0.86)]
    if n_holes == 1 and rng.random() < 0.5:
        xs = [1 - xs[0]]
    for fx in xs:
        cx, cy, r = int(fx * W), int(H / 2 + rng.uniform(-0.04, 0.04) * H), int(rng.uniform(0.05, 0.07) * H)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist = np.hypot(xx - cx, yy - cy)
        wear = np.exp(-np.maximum(dist - r, 0) / (r * 0.8)) * (dist >= r)
        rgb *= (1 - 0.5 * wear[..., None] * (0.5 + aging * 0.5))
        alpha = alpha * np.clip((dist - r) / 1.5 + 0.5, 0, 1)
        holes.append((cx, cy, int(r * 1.5)))
    return {"rgb": rgb, "alpha": alpha, "holes": holes, "kind": "palm_leaf", "palette": pal["name"],
            "aging": aging, "tone": dark}


def make_background(rng, kind, W, H, mcfg):
    return (make_palm_leaf if kind == "palm_leaf" else make_paper)(rng, W, H, mcfg)


# ----------------------------------------------------------------------------
def make_backdrop(rng, W, H, kinds):
    """Scanner bed / cloth behind the folio."""
    kind = pick(rng, kinds)
    palette = {"black": (22, 22, 24), "charcoal": (52, 50, 48), "grey": (118, 118, 116),
               "brown_cloth": (66, 48, 34), "blue_grey": (96, 108, 120), "white": (236, 236, 232)}
    base = np.array(palette.get(kind, (40, 40, 40)), np.float32)
    n = fbm(rng, H, W, 3.0, octaves=2) - 0.5
    weave = (np.sin(np.arange(W) * 1.6)[None, :] * np.sin(np.arange(H) * 1.6)[:, None]) * 0.015 if kind.endswith("cloth") else 0
    shade = 1 + (fbm(rng, H, W, max(W, H) / 2) - 0.5) * 0.18
    img = base[None, None] * (shade + n * 0.08 + weave)[..., None]
    return img.astype(np.float32)
