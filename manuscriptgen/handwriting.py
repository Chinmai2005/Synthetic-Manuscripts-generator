"""Hand-written look for shaped Indic text.

Text is shaped with HarfBuzz/Raqm *per word* (so conjuncts and matras are always
correct), then every word is individually distorted: pen-pressure thickness
variation, rough ink edges, slant, rotation, baseline waviness and elastic
jitter. Words are then placed on wavy, slightly sloping baselines.
"""
import math
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .noise import blur, fbm, smoothstep

_FONT_CACHE = {}


def get_font(path, size):
    key = (path, int(size))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, int(size), layout_engine=ImageFont.Layout.RAQM)
    return _FONT_CACHE[key]


class InkLayers:
    """Accumulates ink: colour, opacity and a line-id label map."""

    def __init__(self, h, w, base_color):
        self.h, self.w = h, w
        self.rgb = np.empty((h, w, 3), np.float32)
        self.rgb[:] = np.asarray(base_color, np.float32)
        self.alpha = np.zeros((h, w), np.float32)
        self.label = np.zeros((h, w), np.uint8)

    def paste(self, a, x, y, color, label_id):
        ah, aw = a.shape
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + aw, self.w), min(y + ah, self.h)
        if x1 <= x0 or y1 <= y0:
            return None
        sub = a[y0 - y:y1 - y, x0 - x:x1 - x]
        cur = self.alpha[y0:y1, x0:x1]
        upd = sub > cur
        self.rgb[y0:y1, x0:x1][upd] = color
        self.alpha[y0:y1, x0:x1] = np.maximum(cur, sub)
        lab = self.label[y0:y1, x0:x1]
        lab[sub > 0.12] = label_id
        ys, xs = np.nonzero(sub > 0.12)
        if xs.size == 0:
            return None
        return (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)


# ----------------------------------------------------------------------------
def render_word(rng, font_path, size, text, style, angle=0.0, slant=0.0, ss=2):
    """Return dict(alpha, ox, oy, adv) – alpha at 1x, (ox, oy) = baseline origin."""
    f = get_font(font_path, size * ss)
    asc, desc = f.getmetrics()
    adv = f.getlength(text)
    pad = int(0.7 * size * ss) // ss * ss
    W = int(adv + 2 * pad) // ss * ss + ss
    H = int(asc + desc + 2 * pad) // ss * ss + ss
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).text((pad, pad + asc), text, font=f, fill=255, anchor="ls")
    a = np.asarray(im, np.float32) / 255.0

    # pen pressure / rough edges: threshold a blurred glyph with a noisy threshold
    b = blur(a, 0.75 * ss * style["soften"])
    press = fbm(rng, H, W, 9 * ss, octaves=2) - 0.5
    rough = fbm(rng, H, W, 2.4 * ss, octaves=2) - 0.5
    thr = 0.5 - style["weight"] + press * style["pen_var"] * 2 + rough * 0.12
    thr = np.clip(thr, 0.24, 0.8)
    a = smoothstep(thr - 0.2, thr + 0.16, b)
    a[b < 0.02] = 0

    # affine: scale + slant + rotation about the baseline origin
    ox, oy = float(pad), float(pad + asc)
    sx = 1 + rng.normal(0, style["scale_jitter"])
    sy = 1 + rng.normal(0, style["scale_jitter"] * 0.7)
    sh = math.tan(math.radians(slant))
    A = np.array([[sx, -sh * sy, 0], [0, sy, 0], [0, 0, 1]], np.float64)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], np.float64)
    T1 = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], np.float64)
    T2 = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    M = T2 @ R @ A @ T1
    a = cv2.warpAffine(a, M[:2], (W, H), flags=cv2.INTER_LINEAR)

    # elastic wobble + tiny wave inside the word
    amp = style["jitter"] * ss * 0.9
    dx = (fbm(rng, H, W, 16 * ss, octaves=2) - 0.5) * 2 * amp
    dy = (fbm(rng, H, W, 16 * ss, octaves=2) - 0.5) * 2 * amp
    xs = np.arange(W, dtype=np.float32)[None, :]
    dy = dy + style["wave_amp"] * 0.35 * ss * np.sin(xs / (22.0 * ss) + rng.uniform(0, 6.28))
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    a = cv2.remap(a, gx + dx, gy + dy, cv2.INTER_LINEAR)

    a = cv2.resize(a, (W // ss, H // ss), interpolation=cv2.INTER_AREA)
    return {"alpha": a, "ox": ox / ss, "oy": oy / ss, "adv": adv / ss * sx}


# ----------------------------------------------------------------------------
def free_segments(x0, x1, y_top, y_bot, reserved, min_w):
    segs = [(x0, x1)]
    for rx0, ry0, rx1, ry1 in reserved:
        if ry1 < y_top or ry0 > y_bot:
            continue
        nxt = []
        for a, b in segs:
            if rx1 <= a or rx0 >= b:
                nxt.append((a, b))
                continue
            if rx0 - a >= min_w:
                nxt.append((a, rx0))
            if b - rx1 >= min_w:
                nxt.append((rx1, b))
        segs = nxt
    return [s for s in segs if s[1] - s[0] >= min_w]


def paint_block(rng, layers, rect, n_lines, size, font, tokens, style, palette, reserved=(),
                line_id0=1, gap_range=(0.14, 0.34), justify=0.6, rubric_prob=0.0, line_spacing=1.5,
                center=False, role="main"):
    """Fill `rect` with `n_lines` of handwritten text. Returns (line_records, next_line_id)."""
    x0, y0, x1, y1 = rect
    first_b = y0 + 0.95 * size
    last_b = y1 - 0.5 * size
    pitch = (last_b - first_b) / max(n_lines - 1, 1) if n_lines > 1 else 0
    fpath = font["path"]
    fsize = int(size * font["size_scale"])
    pend = deque()
    records, lid = [], line_id0

    def nxt():
        return pend.popleft() if pend else next(tokens)

    base_slant = style["slant"]
    for li in range(n_lines):
        b0 = first_b + pitch * li + rng.normal(0, 0.05 * pitch)
        slope = rng.normal(0, 0.0035)
        wave_a = style["wave_amp"] * rng.uniform(0.6, 1.4)
        wave_l = rng.uniform(160, 420)
        wave_p = rng.uniform(0, 6.28)
        drift = rng.normal(0, 0.0004)
        segs = free_segments(x0, x1, b0 - size, b0 + 0.45 * size, reserved, 3 * size)
        if not segs:
            continue
        # global right-edge irregularity
        shrink = rng.uniform(0, 0.035) * (x1 - x0)
        line_words, last_seg = [], len(segs) - 1
        for si, (sx0, sx1) in enumerate(segs):
            sx1 = sx1 - (shrink if si == last_seg else 0)
            width = sx1 - sx0
            items, cur = [], 0.0
            while True:
                try:
                    tok = nxt()
                except StopIteration:
                    break
                wlen = get_font(fpath, fsize).getlength(tok["text"]) * 1.07
                gap = size * rng.uniform(*gap_range)
                if wlen > width * 0.97:
                    continue  # un-renderable in this block
                if items and cur + gap + wlen > width:
                    pend.appendleft(tok)
                    break
                cur += (gap if items else 0) + wlen
                items.append(tok)
            line_words.append((sx0, sx1, items))
        # render words, position, paste
        rec_words, text_parts = [], []
        is_last = li == n_lines - 1
        for sx0, sx1, items in line_words:
            width = sx1 - sx0
            rendered = []
            for tok in items:
                x_mid = 0.0
                ang = math.degrees(math.atan(slope)) + rng.normal(0, 0.7)
                sl = base_slant + rng.normal(0, style["slant_sd"])
                wsize = int(fsize * rng.uniform(0.96, 1.04))
                rendered.append((tok, render_word(rng, fpath, wsize, tok["text"], style, ang, sl)))
            total = sum(r["adv"] for _, r in rendered)
            n = len(rendered)
            gaps = n - 1
            nat = size * float(np.mean(gap_range))
            if gaps > 0 and rng.random() < justify and not (is_last and rng.random() < 0.7):
                g = np.clip((width - total) / gaps, 0.06 * size, 0.9 * size)
            else:
                g = nat * rng.uniform(0.85, 1.2)
            run = total + g * max(gaps, 0)
            if run > width:  # compress, never overflow
                g = max((width - total) / max(gaps, 1), 0.04 * size)
                run = total + g * max(gaps, 0)
            start = sx0 + rng.normal(0, 0.006 * (x1 - x0)) if not center else sx0 + (width - run) / 2
            start = min(max(start, sx0), max(sx0, sx1 - run))
            cx = start
            for tok, r in rendered:
                bx = cx
                by = (b0 + slope * (bx - x0) + drift * (bx - x0) ** 2 / 100.0
                      + wave_a * math.sin(2 * math.pi * (bx - x0) / wave_l + wave_p) + rng.normal(0, 0.02 * size))
                col = palette["red"] if (tok["role"] == "marker" and rng.random() < palette["marker_red_prob"]) \
                    or (tok["role"] == "word" and rng.random() < rubric_prob) else palette["ink"]
                col = np.clip(np.asarray(col, np.float32) * rng.uniform(0.88, 1.12) + rng.normal(0, 3, 3), 0, 255)
                box = layers.paste(r["alpha"], int(round(bx - r["ox"])), int(round(by - r["oy"])), col, lid)
                if box:
                    rec_words.append({"text": tok["text"], "role": tok["role"], "box": box, "baseline": by,
                                      "red": bool(col[0] > col[2] + 60)})
                    text_parts.append(tok["text"])
                cx += r["adv"] + g
        if rec_words:
            records.append({"id": lid, "text": " ".join(text_parts), "words": rec_words, "role": role,
                            "baseline": b0})
            lid += 1
    return records, lid
