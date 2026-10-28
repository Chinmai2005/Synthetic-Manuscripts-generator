"""Builds one synthetic folio: background -> layout -> handwriting -> artefacts -> deformation."""
import math

import cv2
import numpy as np

from .artifacts import apply_ink_artifacts, show_through
from .backgrounds import make_background
from .deform import deform_scene, scan_effects
from .handwriting import InkLayers, paint_block
from .noise import blur, fbm


def U(rng, rg):
    return float(rng.uniform(rg[0], rg[1]))


def weighted(rng, d):
    keys = list(d.keys())
    p = np.array([d[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def block_size(h, n, spacing):
    return h / ((n - 1) * spacing + 1.45)


class Page:
    """Mutable state shared by the layout helpers."""

    def __init__(self, rng, cfg, spec, kind, W, H, bg):
        self.rng, self.cfg, self.spec, self.kind, self.W, self.H, self.bg = rng, cfg, spec, kind, W, H, bg
        self.k = W / 1400.0
        t = cfg["text"]
        self.style = {
            "slant": U(rng, t["slant_deg"]), "slant_sd": t["slant_sd"],
            "wave_amp": U(rng, t["wave_amp"]) * self.k, "jitter": U(rng, t["jitter"]),
            "weight": U(rng, t["weight"]), "pen_var": U(rng, t["pen_var"]),
            "scale_jitter": t["scale_jitter"], "soften": U(rng, [0.8, 1.3]),
        }
        if kind == "palm_leaf":
            self.style["weight"] -= 0.05        # incised strokes are thinner
        self.font = dict(spec.fonts[int(rng.integers(len(spec.fonts)))])
        self.style["weight"] += self.font["weight"]
        ink = cfg["ink"]["colors"][int(rng.integers(len(cfg["ink"]["colors"])))]
        red = ink_red = cfg["ink"]["red"][int(rng.integers(len(cfg["ink"]["red"])))]
        self.ink_name = ink["name"]
        self.palette = {"ink": np.array(ink["rgb"], np.float32), "red": np.array(red, np.float32),
                        "marker_red_prob": float(rng.choice([0.0, 0.0, 0.6, 1.0]))}
        self.layers = InkLayers(H, W, self.palette["ink"])
        self.art_rgb = np.zeros((H, W, 3), np.float32)
        self.art_a = np.zeros((H, W), np.float32)
        self.records, self.groups, self.next_id = [], [], 1
        self.spacing = U(rng, t["line_spacing"])
        self.verse_numbers = rng.random() < cfg["layout"]["verse_number_prob"]

    # -- text ---------------------------------------------------------------
    def paint(self, rect, n, size, role, tokens=None, palette=None, **kw):
        tokens = tokens or self.spec.token_stream(self.rng, self.verse_numbers)
        recs, self.next_id = paint_block(
            self.rng, self.layers, rect, n, size, self.font, tokens, self.style, palette or self.palette,
            reserved=self.reserved, line_id0=self.next_id, rubric_prob=self.cfg["text"]["rubric_prob"],
            gap_range=self.cfg["text"]["gap_ratio"], justify=U(self.rng, self.cfg["text"]["justify_prob"]),
            line_spacing=self.spacing, role=role, **kw)
        if recs:
            self.records += recs
            self.groups.append((role, recs))
        return recs

    # -- hand-drawn line art ---------------------------------------------------
    def draw_poly(self, pts, color, alpha, thick):
        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        a = np.zeros((self.H, self.W), np.float32)
        cv2.polylines(a, [pts], False, 1.0, max(1, int(round(thick))), cv2.LINE_AA)
        a = blur(a, 0.7) * alpha * (0.7 + 0.5 * fbm(self.rng, self.H, self.W, 40))
        upd = a > self.art_a
        self.art_rgb[upd] = color
        self.art_a = np.maximum(self.art_a, a)

    def wobbly_line(self, p0, p1, color, alpha, thick, amp=1.6):
        n = max(int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 40), 2)
        t = np.linspace(0, 1, n)
        xs = p0[0] + (p1[0] - p0[0]) * t + self.rng.normal(0, amp * 0.5, n)
        ys = p0[1] + (p1[1] - p0[1]) * t + self.rng.normal(0, amp, n)
        self.draw_poly(np.stack([xs, ys], 1), color, alpha, thick)


def build_page(cfg, spec, rng):
    """Returns dict(image, markdown_groups, meta)."""
    kind = weighted(rng, spec.cfg["materials"])
    mcfg = cfg["materials"][kind]
    sw, sh = mcfg["sizes"][int(rng.integers(len(mcfg["sizes"])))]
    sc = cfg["image"]["scale"]
    W, H = int(sw * sc), int(sh * sc)
    bg = make_background(rng, kind, W, H, mcfg)
    P = Page(rng, cfg, spec, kind, W, H, bg)
    lay = weighted(rng, cfg["layout"][kind])
    portrait = H > W * 1.1

    # --- margins & content box -----------------------------------------------------
    if kind == "paper":
        ml, mr = U(rng, [.08, .13]) * W, U(rng, [.07, .12]) * W
        mt, mb = U(rng, [.08, .13]) * H, U(rng, [.08, .14]) * H
        if lay == "margin_notes":
            ml, mr = U(rng, [.15, .20]) * W, U(rng, [.13, .18]) * W
            if rng.random() < 0.5:
                ml, mr = mr, ml
    else:
        ml, mr = U(rng, [.05, .08]) * W, U(rng, [.05, .08]) * W
        mt, mb = U(rng, [.15, .21]) * H, U(rng, [.15, .21]) * H
    cx0, cy0, cx1, cy1 = ml, mt, W - mr, H - mb
    P.reserved = [(hx - hr * 1.15, hy - hr * 1.15, hx + hr * 1.15, hy + hr * 1.15) for hx, hy, hr in bg["holes"]]

    sp = P.spacing
    meta_layout = lay
    blocks = []     # (rect, n, size, role)
    nr = (lambda lo, hi: int(rng.integers(lo, hi + 1)))
    if lay == "palm":
        n = nr(3, 5)
        blocks.append(((cx0, cy0, cx1, cy1), n, block_size(cy1 - cy0, n, sp), "main"))
    elif lay == "commentary":
        n1 = nr(4, 6) if not portrait else nr(5, 8)
        h = cy1 - cy0
        r1 = (cx0, cy0, cx1, cy0 + 0.60 * h)
        r2 = (cx0, cy0 + 0.68 * h, cx1, cy1)
        n2 = max(3, n1 - int(rng.integers(0, 2)))
        s1 = block_size(r1[3] - r1[1], n1, sp)
        blocks += [(r1, n1, s1, "main"), (r2, n2, min(block_size(r2[3] - r2[1], n2, sp), 0.8 * s1), "commentary")]
    elif lay == "two_column":
        n = nr(8, 11) if not portrait else nr(11, 15)
        g = U(rng, [.04, .07]) * (cx1 - cx0)
        cw = (cx1 - cx0 - g) / 2
        s = block_size(cy1 - cy0, n, sp)
        blocks += [((cx0, cy0, cx0 + cw, cy1), n, s, "main"), ((cx0 + cw + g, cy0, cx1, cy1), n, s, "main")]
    else:  # single / margin_notes
        n = nr(7, 11) if not portrait else nr(9, 14)
        blocks.append(((cx0, cy0, cx1, cy1), n, block_size(cy1 - cy0, n, sp), "main"))

    # --- optional invocation heading --------------------------------------------------
    first_rect, n0, s0, _ = blocks[0]
    if spec.headings and rng.random() < cfg["layout"]["heading_prob"] and n0 > 3 and lay != "two_column":
        head_h = 1.7 * s0
        hw = (first_rect[2] - first_rect[0])
        hr = (first_rect[0] + 0.2 * hw, first_rect[1], first_rect[2] - 0.2 * hw, first_rect[1] + head_h)
        hp = dict(P.palette)
        if rng.random() < 0.6:
            hp["ink"] = P.palette["red"]
        htok = iter([{"text": w, "role": "word"} for w in spec.headings[int(rng.integers(len(spec.headings)))].split()]
                    + [{"text": spec.danda, "role": "marker"}] * 0)
        P.paint(hr, 1, s0 * 1.05, "heading", tokens=htok, palette=hp, center=True)
        nrect = (first_rect[0], first_rect[1] + head_h + 0.15 * s0, first_rect[2], first_rect[3])
        n0 = max(n0 - 1, 3)
        blocks[0] = (nrect, n0, block_size(nrect[3] - nrect[1], n0, sp), "main")

    for rect, n, size, role in blocks:
        P.paint(rect, n, size, role)
    main_size = blocks[0][2]

    # --- ruled frame, guide lines, ornaments -----------------------------------------------
    red = P.palette["red"]
    line_col = red if rng.random() < 0.65 else P.palette["ink"]
    if kind == "paper" and rng.random() < cfg["layout"]["frame_prob"]:
        e = min(ml, mr, mt, mb) * U(rng, [.3, .5])
        fx0, fy0, fx1, fy1 = cx0 - e, cy0 - e, cx1 + e, cy1 + e
        for off in ([0] if rng.random() < 0.5 else [0, U(rng, [.008, .014]) * min(W, H) + 3]):
            o = off
            c = [(fx0 + o, fy0 + o), (fx1 - o, fy0 + o), (fx1 - o, fy1 - o), (fx0 + o, fy1 - o)]
            for i in range(4):
                P.wobbly_line(c[i], c[(i + 1) % 4], line_col, U(rng, [.7, .95]), 2.2 * P.k + 0.6)
    if rng.random() < cfg["layout"]["guide_line_prob"] and kind == "paper":
        for rect, n, size, role in blocks:
            for r in P.records:
                if r["role"] == role and rect[0] <= r["words"][0]["box"][0] <= rect[2]:
                    y = r["baseline"] + 0.22 * size
                    P.wobbly_line((rect[0] - 6, y), (rect[2] + 6, y), (120, 100, 80), 0.28, 1.2 * P.k + 0.4, amp=0.7)
    if rng.random() < cfg["layout"]["ornament_prob"]:
        for rect, n, size, role in blocks[:1]:
            for r in P.records[:1]:
                if r["role"] == "main" and rect[0] - 1.1 * size > 0.03 * W:
                    ox, oy = rect[0] - 0.7 * size, r["baseline"] - 0.35 * size
                    rr = 0.28 * size
                    cv2.circle(P.art_a, (int(ox), int(oy)), int(rr), 0.9, max(1, int(2 * P.k)), cv2.LINE_AA)
                    cv2.circle(P.art_a, (int(ox), int(oy)), max(1, int(rr * 0.3)), 0.9, -1, cv2.LINE_AA)
                    P.art_rgb[P.art_a > 0.05] = np.where(P.art_a[P.art_a > 0.05][:, None] > 0, red, 0)
    if kind == "palm_leaf":
        for hx, hy, hr in bg["holes"]:        # red ring around the string hole
            if rng.random() < 0.6:
                a = np.zeros((H, W), np.float32)
                cv2.circle(a, (int(hx), int(hy)), int(hr * 0.78), 0.85, max(1, int(3 * P.k)), cv2.LINE_AA)
                a = blur(a, 0.8)
                upd = a > P.art_a
                P.art_rgb[upd] = red
                P.art_a = np.maximum(P.art_a, a)

    # --- marginal annotations / side text ---------------------------------------------------
    notes = []
    want_notes = (lay == "margin_notes" and rng.random() < cfg["text"]["notes_prob"]) or \
                 (kind == "paper" and rng.random() < 0.2)
    if want_notes and kind == "paper":
        left_w, right_w = cx0 - 0.05 * W, W - cx1 - 0.05 * W
        for side in ("L", "R"):
            avail = left_w if side == "L" else right_w
            if avail < 0.09 * W or (notes and rng.random() < 0.35):
                continue
            nsize = main_size * U(rng, [0.62, 0.8])
            words = spec.sample_words(rng, int(rng.integers(2, 5)))
            vertical = rng.random() < 0.7 and avail < 0.22 * W or rng.random() < 0.5
            ncol = P.palette["red"] if rng.random() < 0.4 else P.palette["ink"]
            npal = dict(P.palette, ink=ncol, marker_red_prob=0)
            tok = iter([{"text": w, "role": "word"} for w in words])
            if vertical:
                pw, ph = int(0.62 * H), int(nsize * 1.9)
                patch = InkLayers(ph, pw, ncol)
                save = (P.layers, P.reserved)
                P.layers, P.reserved = patch, []
                recs = P.paint((0, 0, pw, ph), 1, nsize, "note", tokens=tok, palette=npal, center=True)
                P.layers, P.reserved = save
                if not recs:
                    continue
                ys, xs = np.nonzero(patch.alpha > 0.05)
                if xs.size == 0:
                    continue
                crop = lambda m: m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                ra, rr_, rl = crop(patch.alpha), crop(patch.rgb), crop(patch.label)
                k = 1 if side == "L" else 3          # rotate CCW on the left, CW on the right
                ra, rr_, rl = np.rot90(ra, k), np.rot90(rr_, k), np.rot90(rl, k)
                th, tw = ra.shape
                x = int(U(rng, [0.06 * W, max(0.06 * W + 1, cx0 - 0.06 * W - tw)])) if side == "L" \
                    else int(U(rng, [cx1 + 0.05 * W, max(cx1 + 0.05 * W + 1, W - 0.05 * W - tw)]))
                y = int(U(rng, [cy0, max(cy0 + 1, cy1 - th)]))
                if x < 0.03 * W or x + tw > W - 0.03 * W or y < 0.03 * H or y + th > H - 0.03 * H:
                    P.records = [r for r in P.records if r not in recs]
                    P.groups.pop()
                    continue
                lid = recs[0]["id"]
                sub = P.layers.alpha[y:y + th, x:x + tw]
                m = ra > sub
                P.layers.rgb[y:y + th, x:x + tw][m] = rr_[m]
                P.layers.alpha[y:y + th, x:x + tw] = np.maximum(sub, ra)
                P.layers.label[y:y + th, x:x + tw][rl > 0] = lid
                # horizontal -> vertical: update stored word boxes (approximate: whole note box)
                recs[0]["words"] = [{"text": recs[0]["text"], "role": "word", "box": (x, y, x + tw, y + th), "red": False}]
                notes.append(recs[0])
            else:
                rx0 = 0.04 * W if side == "L" else cx1 + 0.04 * W
                rx1 = cx0 - 0.04 * W if side == "L" else W - 0.04 * W
                if rx1 - rx0 < 4 * nsize:
                    continue
                ry0 = U(rng, [cy0, cy0 + 0.5 * (cy1 - cy0)])
                nl = int(rng.integers(2, 5))
                P.paint((rx0, ry0, rx1, ry0 + nl * nsize * 1.5), nl, nsize, "note",
                        palette=npal, tokens=iter([{"text": w, "role": "word"} for w in spec.sample_words(rng, 12) * 3]))
    if kind == "paper" and rng.random() < cfg["text"]["folio_number_prob"] and lay != "palm":
        num = __import__("manuscriptgen.scripts", fromlist=["convert_digits"]).convert_digits(str(int(rng.integers(1, 120))), spec.cfg)
        fs = main_size * 0.6
        fx = rng.choice([cx1 - 3 * fs, cx0]) 
        fy0 = cy1 + 0.25 * mb
        if fy0 + 1.8 * fs < H - 0.03 * H:
            P.paint((fx, fy0, fx + 3 * fs, fy0 + 1.8 * fs), 1, fs, "folio", tokens=iter([{"text": num, "role": "marker"}]),
                    palette=dict(P.palette, marker_red_prob=0), center=True)

    # --- highlighted spans (turmeric / haritala washes) --------------------------------------
    wash_a = np.zeros((H, W), np.float32)
    wash_rgb = np.zeros((H, W, 3), np.float32)
    highlights = []
    mains = [r for r in P.records if r["role"] in ("main", "commentary")]
    if mains and rng.random() < cfg["text"]["highlight_prob"]:
        wc = np.array(cfg["ink"]["highlight"][int(rng.integers(len(cfg["ink"]["highlight"])))], np.float32)
        for _ in range(int(rng.integers(cfg["text"]["highlight_spans"][0], cfg["text"]["highlight_spans"][1] + 1))):
            r = mains[int(rng.integers(len(mains)))]
            ws = r["words"]
            if len(ws) < 2:
                continue
            i = int(rng.integers(0, len(ws)))
            j = min(len(ws), i + int(rng.integers(1, 5)))
            span = ws[i:j]
            if any(w.get("hl") for w in span):
                continue
            x0 = min(w["box"][0] for w in span) - 0.18 * main_size
            x1 = max(w["box"][2] for w in span) + 0.18 * main_size
            y0 = min(w["box"][1] for w in span) - 0.12 * main_size
            y1 = max(w["box"][3] for w in span) + 0.12 * main_size
            pts = []
            for t in np.linspace(0, 1, 6):
                pts.append((x0 + (x1 - x0) * t, y0 + rng.normal(0, 1.5)))
            for t in np.linspace(1, 0, 6):
                pts.append((x0 + (x1 - x0) * t + rng.normal(0, 1.2), y1 + rng.normal(0, 1.5)))
            tmp = np.zeros((H, W), np.float32)
            cv2.fillPoly(tmp, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
            tmp = blur(tmp, 2.2) * rng.uniform(0.32, 0.55) * (0.7 + 0.6 * fbm(rng, H, W, 25))
            upd = tmp > wash_a
            wash_rgb[upd] = wc
            wash_a = np.maximum(wash_a, np.clip(tmp, 0, 1))
            for w in span:
                w["hl"] = True
            highlights.append({"text": " ".join(w["text"] for w in span), "box": [int(x0), int(y0), int(x1), int(y1)],
                               "line_id": r["id"]})

    # --- ink artefacts (bleeding, fading, smudges ...) ------------------------------------------
    apply_ink_artifacts(rng, P.layers, (cx0, cy0, cx1, cy1), main_size, cfg["artifacts"], kind, P.next_id)

    # --- composite on the flat folio ------------------------------------------------------------
    out = bg["rgb"].copy()
    out *= 1 - wash_a[..., None] * (1 - wash_rgb / 255.0)
    for col, a in ((P.art_rgb, P.art_a), (P.layers.rgb, P.layers.alpha)):
        out = out * (1 - a[..., None]) + (out * col / 255.0 * 1.05) * a[..., None]
    if kind == "paper":
        st = U(rng, cfg["artifacts"]["show_through"])
        if st > 0.005:
            out *= 1 - 0.8 * show_through(rng, P.layers.alpha, st)[..., None] * np.array([0.5, 0.7, 0.9])
    else:
        rel = cfg["artifacts"]["relief"]
        sh = np.roll(P.layers.alpha, (1, 1), (0, 1))
        out += 255 * rel * 0.35 * np.clip(sh - P.layers.alpha, 0, 1)[..., None]
    out = np.clip(out, 0, 255)

    # --- physical deformation + scan simulation -----------------------------------------------------
    scene, lab_w, a_w = deform_scene(rng, out, bg["alpha"], P.layers.label, kind, cfg["deform"], cfg["backdrops"])
    from .noise import blur as _b  # noqa
    img = scan_effects(rng, scene, cfg["scan"])

    # --- QA data: warped bounding boxes + boundary check ----------------------------------------------
    Hs, Ws = lab_w.shape
    inner = cv2.erode((a_w > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8))
    leaked = bool(np.any((lab_w > 0) & (inner == 0)))
    ys, xs = np.nonzero(lab_w > 0)
    near_border = bool(xs.size and (xs.min() < 6 or ys.min() < 6 or xs.max() > Ws - 7 or ys.max() > Hs - 7))
    bboxes = {}
    for r in P.records:
        yy, xx = np.nonzero(lab_w == r["id"])
        if xx.size:
            bboxes[r["id"]] = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    meta = {
        "material": kind, "palette": bg["palette"], "layout": meta_layout, "font": __import__("os").path.basename(P.font["path"]),
        "ink": P.ink_name, "image_size": [Ws, Hs], "n_lines": len(P.records),
        "highlights": highlights, "qa": {"text_inside_folio": not leaked, "clear_of_image_border": not near_border},
        "lines": [{"id": r["id"], "role": r["role"], "text": r["text"], "bbox": bboxes.get(r["id"])} for r in P.records],
    }
    return {"image": img, "groups": P.groups, "meta": meta, "ok": (not leaked) and (not near_border) and bool(P.records)}


def build_markdown(groups, style="plain"):
    """Reading order: heading, main block(s), commentary, marginal notes, folio number."""
    order = {"heading": 0, "main": 1, "commentary": 2, "note": 3, "folio": 4}
    titles = {"heading": "Heading", "main": "Main text", "commentary": "Commentary", "note": "Marginal note", "folio": "Folio number"}
    merged = {}
    for role, recs in groups:
        merged.setdefault(role, []).extend(recs)
    parts = []
    for role in sorted(merged, key=lambda r: order.get(r, 9)):
        lines = []
        for rec in merged[role]:
            if style == "structured":
                toks, i = [], 0
                for w in rec["words"]:
                    toks.append(f"=={w['text']}==" if w.get("hl") else w["text"])
                lines.append(" ".join(toks))
            else:
                lines.append(rec["text"])
        body = "\n".join(lines)
        parts.append(f"## {titles[role]}\n\n{body}" if style == "structured" else body)
    return ("\n\n".join(parts)).strip() + "\n"
