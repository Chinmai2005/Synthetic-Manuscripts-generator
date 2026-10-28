"""Dataset assembly: reproducible seeds, train/validation/test split, images + .md + metadata."""
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np

from .compose import build_markdown, build_page
from .config import resolve
from .scripts import ScriptSpec

_CFG = None
_SPECS = {}


def split_counts(total, splits):
    vals = list(splits.values())
    if all(isinstance(v, float) and v <= 1 for v in vals) and abs(sum(vals) - 1) < 1e-6:
        counts = {k: int(round(v * total)) for k, v in splits.items()}
    else:  # absolute counts -> scale to `total`
        s = float(sum(vals))
        counts = {k: int(round(v / s * total)) for k, v in splits.items()}
    first = next(iter(counts))
    counts[first] += total - sum(counts.values())      # remainder goes to train
    return counts


def _init(cfg):
    global _CFG
    _CFG = cfg
    for name in cfg["dataset"]["scripts"]:
        _SPECS[name] = ScriptSpec(name, cfg["scripts"][name], cfg)


def _render(task):
    script, script_idx, n, split, out_dir = task
    cfg, spec = _CFG, _SPECS[script]
    seed = int(cfg["seed"])
    res, attempt = None, 0
    for attempt in range(int(cfg["dataset"]["max_attempts"])):
        rng = np.random.default_rng(np.random.SeedSequence([seed, script_idx, n, attempt]))
        res = build_page(cfg, spec, rng)
        if res["ok"]:
            break
    ocfg = cfg["output"]
    ext = {"png": "png", "jpg": "jpg", "jpeg": "jpg", "webp": "webp"}[ocfg["image_format"].lower()]
    stem = f"{ocfg['image_prefix']}{n}"
    d = Path(out_dir) / script / split
    d.mkdir(parents=True, exist_ok=True)
    bgr = res["image"][..., ::-1]
    if ext == "jpg":
        cv2.imwrite(str(d / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(ocfg["jpeg_quality"])])
    elif ext == "webp":
        cv2.imwrite(str(d / f"{stem}.webp"), bgr, [cv2.IMWRITE_WEBP_QUALITY, int(ocfg["jpeg_quality"])])
    else:
        cv2.imwrite(str(d / f"{stem}.png"), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    md = build_markdown(res["groups"], ocfg["markdown_style"])
    (d / f"{stem}.md").write_text(md, encoding="utf-8")
    m = res["meta"]
    row = {
        "file_name": f"{stem}.{ext}", "text": build_markdown(res["groups"], "plain").strip(),
        "script": script, "material": m["material"], "paper_palette": m["palette"], "layout": m["layout"],
        "font": m["font"], "ink": m["ink"], "width": m["image_size"][0], "height": m["image_size"][1],
        "n_lines": m["n_lines"], "seed": seed, "attempts": attempt + 1, "qa_passed": bool(res["ok"]),
        "lines_json": json.dumps(m["lines"], ensure_ascii=False),
        "highlights_json": json.dumps(m["highlights"], ensure_ascii=False),
    }
    return script, split, n, row


def build_dataset(cfg, num=None, workers=1):
    out_dir = resolve(cfg, cfg["output"]["dir"]) if not Path(cfg["output"]["dir"]).is_absolute() else Path(cfg["output"]["dir"])
    scripts = cfg["dataset"]["scripts"]
    total = int(num or cfg["dataset"]["images_per_script"])
    counts = split_counts(total, cfg["dataset"]["splits"])
    print(f"Output: {out_dir}\nSplits per script: {counts}")

    tasks = []
    for si, script in enumerate(scripts):
        n = 0
        for split, c in counts.items():
            for _ in range(c):
                n += 1
                tasks.append((script, si, n, split, str(out_dir)))
    rows = {}
    t0 = time.time()
    if workers > 1:
        with mp.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
            it = pool.imap_unordered(_render, tasks)
            for k, r in enumerate(it, 1):
                rows.setdefault((r[0], r[1]), []).append((r[2], r[3]))
                _progress(k, len(tasks), t0)
    else:
        _init(cfg)
        for k, t in enumerate(tasks, 1):
            r = _render(t)
            rows.setdefault((r[0], r[1]), []).append((r[2], r[3]))
            _progress(k, len(tasks), t0)
    print()
    if cfg["output"]["write_metadata"]:
        for (script, split), items in rows.items():
            p = Path(out_dir) / script / split / "metadata.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for _, row in sorted(items, key=lambda x: x[0]):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    bad = sum(1 for items in rows.values() for _, r in items if not r["qa_passed"])
    print(f"Done: {len(tasks)} images in {time.time() - t0:.0f}s, {bad} failed QA (should be 0).")


def _progress(k, n, t0):
    el = time.time() - t0
    print(f"\r[{k}/{n}] {el:5.0f}s elapsed, ~{el / k * (n - k):5.0f}s left", end="", flush=True)
