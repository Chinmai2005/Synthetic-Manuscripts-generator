#!/usr/bin/env python3
"""Sanity-check a generated dataset: counts, image/.md pairs, QA flags, image readability.

    python validate.py output
"""
import json
import sys
from pathlib import Path

import cv2

EXPECTED = {"train": 85, "validation": 10, "test": 5}


def main(root="output"):
    root, errors = Path(root), []
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        total = 0
        for split, want in EXPECTED.items():
            d = sdir / split
            imgs = sorted(p for p in d.glob("Image_*") if p.suffix in (".png", ".jpg", ".webp"))
            mds = sorted(d.glob("Image_*.md"))
            total += len(imgs)
            if len(imgs) != want:
                errors.append(f"{sdir.name}/{split}: {len(imgs)} images, expected {want}")
            if {p.stem for p in imgs} != {p.stem for p in mds}:
                errors.append(f"{sdir.name}/{split}: image/.md mismatch")
            for p in imgs:
                if cv2.imread(str(p)) is None:
                    errors.append(f"unreadable image {p}")
            for p in mds:
                if not p.read_text(encoding="utf-8").strip():
                    errors.append(f"empty annotation {p}")
            meta = d / "metadata.jsonl"
            if meta.exists():
                for line in meta.read_text(encoding="utf-8").splitlines():
                    r = json.loads(line)
                    if not r["qa_passed"]:
                        errors.append(f"QA flag failed: {sdir.name}/{split}/{r['file_name']}")
        print(f"{sdir.name}: {total} images")
    print("OK - dataset is consistent" if not errors else "\n".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
