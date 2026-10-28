#!/usr/bin/env python3
"""Synthetic Indic manuscript dataset generator.

    python generate.py                       # full dataset from config.yaml
    python generate.py --scripts modi --num 6 --out preview --set image.scale=0.6
"""
import argparse
import sys

from manuscriptgen.config import load_config
from manuscriptgen.dataset import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", help="output directory (overrides output.dir)")
    ap.add_argument("--scripts", nargs="+", help="subset of scripts to generate")
    ap.add_argument("--num", type=int, help="images per script (splits are scaled proportionally)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config value")
    args = ap.parse_args()

    ov = list(args.set)
    if args.out:
        ov.append(f"output.dir={args.out}")
    if args.scripts:
        ov.append("dataset.scripts=[" + ",".join(args.scripts) + "]")
    if args.seed is not None:
        ov.append(f"seed={args.seed}")
    cfg = load_config(args.config, ov)
    build_dataset(cfg, num=args.num, workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
