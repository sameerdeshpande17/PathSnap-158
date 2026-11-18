"""python -m pathsnap.select_final_model [--metric best_top1]"""
from __future__ import annotations

import argparse
import sys

from pathsnap.model_selection import select_final_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.select_final_model",
        description="Recommend the final model from training-only CV evidence "
                    "(no retraining).")
    p.add_argument("--metric", default="best_top1",
                   help="Primary selection metric (default: best_top1).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        select_final_model(metric=args.metric)
    except Exception as exc:
        print(f"[pathsnap.select_final_model] FAILED: {exc}",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
