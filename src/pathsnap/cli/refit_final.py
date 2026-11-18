"""python -m pathsnap.refit_final --model MODEL_NAME"""
from __future__ import annotations

import argparse
import sys

from pathsnap.model_selection import refit_final


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.refit_final",
        description="Refit the selected model on the full training partition "
                    "and save the final checkpoint.")
    p.add_argument("--model", required=True,
                   help="Model to refit (must have completed optimization).")
    p.add_argument("--strategy", default="consensus",
                   help="Weak-label supervision strategy (default: consensus).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        refit_final(args.model, strategy=args.strategy)
    except Exception as exc:
        print(f"[pathsnap.refit_final] FAILED: {exc}", file=sys.stderr,
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
