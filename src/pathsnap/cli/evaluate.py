"""python -m pathsnap.evaluate --model MODEL_NAME [--split SPLIT] [--run-id RUN_ID]"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

from pathsnap.evaluation import evaluate_model, evaluate_all
from pathsnap.training import ALL_MODELS, load_best


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.evaluate",
        description="Load a trained model and evaluate without retraining.")
    p.add_argument("--model", choices=ALL_MODELS + ["all"],
                   required=True, help="Model to evaluate, or 'all'")
    p.add_argument("--split", choices=["train", "student_test"],
                   default="student_test",
                   help="Split to evaluate on (default: student_test)")
    p.add_argument("--run-id", metavar="RUN_ID", default=None,
                   help="Specific run ID (default: best checkpoint)")
    p.add_argument("--consensus-only", action="store_true", default=True,
                   help="Use B/C-consensus labeled population only "
                   "(Phase 2A recommended supervision; default)")
    p.add_argument("--union", action="store_true", default=False,
                   help="Use consensus + B-extra labeled population "
                   "(larger, less conservative)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    consensus_only = args.consensus_only and not args.union
    if args.model == "all":
        evaluate_all(split=args.split, consensus_only=consensus_only)
        return 0
    best = load_best(args.model)
    run_id = args.run_id or (best["best_run_id"] if best else None)
    try:
        evaluate_model(args.model, split=args.split, run_id=run_id,
                       consensus_only=consensus_only)
    except Exception as exc:
        print(f"[pathsnap.evaluate] {args.model} FAILED: {exc}", file=sys.stderr,
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
