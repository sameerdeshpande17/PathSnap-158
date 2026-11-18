"""python -m pathsnap.train --model MODEL_NAME [--cv] [--skip-existing] [--force] [--resume RUN_ID]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pathsnap.training import (ALL_MODELS, DEFAULT_STRATEGY, train_model,
                               load_run_config)

STRATEGIES = ["consensus", "consensus_plus_B"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.train",
        description="Train a PathSnap candidate-ranking model.")
    p.add_argument("--model", choices=ALL_MODELS + ["all"],
                   required=True, help="Model to train, or 'all'")
    p.add_argument("--strategy", choices=STRATEGIES, default=DEFAULT_STRATEGY,
                   help="Weak-label supervision strategy")
    p.add_argument("--cv", action="store_true",
                   help="Run trip-grouped cross-validation instead of final training")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip if a compatible checkpoint already exists")
    p.add_argument("--force", action="store_true",
                   help="Create a new run even if a compatible checkpoint exists")
    p.add_argument("--resume", metavar="RUN_ID", default=None,
                   help="Resume an interrupted run (PyTorch MLP only)")
    p.add_argument("--config", metavar="PATH", default=None,
                   help="Path to experiment config JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = None
    if args.config:
        config = json.loads(Path(args.config).read_text())
    else:
        config = load_run_config()
    if args.skip_existing and args.force:
        print("[pathsnap.train] error: use either --skip-existing or --force",
              file=sys.stderr)
        return 2
    models = ALL_MODELS if args.model == "all" else [args.model]
    started = time.time()
    results = []
    for model_name in models:
        print(f"\n=== pathsnap.train: {model_name} | strategy={args.strategy} "
              f"cv={args.cv} ===", flush=True)
        try:
            result = train_model(
                model_name, strategy=args.strategy, cv=args.cv,
                skip_existing=args.skip_existing, force=args.force,
                resume=args.resume, config=config)
            results.append(result)
        except Exception as exc:  # pragma: no cover - report and continue
            print(f"[pathsnap.train] {model_name} FAILED: {exc}", file=sys.stderr,
                  flush=True)
            return 1
    elapsed = time.time() - started
    print(f"\n[pathsnap.train] completed {len(results)} model(s) in "
          f"{elapsed:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
