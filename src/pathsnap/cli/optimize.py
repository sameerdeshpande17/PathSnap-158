"""python -m pathsnap.optimize --model MODEL_NAME --trials N

Optuna Bayesian hyperparameter optimization (TPE sampler).
Runs only when manually invoked. Does not retrain a final model.
"""
from __future__ import annotations

import argparse
import sys

from pathsnap.optimization import ALL_MODELS, DEFAULT_N_TRIALS, optimize
from pathsnap.training import STRATEGIES


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.optimize",
        description="Run Optuna TPE hyperparameter optimization for a model "
                    "(trip-grouped CV on training trips only).")
    p.add_argument("--model", choices=ALL_MODELS, required=True,
                   help="Model to optimize.")
    p.add_argument("--strategy", choices=STRATEGIES, default="consensus",
                   help="Weak-label supervision strategy.")
    p.add_argument("--trials", type=int, default=None,
                   help="Number of Optuna trials (default depends on model).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--folds", type=int, default=3,
                   help="Number of trip-grouped CV folds.")
    p.add_argument("--no-resume", action="store_true", default=False,
                   help="Start a new study (do not resume existing SQLite study).")
    p.add_argument("--config", metavar="PATH", default=None,
                   help="Path to experiment config JSON.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from pathsnap.training import load_run_config
    config = None
    if args.config:
        import json
        from pathlib import Path
        config = json.loads(Path(args.config).read_text())
    else:
        config = load_run_config()
    n_trials = args.trials if args.trials is not None else \
        DEFAULT_N_TRIALS.get(args.model, 30)
    try:
        optimize(args.model, strategy=args.strategy, n_trials=n_trials,
                 seed=args.seed, n_folds=args.folds,
                 resume=not args.no_resume, config=config)
    except Exception as exc:
        print(f"[pathsnap.optimize] {args.model} FAILED: {exc}",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
