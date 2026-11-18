"""python -m pathsnap.predict --model MODEL_NAME --trip TRIP_ID [--split SPLIT] [--run-id RUN_ID]"""
from __future__ import annotations

import argparse
import sys

from pathsnap.prediction import predict_trip
from pathsnap.training import ALL_MODELS


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.predict",
        description="Load a checkpoint and predict a complete trip.")
    p.add_argument("--model", choices=ALL_MODELS, required=True,
                   help="Model to use for prediction")
    p.add_argument("--trip", required=True, help="Trip ID to predict")
    p.add_argument("--split", choices=["train", "student_test"],
                   default="student_test",
                   help="Split containing the trip (default: student_test)")
    p.add_argument("--run-id", metavar="RUN_ID", default=None,
                   help="Specific run ID (default: best checkpoint)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        predict_trip(args.model, args.trip, run_id=args.run_id, split=args.split)
    except Exception as exc:
        print(f"[pathsnap.predict] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
