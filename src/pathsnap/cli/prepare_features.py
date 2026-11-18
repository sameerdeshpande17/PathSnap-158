"""python -m pathsnap.prepare_features

Generate and cache candidate feature datasets. Cache keys reflect dataset
identity, feature configuration and alignment version.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pathsnap.candidates import generate, load_inputs
from pathsnap.features import (CandidateCache, dataset_identity_from_trips,
                               generate as _generate_unused)  # noqa: F401
from pathsnap.explore import ROOT

RAW_ROOT = ROOT / "data" / "raw" / "Student folder"
CACHE_ROOT = ROOT / "data" / "candidates" / "cache"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.prepare_features",
        description="Generate and cache road-candidate feature datasets.")
    p.add_argument("--split", choices=["train", "student_test", "all"],
                   default="all",
                   help="Dataset split to cache (default: all)")
    p.add_argument("--radius-m", type=float, default=50.0,
                   help="Search radius in metres (default: 50)")
    p.add_argument("--max-candidates", type=int, default=12,
                   help="Max candidates per observation (default: 12)")
    p.add_argument("--fallback", choices=["nearest", "none"], default="nearest",
                   help="Fallback for observations with no nearby candidates")
    p.add_argument("--gap-s", type=float, default=5.0,
                   help="Large-gap threshold in seconds (default: 5)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild cache even if a compatible key exists")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = dict(radius_m=args.radius_m, max_candidates=args.max_candidates,
                  fallback=args.fallback, gap_s=args.gap_s)
    cache = CandidateCache(CACHE_ROOT)
    splits = ["train", "student_test"] if args.split == "all" else [args.split]
    started = time.time()
    total_rows = 0
    for split in splits:
        raw = RAW_ROOT / split
        trip_dirs = sorted(p for p in raw.iterdir() if p.is_dir())
        print(f"[prepare_features] {split}: {len(trip_dirs)} trips",
              flush=True)
        ds_identity = dataset_identity_from_trips(trip_dirs)
        key = cache.cache_key(ds_identity, config, "phase2a-frozen")
        if cache.exists(key) and not args.force:
            print(f"[prepare_features] cache hit: {key} ({split})", flush=True)
            frame = cache.read(key)
            total_rows += len(frame)
            continue
        frames = []
        for trip_dir in trip_dirs:
            trip = load_inputs(trip_dir)
            frames.append(generate(trip, config, trip_dir.name))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        cache.write(key, frame)
        total_rows += len(frame)
        print(f"[prepare_features] wrote {key}: {len(frame):,} rows ({split})",
              flush=True)
    elapsed = time.time() - started
    manifest = cache.manifest()
    print(f"[prepare_features] done in {elapsed:.2f}s, {total_rows:,} total rows, "
          f"{len(manifest)} cached entries", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
