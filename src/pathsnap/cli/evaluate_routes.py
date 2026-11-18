"""python -m pathsnap.evaluate_routes --split student_test [--model MODEL]

Evaluate classifier-only vs HMM/Viterbi route predictions on a split.

Uses trained model + HMM decoder. Compares:
1. Independent classifier Top-1 predictions
2. Classifier + HMM/Viterbi predictions
3. Expanded connected routes

Metrics are weak-label agreement (NOT verified ground-truth accuracy).
Does NOT retrain models.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from pathsnap.explore import ROOT
from pathsnap.hmm import HMMSettings
from pathsnap.map_matching_pipeline import MapMatchingPipeline, PipelineConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.evaluate_routes",
        description="Evaluate route predictions on a dataset split.",
    )
    p.add_argument(
        "--split", choices=["train", "student_test"], default="student_test",
        help="Dataset split to evaluate (default: student_test)",
    )
    p.add_argument(
        "--model", required=True,
        help="Model name to use (e.g. hist_gradient_boosting)",
    )
    p.add_argument(
        "--run-dir", metavar="PATH", default=None,
        help="Specific model run directory (default: best checkpoint)",
    )
    p.add_argument(
        "--trip", metavar="TRIP_ID", default=None,
        help="Evaluate only this single trip (default: all trips in split)",
    )
    p.add_argument(
        "--beam-width", type=int, default=64,
        help="Viterbi beam width (0 = no pruning; default: 64)",
    )
    p.add_argument(
        "--max-candidates", type=int, default=32,
        help="Max candidates per observation (default: 32)",
    )
    p.add_argument(
        "--output-csv", metavar="PATH", default=None,
        help="Save per-trip evaluation CSV to this path",
    )
    p.add_argument(
        "--output-json", metavar="PATH", default=None,
        help="Save summary JSON to this path",
    )
    return p.parse_args(argv)


def _find_best_run(model_name: str) -> str | None:
    from pathsnap.training import load_best

    best = load_best(model_name)
    if best is None:
        return None
    return best.get("best_run_id")


def main(argv=None) -> int:
    args = parse_args(argv)

    run_dir = args.run_dir
    if run_dir is None:
        best_run_id = _find_best_run(args.model)
        if best_run_id is None:
            print(
                f"[pathsnap.evaluate_routes] ERROR: no trained checkpoint for "
                f"model '{args.model}'. Train one first.",
                file=sys.stderr,
            )
            return 1
        from pathsnap.training import run_dir as _run_dir
        run_dir = str(_run_dir(args.model, best_run_id))

    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"[pathsnap.evaluate_routes] ERROR: run dir not found: {run_dir}",
              file=sys.stderr)
        return 1

    hmm_settings = HMMSettings(
        beam_width=args.beam_width if args.beam_width > 0 else None,
        max_candidates=args.max_candidates,
    )
    config = PipelineConfig(
        model_name=args.model,
        model_run_dir=run_path,
        hmm_settings=hmm_settings,
        candidate_max_candidates=args.max_candidates,
    )

    try:
        pipeline = MapMatchingPipeline(config)
    except Exception as exc:
        print(f"[pathsnap.evaluate_routes] ERROR loading model: {exc}",
              file=sys.stderr)
        return 1

    # Determine trips to evaluate
    raw = ROOT / "data" / "raw" / "Student folder" / args.split
    if args.trip:
        trip_ids = [args.trip]
    else:
        trip_ids = sorted(p.name for p in raw.iterdir() if p.is_dir())

    results = []
    overall_start = time.perf_counter()

    for trip_id in trip_ids:
        try:
            pred = pipeline.decode_trip(trip_id, split=args.split)
        except Exception as exc:
            print(f"[pathsnap.evaluate_routes] ERROR on {trip_id}: {exc}",
                  file=sys.stderr)
            continue

        # Classifier-only metrics (weak-label agreement)
        classifier_top1 = pred.classifier_top1_arc
        # HMM metrics
        hmm_arcs = pred.hmm_decoded_arcs

        # Agreement between classifier and HMM
        n_obs = pred.n_observations
        agree_cls_hmm = sum(
            1 for a, b in zip(classifier_top1, hmm_arcs) if a == b and a >= 0
        )
        agree_rate = agree_cls_hmm / max(n_obs, 1)

        route = pred.route
        result = dict(
            trip_id=trip_id,
            n_observations=n_obs,
            n_candidates=pred.n_candidates,
            classifier_top1_arcs=classifier_top1,
            hmm_decoded_arcs=hmm_arcs,
            classifier_hmm_agreement_rate=agree_rate,
            hmm_log_prob=pred.hmm_log_prob,
            hmm_n_connected=pred.hmm_n_connected,
            hmm_n_disconnected=pred.hmm_n_disconnected,
            hmm_n_dwells=pred.hmm_n_dwells,
            route_n_segments=route.n_route_segments if route else 0,
            route_n_connectors=route.n_connector_segments if route else 0,
            route_n_disconnected_gaps=route.n_disconnected_gaps if route else 0,
            route_n_repeated_traversals=route.n_repeated_arc_traversals if route else 0,
            route_total_length_m=route.total_length_m if route else 0.0,
            timing_s=pred.total_s,
        )
        results.append(result)
        print(f"  {trip_id}: {n_obs} obs, classifier-HMM agree {agree_rate:.3f}, "
              f"route gaps={result['route_n_disconnected_gaps']}", flush=True)

    overall_elapsed = time.perf_counter() - overall_start

    summary = dict(
        model=args.model,
        model_run_dir=str(run_path),
        split=args.split,
        n_trips=len(results),
        total_observations=sum(r["n_observations"] for r in results),
        total_candidates=sum(r["n_candidates"] for r in results),
        overall_elapsed_s=overall_elapsed,
        mean_classifier_hmm_agreement=(
            np.mean([r["classifier_hmm_agreement_rate"] for r in results])
            if results else 0.0
        ),
        total_route_gaps=sum(r["route_n_disconnected_gaps"] for r in results),
        total_repeated_traversals=sum(r["route_n_repeated_traversals"] for r in results),
        results=results,
    )

    print(json.dumps(summary, indent=2))

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trip_id", "n_observations", "n_candidates",
                    "classifier_hmm_agreement_rate", "hmm_log_prob",
                    "hmm_n_connected", "hmm_n_disconnected", "hmm_n_dwells",
                    "route_n_segments", "route_n_connectors",
                    "route_n_disconnected_gaps", "route_n_repeated_traversals",
                    "route_total_length_m", "timing_s",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        print(f"[pathsnap.evaluate_routes] CSV saved to {out_csv}", flush=True)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"[pathsnap.evaluate_routes] JSON saved to {out_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
