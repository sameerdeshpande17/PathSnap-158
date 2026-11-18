"""python -m pathsnap.decode_route --trip TRIP_ID --model MODEL [--split SPLIT]

Decode a single trip: classifier + HMM + Viterbi + route reconstruction.

Does NOT access .route files. Uses track/nodes/arcs + trained classifier only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pathsnap.explore import ROOT
from pathsnap.hmm import HMMSettings
from pathsnap.map_matching_pipeline import MapMatchingPipeline, PipelineConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pathsnap.decode_route",
        description="Decode a GPS trip into a connected road-arc route "
        "(classifier + HMM + Viterbi).",
    )
    p.add_argument("--trip", required=True, help="Trip ID to decode")
    p.add_argument(
        "--split", choices=["train", "student_test"], default="student_test",
        help="Dataset split containing the trip (default: student_test)",
    )
    p.add_argument(
        "--model", required=True,
        help="Model name to use (e.g. hist_gradient_boosting, random_forest, mlp, ...)",
    )
    p.add_argument(
        "--run-dir", metavar="PATH", default=None,
        help="Specific model run directory (default: best checkpoint for --model)",
    )
    p.add_argument(
        "--beam-width", type=int, default=64,
        help="Viterbi beam width (None = no pruning; default: 64)",
    )
    p.add_argument(
        "--max-candidates", type=int, default=32,
        help="Max candidates per observation (default: 32)",
    )
    p.add_argument(
        "--radius-m", type=float, default=50.0,
        help="Candidate search radius in metres (default: 50)",
    )
    p.add_argument(
        "--output", metavar="PATH", default=None,
        help="Save prediction JSON to this path",
    )
    return p.parse_args(argv)


def _find_best_run(model_name: str) -> str | None:
    """Find the best checkpoint directory for a model."""
    from pathsnap.training import load_best

    best = load_best(model_name)
    if best is None:
        return None
    return best.get("best_run_id")


def main(argv=None) -> int:
    args = parse_args(argv)

    # Resolve model run directory
    run_dir = args.run_dir
    if run_dir is None:
        best_run_id = _find_best_run(args.model)
        if best_run_id is None:
            print(
                f"[pathsnap.decode_route] ERROR: no trained checkpoint found for "
                f"model '{args.model}'. Train one first with pathsnap.train.",
                file=sys.stderr,
            )
            return 1
        from pathsnap.training import run_dir as _run_dir
        run_dir = str(_run_dir(args.model, best_run_id))

    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"[pathsnap.decode_route] ERROR: run directory not found: {run_dir}",
              file=sys.stderr)
        return 1

    # Build HMM settings from CLI
    hmm_settings = HMMSettings(
        beam_width=args.beam_width if args.beam_width > 0 else None,
        max_candidates=args.max_candidates,
    )

    config = PipelineConfig(
        model_name=args.model,
        model_run_dir=run_path,
        hmm_settings=hmm_settings,
        candidate_radius_m=args.radius_m,
        candidate_max_candidates=args.max_candidates,
    )

    try:
        pipeline = MapMatchingPipeline(config)
        t0 = time.perf_counter()
        pred = pipeline.decode_trip(args.trip, split=args.split)
        total = time.perf_counter() - t0
    except Exception as exc:
        print(f"[pathsnap.decode_route] ERROR: {exc}", file=sys.stderr)
        return 1

    # Build output
    output = dict(
        trip_id=pred.trip_id,
        split=pred.split,
        model=args.model,
        model_run_dir=str(run_path),
        device=pred.device,
        n_observations=pred.n_observations,
        n_candidates=pred.n_candidates,
        classifier_top1_arcs=pred.classifier_top1_arc,
        classifier_top1_probs=pred.classifier_top1_prob,
        hmm_decoded_arcs=pred.hmm_decoded_arcs,
        hmm_log_prob=pred.hmm_log_prob,
        hmm_n_connected=pred.hmm_n_connected,
        hmm_n_disconnected=pred.hmm_n_disconnected,
        hmm_n_dwells=pred.hmm_n_dwells,
        route=(
            dict(
                n_segments=pred.route.n_route_segments,
                n_connectors=pred.route.n_connector_segments,
                n_disconnected_gaps=pred.route.n_disconnected_gaps,
                n_repeated_traversals=pred.route.n_repeated_arc_traversals,
                n_dwells=pred.route.n_dwell_transitions,
                total_length_m=pred.route.total_length_m,
                arc_id_sequence=pred.route.arc_id_sequence,
                segments=[
                    dict(
                        arc_id=s.arc_id,
                        from_node=s.from_node,
                        to_node=s.to_node,
                        length_m=s.length_m,
                        kind=s.kind,
                        observation_indices=s.observation_indices,
                        connects_from=s.connects_from,
                        connects_to=s.connects_to,
                    )
                    for s in pred.route.segments
                ],
            )
            if pred.route is not None
            else None
        ),
        timing_s=dict(
            candidate_gen=pred.candidate_gen_s,
            classifier_score=pred.classifier_score_s,
            viterbi=pred.viterbi_s,
            route_reconstruct=pred.route_reconstruct_s,
            total=pred.total_s,
        ),
        warnings=pred.warnings,
    )

    print(json.dumps(output, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[pathsnap.decode_route] output saved to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
