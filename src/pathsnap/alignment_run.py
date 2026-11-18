"""Phase 2A commands: fit, benchmark, freeze, run. No student-test tuning."""
import argparse
import bisect
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import geometry, solve, diagnostics
from .data import load_trip
from .explore import ROOT, save_json

OUT = ROOT / "experiments/alignment"
RAW = ROOT / "data/raw/Student folder"
LABELS = ROOT / "data/alignments"


def training_paths(paths):
    paths = [Path(p).resolve() for p in paths]
    expected = (RAW / "train").resolve()
    if not paths or any(p.parent != expected for p in paths):
        raise ValueError("Parameter fitting accepts training trips only")
    return paths


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit():
    paths = training_paths(sorted((RAW / "train").iterdir()))
    residuals, deviations, speeds, provenance = [], [], [], []
    for i, path in enumerate(paths):
        trip = load_trip(path)
        g = geometry(trip)
        nearest = np.argmin(g.distances, axis=1)
        residuals.extend(g.distances[np.arange(len(nearest)), nearest])
        delta = np.diff(g.progress[np.arange(len(nearest)), nearest])
        # Estimate typical progress uncertainty from nearby forward occurrences,
        # excluding gaps and supplied route breaks.
        valid = ((g.dt[1:] <= 5) & (np.diff(nearest) >= 0) &
                 (np.diff(nearest) <= 1) &
                 (np.diff(g.break_prefix[nearest]) == 0))
        deviations.extend(np.abs(delta[valid] - g.steps[1:][valid]))
        speeds.extend((g.steps[1:] / g.dt[1:])[g.dt[1:] <= 5])
        provenance.append({
            "trip_id": path.name,
            "files": {f.name: digest(f) for f in sorted(path.iterdir()) if f.is_file()},
        })
        print(f"CALIBRATE train {i + 1}/80 {path.name}", flush=True)
    sigma = float(max(5, np.median(residuals) / .67448975))
    beta = float(max(10, np.quantile(deviations, .75)))
    speed = float(np.clip(np.quantile(speeds, .99), 15, 70))
    config = {
        "sigma_m": sigma, "beta_m": beta, "max_speed_mps": speed,
        "speed_slack_m": max(30., 3 * sigma),
        "progress_weight": 1., "speed_penalty": 4.,
        "break_penalty": 8., "beam_width": 32, "candidate_count": 24,
        "residual_limit_m": max(30., 3 * sigma), "ambiguity_m": 2.,
        "gap_s": 5., "seed": 0,
    }
    save_json(OUT / "calibration.json", {
        "config": config, "fit_split": "train", "fit_trip_count": len(paths),
        "fit_observations": len(residuals), "provenance": provenance,
        "estimates": {
            "median_reference_distance_m": float(np.median(residuals)),
            "p75_progress_mismatch_m": float(np.quantile(deviations, .75)),
            "p99_observed_speed_mps": float(np.quantile(speeds, .99)),
        },
        "rationale": "Robust scale proxies, not independently estimated sensor error. "
                     "Bounds and all screening thresholds declared before student evaluation.",
    })


def metrics(g, path, config):
    distance, progress, flags, eligible = diagnostics(g, path, config)
    gaps = g.dt > config["gap_s"]
    collapsed = path[np.r_[True, np.diff(path) != 0]]
    tails = []
    for value in collapsed:
        index = bisect.bisect_right(tails, value)
        if index == len(tails):
            tails.append(value)
        else:
            tails[index] = value
    return dict(
        observations=len(path), assigned_fraction=1.,
        quality_eligible_fraction=float(eligible.mean()),
        distance_median_m=float(np.median(distance)),
        distance_p95_m=float(np.quantile(distance, .95)),
        distance_max_m=float(distance.max()),
        residual_rmse_m=float(np.sqrt(np.mean(distance ** 2))),
        route_order_violations=int(flags["route_order_violation"].sum()),
        unexplained_jumps=int(flags["unexplained_jump"].sum()),
        backward_progress_steps=int(flags["backward_progress"].sum()),
        reference_break_crossings=int(flags["reference_break_crossing"].sum()),
        geometric_ambiguity_fraction=float(flags["geometric_occurrence_ambiguity"].mean()),
        occurrence_coverage_fraction=float(len(np.unique(path)) / g.distances.shape[1]),
        sequence_monotone_subsequence_fraction=len(tails) / len(collapsed),
        gap_observations=int(gaps.sum()),
        post_gap_distance_median_m=float(np.median(distance[gaps])) if gaps.any() else None,
        regular_distance_median_m=float(np.median(distance[~gaps])),
        gap_order_violations=int((flags["route_order_violation"] & gaps).sum()),
        gap_unexplained_jumps=int((flags["unexplained_jump"] & gaps).sum()),
        high_residual_observations=int(flags["high_residual"].sum()),
    )


def benchmark():
    config = json.loads((OUT / "calibration.json").read_text())["config"]
    stats = pd.read_csv(ROOT / "experiments/trip_statistics.csv", dtype={"trip_id": str})
    train = stats[stats.split == "train"].copy()
    train["cells"] = train.gps_count * train.route_count
    ids = list(dict.fromkeys([
        train.loc[train.cells.idxmax(), "trip_id"],
        train.loc[train.max_gap_s.idxmax(), "trip_id"],
        train.loc[train.route_repeated_arc_occurrences.idxmax(), "trip_id"],
        train.loc[train.route_breaks.idxmax(), "trip_id"],
    ]))
    rows, comparisons = [], []
    variants = [
        ("base", {}),
        ("wide", {"beam_width": 64, "candidate_count": 48}),
        ("half_sigma", {"sigma_m": config["sigma_m"] / 2}),
        ("double_sigma", {"sigma_m": config["sigma_m"] * 2}),
    ]
    for trip_id in ids:
        trip = load_trip(RAW / "train" / trip_id)
        g = geometry(trip)
        paths = {}
        for variant, changes in variants:
            cfg = config | changes
            for method in ["A", "B", "C"]:
                if method == "A" and variant != "base":
                    continue
                start = time.perf_counter()
                path, details = solve(g, method, cfg)
                elapsed = time.perf_counter() - start
                paths[(variant, method)] = path
                rows.append(dict(trip_id=trip_id, split="train", variant=variant,
                                 method=method, runtime_s=elapsed, **metrics(g, path, config)))
                if variant != "base":
                    base = paths[("base", method)]
                    comparisons.append(dict(
                        trip_id=trip_id, method=method, variant=variant,
                        occurrence_disagreement_fraction=float(np.mean(path != base)),
                        arc_disagreement_fraction=float(np.mean(trip.route[path] != trip.route[base])),
                    ))
                print(f"BENCHMARK {trip_id} {variant} {method}: {elapsed:.3f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "training_benchmark.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "training_sensitivity.csv", index=False)
    save_json(OUT / "benchmark_selection.json", {
        "training_trip_ids": ids,
        "selection": "max N*M, max gap, most repeated occurrences, most reference breaks",
        "no_student_data_used": True,
    })


def freeze():
    calibration = json.loads((OUT / "calibration.json").read_text())
    sensitivity = pd.read_csv(OUT / "training_sensitivity.csv")
    wide = sensitivity[sensitivity.variant == "wide"]
    cfg = calibration["config"].copy()
    # Predeclared rule: widen if any representative training trip changes >1%.
    widen = bool((wide.arc_disagreement_fraction > .01).any())
    if widen:
        cfg.update(beam_width=64, candidate_count=48)
    frozen = {
        "config": cfg, "fit_split": "train", "fit_trip_count": 80,
        "calibration_sha256": digest(OUT / "calibration.json"),
        "sensitivity_sha256": digest(OUT / "training_sensitivity.csv"),
        "parameter_freeze_before_student_evaluation": True,
        "width_rule": "Use 64/48 if any representative training arc disagreement "
                      "between 32/24 and 64/48 exceeds 1%; otherwise 32/24.",
        "widened": widen,
        "method_A": "Nearest route-occurrence segment; earliest occurrence resolves ties.",
        "method_B": "Beam-pruned monotone DP, L1 distance/progression cost and soft speed penalty.",
        "method_C": "Beam-pruned conditional probabilistic chain: Gaussian emissions; "
                    "row-normalized Laplace progression transitions, hard speed constraint "
                    "on occurrence changes except at documented reference breaks; "
                    "self-occurrence projection jumps remain flagged; log-space Viterbi.",
        "confidence": "Exported exp(-distance^2/(2*sigma^2)) is only a spatial compatibility "
                      "score, not posterior probability or calibrated correctness.",
        "truth_status": "weak/inferred supervision; no independent pointwise truth",
        "reference_features_for_inference": "forbidden",
        "consensus_policy": "B and C same arc AND occurrence, both quality eligible, "
                            "no geometric occurrence ambiguity. No correctness claim.",
        "no_random_sampling": True,
    }
    save_json(OUT / "frozen_methods.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)


def evaluate():
    frozen_path = OUT / "frozen_methods.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["fit_split"] == "train"
    config = frozen["config"]
    config_hash = digest(frozen_path)
    rows, differences, runtimes = [], [], []
    started = time.perf_counter()
    for split in ["train", "student_test"]:
        folder = LABELS / split
        folder.mkdir(parents=True, exist_ok=True)
        for directory in sorted((RAW / split).iterdir()):
            begin = time.perf_counter()
            trip = load_trip(directory)
            geometry_start = time.perf_counter()
            g = geometry(trip)
            geometry_s = time.perf_counter() - geometry_start
            outputs, paths, eligibility = {}, {}, {}
            for method in ["A", "B", "C"]:
                timer = time.perf_counter()
                path, detail = solve(g, method, config, lambda i, n: print(
                    f"  DP {split}/{directory.name} {method} {i}/{n}", flush=True))
                seconds = time.perf_counter() - timer
                paths[method] = path
                distance, progress, flags, eligible = diagnostics(g, path, config)
                eligibility[method] = eligible
                frame = pd.DataFrame({
                    "trip_id": directory.name, "split": split,
                    "observation_index": np.arange(len(path)),
                    "longitude": trip.track[:, 0], "latitude": trip.track[:, 1],
                    "timestamp": trip.track[:, 2], "assigned_arc_id": trip.route[path],
                    "assigned_occurrence_index": path, "alignment_method": method,
                    "projection_distance_m": distance, "route_progress_m": progress,
                    "spatial_compatibility_score": np.exp(-.5 * (distance / config["sigma_m"]) ** 2),
                    "quality_eligible": eligible,
                    "quality_flags": [";".join(k for k, v in flags.items() if v[i])
                                      for i in range(len(path))],
                    "label_status": "weak_inferred",
                })
                outputs[method] = frame
                rows.append(dict(split=split, trip_id=directory.name, method=method,
                                 runtime_s=seconds, geometry_s=geometry_s,
                                 config_sha256=config_hash, **detail,
                                 **metrics(g, path, config)))
                print(f"ALIGN {split}/{directory.name} {method} {seconds:.3f}s "
                      f"eligible={eligible.mean():.3%}", flush=True)
            agreement = paths["B"] == paths["C"]
            ambiguous = diagnostics(g, paths["C"], config)[2]["geometric_occurrence_ambiguity"]
            consensus = agreement & eligibility["B"] & eligibility["C"] & ~ambiguous
            for method, frame in outputs.items():
                frame["bc_occurrence_agreement"] = agreement
                frame["recommended_supervision_use"] = consensus & (method == "C")
                frame.to_csv(folder / f"{directory.name}_{method}.csv", index=False)
            for first, second in [("A", "B"), ("A", "C"), ("B", "C")]:
                differences.append(dict(
                    split=split, trip_id=directory.name, first=first, second=second,
                    observations=len(trip.track),
                    arc_disagreement_fraction=float(np.mean(trip.route[paths[first]] != trip.route[paths[second]])),
                    occurrence_disagreement_fraction=float(np.mean(paths[first] != paths[second])),
                    bc_consensus_eligible_count=int(consensus.sum()),
                ))
            runtimes.append(dict(split=split, trip_id=directory.name,
                                 total_wall_s=time.perf_counter() - begin))
            assert digest(frozen_path) == config_hash
    pd.DataFrame(rows).to_csv(ROOT / "experiments/alignment_comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(OUT / "method_disagreements.csv", index=False)
    pd.DataFrame(runtimes).to_csv(OUT / "trip_wall_times.csv", index=False)
    save_json(OUT / "run.json", {
        "elapsed_s": time.perf_counter() - started, "trip_method_runs": len(rows),
        "config_sha256_before_and_after": config_hash,
        "models_trained": 0, "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "pandas", "pyproj", "folium", "reportlab"]},
        "timing_definition": "method runtime includes candidate ordering and DP; "
                             "geometry time separate; total wall includes parsing and exports",
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "benchmark", "freeze", "run"])
    args = parser.parse_args()
    {"fit": fit, "benchmark": benchmark, "freeze": freeze, "run": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
