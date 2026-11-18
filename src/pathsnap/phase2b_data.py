"""Offline weak-label joins and frozen Phase 2B experiment setup."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from .candidates import FEATURES, SCHEMA_VERSION, generate, load_inputs
from .explore import ROOT, save_json

OUT = ROOT / "experiments/phase2b"
DATA = ROOT / "data/candidates"
ARTIFACTS = ROOT / "artifacts/phase2b"
RAW = ROOT / "data/raw/Student folder"
MODELS = ["logistic", "knn", "random_forest", "hist_gradient_boosting", "mlp"]
STRATEGIES = ["consensus", "consensus_plus_B"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize():
    for path in [OUT, DATA, ARTIFACTS, ROOT / "visualizations/phase2b"]:
        path.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": 42, "folds": 3, "cpu_threads": 4,
        "candidates": {"radius_m": 50., "max_candidates": 12,
                       "fallback": "nearest", "gap_s": 5.},
        "consensus_observation_budget": 20000, "B_extra_observation_budget": 5000,
        "models": {
            "logistic": {"C": 1., "max_iter": 500},
            "knn": {"n_neighbors": 11, "weights": "distance", "algorithm": "ball_tree",
                    "leaf_size": 40, "n_jobs": 4},
            "random_forest": {"n_estimators": 100, "max_depth": 18, "min_samples_leaf": 5,
                              "max_features": "sqrt", "n_jobs": 4},
            "hist_gradient_boosting": {"max_iter": 150, "learning_rate": .08,
                                       "max_leaf_nodes": 31, "l2_regularization": 1.,
                                       "early_stopping": False},
            "mlp": {"hidden": [64, 32], "epochs": 15, "batch_size": 4096,
                    "learning_rate": .001, "weight_decay": .0001, "device": "cuda"},
        },
        "feature_schema": FEATURES, "feature_version": SCHEMA_VERSION,
        "primary_selection": "CV trip-macro observation Top-1 weak-label agreement on "
                             "fixed consensus targets, including candidate misses",
        "selection_tiebreakers": ["CV candidate Macro F1", "CV MRR", "lower prediction latency"],
        "binary_threshold": .5,
        "weak_target_policy": "Use fixed Phase 2A targets; CV is conditional on that "
                              "pre-existing weak-label corpus, whose calibration used all "
                              "80 training trips. No predictor sees reference-derived features.",
        "supplement_policy": "Same sampled consensus observations plus up to 5000 B-only "
                             "quality-eligible, geometrically unambiguous observations. "
                             "Never substitute C's rejected labels.",
        "candidate_misses": "Exclude all rows of missed-positive observations from fitting; "
                            "count as Top-1 and reciprocal-rank zero during evaluation",
        "negative_policy": "All generated candidates for sampled covered observations; "
                           "no class weights, no label-dependent candidate injection",
        "training_sampling": "Seeded trip-balanced observation sampling within fitting trips only",
        "cv_target": "Same consensus observations for both supervision strategies; B-only "
                     "evaluation is a separately reported secondary stratum",
        "student_policy": "Generate/evaluate student candidates only after CV selection is frozen",
    }
    save_json(OUT / "config.json", config)
    ids = sorted(p.name for p in (RAW / "train").iterdir() if p.is_dir())
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = []
    for index, (train, valid) in enumerate(splitter.split(ids, groups=ids)):
        folds.append({"fold": index, "train_trips": [ids[i] for i in train],
                      "validation_trips": [ids[i] for i in valid]})
    save_json(OUT / "folds.json", folds)
    save_json(OUT / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sklearn_models_device": "CPU (no CUDA device parameter in selected estimators)",
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "phase2a_frozen_methods_sha256": sha(ROOT / "experiments/alignment/frozen_methods.json"),
        "new_dependencies_installed": [],
    })
    print("Frozen experiment design and 3 trip-grouped folds.", flush=True)


def label_table(split, trip_id):
    root = ROOT / "data/alignments" / split
    b = pd.read_csv(root / f"{trip_id}_B.csv", dtype={"trip_id": str})
    c = pd.read_csv(root / f"{trip_id}_C.csv", dtype={"trip_id": str})
    consensus = c.recommended_supervision_use.to_numpy()
    b_good = b.quality_eligible & ~b.quality_flags.fillna("").str.contains("geometric_occurrence_ambiguity")
    assert (b_good[consensus]).all()
    np.testing.assert_array_equal(b.assigned_arc_id[consensus], c.assigned_arc_id[consensus])
    return pd.DataFrame({
        "trip_id": trip_id, "observation_index": b.observation_index,
        "consensus_eligible": consensus, "B_extra_eligible": b_good.to_numpy() & ~consensus,
        "weak_arc_id": b.assigned_arc_id,  # equal to C on consensus subset
    })


def build(split):
    if split == "student_test" and not (OUT / "selection.json").exists():
        raise ValueError("Student generation blocked until training-CV selection is frozen")
    cfg_path = OUT / "config.json"
    cfg = json.loads(cfg_path.read_text())
    config_sha = sha(cfg_path)
    (DATA / split).mkdir(parents=True, exist_ok=True)
    statistics, recall_curve = [], []
    for directory in sorted((RAW / split).iterdir()):
        begin = time.perf_counter()
        trip = load_inputs(directory)
        gen_start = time.perf_counter()
        frame = generate(trip, cfg["candidates"], directory.name)
        generation_s = time.perf_counter() - gen_start
        # Explicit offline join after inference-only feature construction.
        labels = label_table(split, directory.name)
        obs = frame.observation_index.to_numpy()
        positive = frame.arc_id.to_numpy() == labels.weak_arc_id.to_numpy()[obs]
        covered_ids = frame.observation_index.to_numpy()[positive]
        labels["positive_in_candidates"] = False
        labels.loc[covered_ids, "positive_in_candidates"] = True
        frame["weak_target"] = np.where(
            (labels.consensus_eligible | labels.B_extra_eligible).to_numpy()[obs], positive, -1).astype(np.int8)
        frame["weak_arc_id"] = labels.weak_arc_id.to_numpy()[obs]
        frame["consensus_eligible"] = labels.consensus_eligible.to_numpy()[obs]
        frame["B_extra_eligible"] = labels.B_extra_eligible.to_numpy()[obs]
        frame["positive_in_candidates"] = labels.positive_in_candidates.to_numpy()[obs]
        frame.to_parquet(DATA / split / f"{directory.name}.parquet", index=False)
        labels.to_parquet(DATA / split / f"{directory.name}_observations.parquet", index=False)
        for strategy, mask in [
            ("consensus", labels.consensus_eligible),
            ("B_extra", labels.B_extra_eligible),
            ("consensus_plus_B", labels.consensus_eligible | labels.B_extra_eligible),
        ]:
            eligible = int(mask.sum())
            count = int((mask & labels.positive_in_candidates).sum())
            statistics.append(dict(
                split=split, trip_id=directory.name, supervision=strategy,
                observations=len(trip.track), candidate_rows=len(frame),
                weak_labeled_observations=eligible, covered_weak_observations=count,
                weak_label_candidate_recall=count / eligible if eligible else None,
                observations_without_candidates=int(len(trip.track) - frame.observation_index.nunique()),
                fallback_observations=int(frame.loc[frame.fallback_used == 1, "observation_index"].nunique()),
                generation_s=generation_s, total_s=time.perf_counter() - begin,
            ))
        if split == "train":
            for k in [1, 2, 4, 8, 12]:
                hits = frame.loc[positive & (frame.candidate_rank < k), "observation_index"].to_numpy()
                covered = np.zeros(len(labels), bool)
                covered[hits] = True
                recall_curve.append(dict(trip_id=directory.name, candidate_cap=k,
                    labeled=int(labels.consensus_eligible.sum()),
                    covered=int((covered & labels.consensus_eligible).sum())))
        print(f"CANDIDATES {split}/{directory.name}: {len(frame):,} rows "
              f"{generation_s:.3f}s", flush=True)
    pd.DataFrame(statistics).to_csv(OUT / f"candidates_{split}.csv", index=False)
    if recall_curve:
        pd.DataFrame(recall_curve).to_csv(OUT / "train_candidate_cap_diagnostic.csv", index=False)
    tables = [pd.read_csv(p, dtype={"trip_id": str}) for p in OUT.glob("candidates_*.csv")]
    pd.concat(tables, ignore_index=True).to_csv(ROOT / "experiments/candidate_generation_statistics.csv", index=False)
    assert sha(cfg_path) == config_sha


def load_partition(split):
    files = sorted(p for p in (DATA / split).glob("*.parquet") if "_observations" not in p.name)
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def sample_observations(frame, trip_ids, strategy, config):
    """Return complete candidate groups; never sample rows independently."""
    available = frame[frame.trip_id.isin(trip_ids) & frame.positive_in_candidates]
    unique = available[["trip_id", "observation_index", "consensus_eligible", "B_extra_eligible"]].drop_duplicates()
    selected = []
    for mask_name, budget in [
        ("consensus_eligible", config["consensus_observation_budget"]),
        ("B_extra_eligible", config["B_extra_observation_budget"] if strategy == "consensus_plus_B" else 0),
    ]:
        groups = unique[unique[mask_name]].groupby("trip_id", sort=True)
        rng = np.random.default_rng(config["seed"])
        queues = [group.iloc[rng.permutation(len(group))] for _, group in groups]
        # Round-robin trip balance without dependence on candidate multiplicity.
        rounds = []
        for group in queues:
            copy = group.copy()
            copy["_round"] = np.arange(len(copy))
            rounds.append(copy)
        if rounds and budget:
            sample = pd.concat(rounds).sort_values(["_round", "trip_id"]).head(budget)
            selected.append(sample[["trip_id", "observation_index"]])
    if not selected:
        raise ValueError("No supervised fitting observations")
    keys = pd.concat(selected, ignore_index=True)
    result = available.merge(keys, on=["trip_id", "observation_index"], how="inner")
    result = result.sort_values(["trip_id", "observation_index", "candidate_rank"]).reset_index(drop=True)
    return result, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["init", "train", "student_test"])
    args = parser.parse_args()
    initialize() if args.stage == "init" else build(args.stage)


if __name__ == "__main__":
    main()
