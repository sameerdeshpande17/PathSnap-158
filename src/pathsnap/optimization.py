"""Optuna Bayesian hyperparameter optimization for PathSnap candidate ranking.

Uses trip-grouped cross-validation strictly within the 80 training trips.
The student-testing partition is never used for tuning, trial selection,
calibration, early stopping, or repeated model-selection decisions.

Primary objective: observation-level Top-1 weak-label agreement (trip-macro
across CV folds). Candidate log loss and per-trip stability are secondary
diagnostics. Candidate-level binary accuracy is deliberately NOT the primary
objective because negatives dominate.

All trials are persisted to SQLite under artifacts/optuna/ and studies can be
resumed after interruption. No trained checkpoint is ever overwritten.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from pathsnap.candidates import generate, load_inputs
from pathsnap.features import DEFAULT_CANDIDATE_CONFIG
from pathsnap.explore import ROOT
from pathsnap.labels import (
    STRATEGIES, build_label_table, join_labels_with_candidates,
    sample_observations,
)
from pathsnap.models import ALL_MODELS, MODEL_REGISTRY
from pathsnap.splits import available_trips, make_folds
from pathsnap.training import (
    ARTIFACTS_ROOT, environment_snapshot, load_run_config,
)

OPTUNA_ROOT = ARTIFACTS_ROOT / "optuna"

DEFAULT_N_TRIALS = {"hist_gradient_boosting": 50, "random_forest": 30,
                    "mlp": 30}
DEFAULT_N_FOLDS = 3
DEFAULT_SEED = 42


def study_db_path(model_name: str, strategy: str) -> Path:
    OPTUNA_ROOT.mkdir(parents=True, exist_ok=True)
    return OPTUNA_ROOT / f"phase3a_{model_name}_{strategy}.db"


def study_name(model_name: str, strategy: str) -> str:
    return f"phase3a_{model_name}_{strategy}"


def _build_search_space(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    if model_name == "hist_gradient_boosting":
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 2e-1,
                                              log=True),
            max_iter=trial.suggest_int("max_iter", 50, 500),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 127),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4,
                                                  1.0, log=True),
            max_bins=trial.suggest_int("max_bins", 50, 255),
            early_stopping=False,
        )
    if model_name == "random_forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced"]),
            n_jobs=4,
        )
    if model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden = []
        for i in range(n_layers):
            hidden.append(trial.suggest_int(f"hidden_{i}", 16, 256))
        return dict(
            hidden=hidden,
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2,
                                              log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2,
                                             log=True),
            batch_size=trial.suggest_categorical(
                "batch_size", [256, 512, 1024, 2048, 4096]),
            epochs=trial.suggest_int("epochs", 10, 60),
            device="cuda",
        )
    raise ValueError(f"Unknown model {model_name}")


@dataclass
class FoldResult:
    fold: int
    model: str
    strategy: str
    train_trips: list[str]
    val_trips: list[str]
    n_train_rows: int
    n_val_observations: int
    train_seconds: float
    top1_agreement: float
    top1_agreement_covered: float
    mrr: float
    candidate_macro_f1: float
    candidate_log_loss: float
    candidate_miss_rate: float
    per_trip_top1: dict[str, float]


def _evaluate_frame_metrics(frame, feature_schema, model_wrapper,
                            consensus_only: bool):
    """Score a validation frame and return observation + candidate metrics."""
    from pathsnap.phase2b_metrics import evaluate_frame

    X = np.ascontiguousarray(frame[feature_schema].to_numpy(dtype=np.float32))
    scores = model_wrapper.predict_proba(X)
    frame = frame.copy()
    frame["candidate_score"] = scores
    obs, cand = evaluate_frame(frame, consensus_only=consensus_only)
    return obs, cand


def run_cv_fold(model_name: str, params: dict, train_frame,
                val_frame, feature_schema, strategy, consensus_only,
                seed):
    """Train on train_frame, evaluate on val_frame. Returns FoldResult."""
    import time as _time
    from sklearn.preprocessing import StandardScaler

    X = np.ascontiguousarray(train_frame[feature_schema].to_numpy(dtype=np.float32))
    y = train_frame.weak_target.to_numpy(dtype=np.float64)
    covered = y >= 0
    X, y = X[covered], y[covered]

    t0 = _time.time()
    wrapper_cls = MODEL_REGISTRY[model_name]
    wrapper = wrapper_cls(name=model_name, params=params,
                          n_features=X.shape[1],
                          feature_names=list(feature_schema),
                          random_state=seed)
    # For MLP, pass a small validation slice for early-stopping diagnostics.
    X_val = np.ascontiguousarray(
        val_frame[feature_schema].to_numpy(dtype=np.float32))
    y_val = val_frame.weak_target.to_numpy(dtype=np.float64)
    val_covered = y_val >= 0
    wrapper.fit(X, y,
                X_val=X_val[val_covered],
                y_val=y_val[val_covered])
    train_s = _time.time() - t0

    # Score the validation frame (assign candidate_score so per-trip diagnostics
    # and evaluate_frame share the same scored frame).
    Xv = np.ascontiguousarray(val_frame[feature_schema].to_numpy(dtype=np.float32))
    val_frame = val_frame.copy()
    val_frame["candidate_score"] = wrapper.predict_proba(Xv)

    obs, cand = _evaluate_frame_metrics(val_frame, feature_schema, wrapper,
                                        consensus_only)

    # Per-trip Top-1 for stability diagnostic.
    per_trip = {}
    val_consensus = val_frame[val_frame.consensus_eligible]
    from pathsnap.phase2b_metrics import top1_arc
    best = top1_arc(val_frame)
    weak = val_frame.set_index(["trip_id", "observation_index"]).weak_arc_id
    for trip_id, group in val_consensus.groupby("trip_id", sort=True):
        keys = list(zip(group.trip_id, group.observation_index))
        top_map = dict(zip(best.index, best.to_numpy()))
        weak_map = dict(zip(weak.index, weak.to_numpy()))
        agree = [top_map.get(k) == weak_map.get(k) for k in keys]
        per_trip[trip_id] = float(np.mean(agree)) if agree else 0.0

    return FoldResult(
        fold=-1, model=model_name, strategy=strategy,
        train_trips=sorted(train_frame.trip_id.unique().tolist()),
        val_trips=sorted(val_frame.trip_id.unique().tolist()),
        n_train_rows=int(len(X)), n_val_observations=int(obs["observation_count"]),
        train_seconds=train_s,
        top1_agreement=float(obs["top1_agreement"]),
        top1_agreement_covered=float(obs["top1_agreement_covered_only"]),
        mrr=float(obs["mrr"]),
        candidate_macro_f1=float(cand["candidate_macro_f1"]),
        candidate_log_loss=float(cand["candidate_log_loss"]),
        candidate_miss_rate=float(obs["candidate_miss_rate"]),
        per_trip_top1=per_trip,
    )


def build_training_frame_full(split: str, trip_ids: list[str], strategy: str,
                              config: dict, cache) -> pd.DataFrame:
    """Build the full labeled candidate frame for given trips (no sampling)."""
    import pandas as pd
    frames = []
    raw = ROOT / "data" / "raw" / "Student folder"
    for trip_id in trip_ids:
        trip_dir = raw / split / trip_id
        trip = load_inputs(trip_dir)
        gen = generate(trip, config["candidates"], trip_id)
        labels = build_label_table(split, trip_id)
        frames.append(join_labels_with_candidates(gen, labels))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frame


def optimize(model_name: str, strategy: str = "consensus",
             n_trials: int | None = None, seed: int = DEFAULT_SEED,
             n_folds: int = DEFAULT_N_FOLDS, resume: bool = True,
             config: dict | None = None) -> dict[str, Any]:
    """Run Optuna TPE optimization for one model.

    Returns the study summary dict. Does NOT retrain a final model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name}. Available: {ALL_MODELS}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}. Available: {STRATEGIES}")
    config = config or load_run_config()
    if n_trials is None:
        n_trials = DEFAULT_N_TRIALS.get(model_name, 30)

    from pathsnap.features import CandidateCache, dataset_identity_from_trips
    import pandas as pd

    # Build full labeled candidate frame for all training trips (cached).
    train_ids = available_trips("train")
    cache = CandidateCache()
    train_dirs = [ROOT / "data" / "raw" / "Student folder" / "train" / t
                  for t in train_ids]
    ds_identity = dataset_identity_from_trips(train_dirs)
    cache_key = cache.cache_key(ds_identity, config["candidates"],
                                "phase2a-frozen")
    if not cache.exists(cache_key):
        print(f"[pathsnap.optimize] building candidate cache for {len(train_ids)} "
              f"training trips", flush=True)
        frames = []
        for trip_id in train_ids:
            trip = load_inputs(ROOT / "data" / "raw" / "Student folder" / "train" / trip_id)
            frames.append(generate(trip, config["candidates"], trip_id))
        cache.write(cache_key, pd.concat(frames, ignore_index=True))
    full_frame = cache.read(cache_key)
    # Attach labels to every observation (full frame, not sampled).
    labeled_frames = []
    for trip_id in train_ids:
        gen = full_frame[full_frame.trip_id == trip_id].copy()
        labels = build_label_table("train", trip_id)
        labeled_frames.append(join_labels_with_candidates(gen, labels))
    full_labeled = pd.concat(labeled_frames, ignore_index=True) if labeled_frames \
        else pd.DataFrame()
    full_labeled = full_labeled[full_labeled.weak_target >= 0].copy()

    folds = make_folds(train_ids, n_splits=n_folds, seed=seed)
    feature_schema = list(config.get("feature_schema", []))

    db_path = study_db_path(model_name, strategy)
    storage = f"sqlite:///{db_path}"
    sampler = TPESampler(seed=seed, n_startup_trials=min(10, n_trials))
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=study_name(model_name, strategy), storage=storage,
        sampler=sampler, pruner=pruner, direction="maximize",
        load_if_exists=resume)

    already = len(study.trials)
    remaining = max(0, n_trials - already)
    print(f"[pathsnap.optimize] {model_name}/{strategy}: study "
          f"'{study_name(model_name, strategy)}' DB={db_path}", flush=True)
    print(f"[pathsnap.optimize] existing trials={already}, "
          f"running {remaining} more (target {n_trials})", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = _build_search_space(model_name, trial)
        fold_results = []
        for fold in folds:
            train_trips = fold.train_trips
            val_trips = fold.validation_trips
            train_df = full_labeled[full_labeled.trip_id.isin(train_trips)]
            val_df = full_labeled[full_labeled.trip_id.isin(val_trips)]
            # Sample training observations per strategy (whole observation groups).
            train_sample, _ = sample_observations(train_df, train_trips, strategy,
                                                  config)
            fr = run_cv_fold(model_name, params, train_sample, val_df,
                             feature_schema, strategy,
                             consensus_only=(strategy == "consensus"),
                             seed=seed)
            fr.fold = fold.fold
            fold_results.append(fr)
            trial.report(fr.top1_agreement, step=fold.fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_top1 = float(np.mean([fr.top1_agreement for fr in fold_results]))
        std_top1 = float(np.std([fr.top1_agreement for fr in fold_results]))
        mean_logloss = float(np.mean([fr.candidate_log_loss for fr in fold_results]))
        # Per-trip stability: std of per-trip Top-1 across all val trips.
        all_trip_top1 = []
        for fr in fold_results:
            all_trip_top1.extend(fr.per_trip_top1.values())
        trip_stability = float(np.std(all_trip_top1)) if all_trip_top1 else 0.0

        trial.set_user_attr("model", model_name)
        trial.set_user_attr("strategy", strategy)
        trial.set_user_attr("hyperparameters", params)
        trial.set_user_attr("cv_results", [fr.__dict__ for fr in fold_results])
        trial.set_user_attr("mean_top1", mean_top1)
        trial.set_user_attr("std_top1", std_top1)
        trial.set_user_attr("mean_logloss", mean_logloss)
        trial.set_user_attr("trip_stability", trip_stability)
        trial.set_user_attr("seed", seed)
        trial.set_user_attr("n_folds", n_folds)
        trial.set_user_attr("data_version", cache_key)
        trial.set_user_attr("feature_schema", feature_schema)
        trial.set_user_attr("environment", environment_snapshot())
        return mean_top1

    t0 = time.time()
    study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    wall = time.time() - t0

    best = study.best_trial
    summary = dict(
        model=model_name, strategy=strategy, study_name=study_name(model_name, strategy),
        db_path=str(db_path), n_trials_completed=len(study.trials),
        n_trials_run_this_session=remaining, seed=seed, n_folds=n_folds,
        wall_seconds=wall, direction="maximize",
        best_trial=best.number, best_value=best.value,
        best_hyperparameters=best.params,
        best_user_attrs={k: v for k, v in best.user_attrs.items()
                         if k not in ("cv_results", "environment")},
        pruned_trials=sum(1 for t in study.trials
                          if t.state == optuna.trial.TrialState.PRUNED),
        failed_trials=sum(1 for t in study.trials
                          if t.state == optuna.trial.TrialState.FAIL),
    )
    (OPTUNA_ROOT / f"{study_name(model_name, strategy)}_summary.json").write_text(
        json.dumps(summary, indent=2))
    # Save best hyperparameters separately (never overwrites a checkpoint).
    (OPTUNA_ROOT / f"{study_name(model_name, strategy)}_best_params.json").write_text(
        json.dumps(dict(model=model_name, strategy=strategy,
                        hyperparameters=best.params,
                        mean_top1=best.value, trial=best.number,
                        seed=seed), indent=2))
    print(f"[pathsnap.optimize] {model_name}: best trial {best.number} "
          f"Top-1 {best.value:.4f} params={best.params}", flush=True)
    return summary


if __name__ == "__main__":
    import sys
    from pathsnap.cli.optimize import main
    sys.exit(main(sys.argv[1:]))
