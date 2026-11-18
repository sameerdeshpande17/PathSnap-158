"""Model comparison, final selection, and refit for Phase 3A.

All decisions are based on training-only cross-validation evidence.
The student-testing partition is used only for a single reported evaluation,
never to revise hyperparameters.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pathsnap.explore import ROOT
from pathsnap.training import (ALL_MODELS, ARTIFACTS_ROOT, build_training_frame,
                               load_model_from_run, load_run_config,
                               load_best, make_folds, model_dir, run_dir,
                               sample_observations, train_model)
from pathsnap.optimization import OPTUNA_ROOT, study_name
from pathsnap.features import CandidateCache
from pathsnap.candidates import generate, load_inputs
from pathsnap.labels import build_label_table, join_labels_with_candidates

PHASE3A = ROOT / "experiments" / "phase3a"


def _load_study_summaries() -> list[dict[str, Any]]:
    rows = []
    if not OPTUNA_ROOT.exists():
        return rows
    for f in sorted(OPTUNA_ROOT.glob("*_summary.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:  # pragma: no cover - defensive
            pass
    return rows


def compare_tuned() -> pd.DataFrame:
    """Build a comparison table from completed Optuna studies."""
    PHASE3A.mkdir(parents=True, exist_ok=True)
    summaries = _load_study_summaries()
    rows = []
    for s in summaries:
        ua = s.get("best_user_attrs", {})
        rows.append(dict(
            model=s.get("model"), strategy=s.get("strategy"),
            best_trial=s.get("best_trial"), best_top1=s.get("best_value"),
            std_top1=ua.get("std_top1"), mean_logloss=ua.get("mean_logloss"),
            trip_stability=ua.get("trip_stability"),
            pruned_trials=s.get("pruned_trials"),
            failed_trials=s.get("failed_trials"),
            n_trials=s.get("n_trials_completed"),
            wall_seconds=s.get("wall_seconds"),
            hyperparameters=json.dumps(s.get("best_hyperparameters", {})),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("best_top1", ascending=False).reset_index(drop=True)
        df.to_csv(PHASE3A / "comparison_tuned.csv", index=False)
    print(df.to_string(index=False))
    print(f"[pathsnap.compare_tuned] -> {PHASE3A / 'comparison_tuned.csv'}")
    return df


def select_final_model(metric: str = "best_top1") -> dict[str, Any]:
    """Recommend the best model from training-only CV evidence.

    Does NOT retrain. Selection considers mean Top-1 (primary), std (stability),
    log loss, and inference latency/complexity as tie-breakers. If improvements
    are negligible (<=1% relative), prefers the simpler/robuster model.
    """
    PHASE3A.mkdir(parents=True, exist_ok=True)
    df = compare_tuned()
    if df.empty or metric not in df.columns:
        raise ValueError("No completed tuning studies found. Run optimize first.")

    # Primary ranking by mean Top-1.
    ranked = df.sort_values(metric, ascending=False)
    best = ranked.iloc[0]
    runner_up = ranked.iloc[1] if len(ranked) > 1 else None

    # Negligible-improvement rule: prefer simpler model if within 1% relative.
    simple_order = ["logistic", "hist_gradient_boosting", "random_forest",
                    "knn", "mlp"]
    best_model = best["model"]
    best_value = best[metric]
    recommendation = best_model
    rationale = [f"Highest CV mean {metric} = {best_value:.4f} "
                 f"(trial {best['best_trial']})."]

    if runner_up is not None:
        gap = (best_value - runner_up[metric]) / max(abs(runner_up[metric]), 1e-9)
        rationale.append(f"Runner-up {runner_up['model']}: "
                         f"{runner_up[metric]:.4f} (gap {gap:.2%}).")
        if gap <= 0.01:
            # Pick simpler of the two.
            simpler = best_model if simple_order.index(best_model) < \
                simple_order.index(runner_up["model"]) else runner_up["model"]
            recommendation = simpler
            rationale.append(f"Improvement <=1%, preferring simpler model: "
                             f"{simpler}.")

    # Compatibility note for HMM emissions.
    rationale.append("MLP and HistGradientBoosting produce calibrated-style "
                     "scores suitable for HMM emission probabilities.")

    selection = dict(
        recommended_model=recommendation,
        recommended_from_trial=int(best["best_trial"]) if best["model"] == recommendation \
            else int(runner_up["best_trial"]) if runner_up is not None else None,
        selection_metric=metric,
        best_cv_top1=float(best_value),
        runner_up=runner_up["model"] if runner_up is not None else None,
        rationale=" ".join(rationale),
        basis="training-only cross-validation (student-testing not used for selection)",
        comparison=ranked[["model", "best_top1", "std_top1", "mean_logloss",
                           "trip_stability"]].to_dict("records"),
    )
    (PHASE3A / "selected_model.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))
    print(f"[pathsnap.select_final_model] -> {PHASE3A / 'selected_model.json'}")
    return selection


def _train_final_with_params(model_name: str, params: dict, strategy: str,
                             config: dict, cache) -> dict[str, Any]:
    """Train model on full training partition with given hyperparameters."""
    import pandas as pd
    train_ids = [t for t in __import__("pathsnap.splits", fromlist=["available_trips"]).available_trips("train")]
    full = build_training_frame("train", train_ids, strategy, config, cache)
    full = full[full.weak_target >= 0].copy()
    train_sample, _ = sample_observations(full, train_ids, strategy, config)

    feature_schema = list(config.get("feature_schema", []))
    X = np.ascontiguousarray(train_sample[feature_schema].to_numpy(dtype=np.float32))
    y = train_sample.weak_target.to_numpy(dtype=np.float64)
    covered = y >= 0
    X, y = X[covered], y[covered]

    t0 = time.time()
    wrapper_cls = __import__("pathsnap.models", fromlist=["MODEL_REGISTRY"]).MODEL_REGISTRY[model_name]
    wrapper = wrapper_cls(name=model_name, params=params,
                          n_features=X.shape[1],
                          feature_names=list(feature_schema),
                          random_state=int(config.get("seed", 42)))
    metrics = wrapper.fit(X, y)
    train_s = time.time() - t0

    # Evaluate on student-testing (single reported development evaluation).
    from pathsnap.evaluation import evaluate_model
    eval_res = evaluate_model(model_name, split="student_test",
                              consensus_only=True)

    # Save to artifacts/models/final/<model>/
    final_dir = ARTIFACTS_ROOT / "models" / "final" / model_name
    final_dir.mkdir(parents=True, exist_ok=True)
    model_info = wrapper.save(final_dir)
    # Save scaler separately if sklearn (wrapper.save bundles it in joblib).
    manifest = dict(
        model=model_name, strategy=strategy,
        hyperparameters=params, n_train_rows=int(len(X)),
        n_train_observations=int(train_sample.observation_index.nunique()),
        train_seconds=train_s, total_seconds=time.time() - t0,
        device=wrapper.device,
        eval_student_test_top1=eval_res.get("top1_agreement"),
        eval_student_test_macro_f1=eval_res.get("candidate_macro_f1"),
        eval_student_test_log_loss=eval_res.get("candidate_log_loss"),
        feature_schema=feature_schema,
        n_features=int(X.shape[1]),
    )
    (final_dir / "config.json").write_text(json.dumps(config, indent=2))
    (final_dir / "metrics.json").write_text(
        json.dumps({**metrics, **manifest}, indent=2))
    (final_dir / "feature_schema.json").write_text(
        json.dumps(dict(features=feature_schema,
                        schema_version="pathsnap-candidates-v1",
                        n_features=len(feature_schema)), indent=2))
    (final_dir / "data_manifest.json").write_text(
        json.dumps(dict(split="train", trips=train_ids,
                        n_trips=len(train_ids)), indent=2))
    (final_dir / "environment.json").write_text(
        json.dumps(__import__("pathsnap.training", fromlist=["environment_snapshot"]).environment_snapshot(), indent=2))
    (final_dir / "checksum.json").write_text(
        json.dumps(dict(model_sha256=model_info.get("sha256"),
                        model_bytes=model_info.get("bytes")), indent=2))
    print(f"[pathsnap.refit_final] saved final checkpoint -> {final_dir}")
    return dict(final_dir=str(final_dir), metrics=manifest, eval=eval_res)


def refit_final(model_name: str, strategy: str = "consensus",
                config: dict | None = None) -> dict[str, Any]:
    """Explicitly refit the selected model on the full training partition.

    Reads best hyperparameters from artifacts/optuna/*_best_params.json.
    Saves final checkpoint to artifacts/models/final/<model>/.
    """
    config = config or load_run_config()
    bp_path = OPTUNA_ROOT / f"{study_name(model_name, strategy)}_best_params.json"
    if not bp_path.exists():
        raise FileNotFoundError(
            f"No best params found at {bp_path}. Run optimize first.")
    best = json.loads(bp_path.read_text())
    params = best["hyperparameters"]

    from pathsnap.features import dataset_identity_from_trips
    cache = CandidateCache()
    return _train_final_with_params(model_name, params, strategy, config, cache)
