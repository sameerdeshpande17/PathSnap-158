"""Training orchestrator: run management, checkpointing, skip-existing, force, resume."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import importlib.metadata
import numpy as np
import pandas as pd

from pathsnap.features import (CandidateCache, DEFAULT_CANDIDATE_CONFIG,
                               generate, load_inputs)
from pathsnap.explore import ROOT
from pathsnap.labels import (
    STRATEGIES, build_label_table, join_labels_with_candidates,
    sample_observations,
)
from pathsnap.models import ALL_MODELS, MODEL_REGISTRY
from pathsnap.splits import make_folds, training_trips

ARTIFACTS_ROOT = ROOT / "artifacts" / "models"
CONFIG_PATH = ROOT / "experiments" / "phase2b" / "config.json"
DATA_ROOT = ROOT / "data"
RAW_ROOT = ROOT / "data" / "raw" / "Student folder"

DEFAULT_STRATEGY = "consensus"
DEFAULT_FOLDS = 3
DEFAULT_SEED = 42


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for d in importlib.metadata.distributions():
        try:
            packages[d.metadata["Name"]] = d.version
        except Exception:
            pass
    import torch
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        gpu=torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None,
        cpu_count=int(__import__("os").cpu_count()),
        packages=dict(sorted(packages.items())),
    )


@dataclass
class RunInfo:
    model_name: str
    strategy: str
    run_id: str
    run_dir: Path
    config: dict
    started_at: float


def make_run_id(model_name: str, strategy: str, config: dict) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    payload = json.dumps({
        "model": model_name, "strategy": strategy,
        "config": config, "seed": DEFAULT_SEED,
    }, sort_keys=True)
    short = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{model_name}_{strategy}_{stamp}_{short}"


def load_run_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return dict(
        seed=DEFAULT_SEED,
        folds=DEFAULT_FOLDS,
        cpu_threads=4,
        candidates=DEFAULT_CANDIDATE_CONFIG,
        consensus_observation_budget=20000,
        B_extra_observation_budget=5000,
        models={m: {} for m in ALL_MODELS},
    )


def model_dir(model_name: str) -> Path:
    return ARTIFACTS_ROOT / model_name


def run_dir(model_name: str, run_id: str) -> Path:
    return model_dir(model_name) / run_id


def best_pointer_path(model_name: str) -> Path:
    return model_dir(model_name) / "best.json"


def list_runs(model_name: str) -> list[dict]:
    out = []
    mdir = model_dir(model_name)
    if not mdir.exists():
        return out
    for run_path in sorted(mdir.iterdir()):
        if not run_path.is_dir():
            continue
        meta = {}
        mp = run_path / "metadata.json"
        jp = run_path / "metrics.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        elif jp.exists():
            meta = json.loads(jp.read_text())
        best = None
        bp = best_pointer_path(model_name)
        if bp.exists():
            best = json.loads(bp.read_text())
        out.append(dict(
            run_id=run_path.name,
            run_dir=str(run_path),
            best=best.get("run_id") == run_path.name if best else None,
            created=meta.get("metrics", {}).get("train_seconds"),
            device=meta.get("device"),
        ))
    return out


def compatible_checkpoint(run_path: Path, model_name: str,
                          config: dict, strategy: str) -> bool:
    """Check whether an existing run can satisfy --skip-existing.

    Compatible if metadata matches model, strategy, feature schema, and
    hyperparameters, and the run completed successfully (metrics.json present).
    """
    meta_path = run_path / "metadata.json"
    metrics_path = run_path / "metrics.json"
    if not (meta_path.exists() and metrics_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
        metrics = json.loads(metrics_path.read_text())
    except Exception:
        return False
    if meta.get("model_name") != model_name:
        return False
    if meta.get("params") != config.get("models", {}).get(model_name, {}):
        return False
    if meta.get("feature_names") != config.get("feature_schema", []):
        return False
    if metrics.get("strategy") != strategy:
        return False
    return True


def save_run_artifacts(run_info: RunInfo, model_wrapper, metrics: dict,
                       data_manifest: dict, extra: dict | None = None) -> None:
    run_info.run_dir.mkdir(parents=True, exist_ok=True)
    # Model save
    model_info = model_wrapper.save(run_info.run_dir)
    # metrics.json
    metrics_record = dict(metrics)
    metrics_record.update(dict(
        model=run_info.model_name, strategy=run_info.strategy,
        run_id=run_info.run_id, run_dir=str(run_info.run_dir),
        completed_at=time.time(),
    ))
    if extra:
        metrics_record.update(extra)
    (run_info.run_dir / "metrics.json").write_text(
        json.dumps(metrics_record, indent=2))
    # feature_schema.json
    (run_info.run_dir / "feature_schema.json").write_text(json.dumps(
        dict(features=run_info.config.get("feature_schema", []),
             schema_version="pathsnap-candidates-v1",
             n_features=len(run_info.config.get("feature_schema", []))),
        indent=2))
    # data_manifest.json
    (run_info.run_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2))
    # environment.json
    (run_info.run_dir / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2))
    # config.json (effective run config)
    (run_info.run_dir / "config.json").write_text(
        json.dumps(run_info.config, indent=2))
    # training_log.jsonl (from model metrics if present)
    log_path = run_info.run_dir / "training_log.jsonl"
    log_entries = metrics_record.get("training_log", [])
    with log_path.open("w") as fh:
        for entry in log_entries:
            fh.write(json.dumps(entry) + "\n")


def mark_best(model_name: str, run_id: str, metric: str, value: float,
              higher_is_better: bool) -> None:
    mdir = model_dir(model_name)
    mdir.mkdir(parents=True, exist_ok=True)
    pointer = dict(model=model_name, best_run_id=run_id, metric=metric,
                   value=value, higher_is_better=higher_is_better,
                   updated_at=time.time())
    best_pointer_path(model_name).write_text(json.dumps(pointer, indent=2))


def load_best(model_name: str) -> dict | None:
    bp = best_pointer_path(model_name)
    if not bp.exists():
        return None
    return json.loads(bp.read_text())


def load_model_from_run(model_name: str, run_id: str | None = None):
    """Load a model wrapper from a run directory (or best run if run_id None)."""
    if run_id is None:
        best = load_best(model_name)
        if best is None:
            raise FileNotFoundError(f"No best checkpoint for {model_name}")
        run_id = best["best_run_id"]
    rd = run_dir(model_name, run_id)
    wrapper_cls = MODEL_REGISTRY[model_name]
    return wrapper_cls.load(rd), run_id, rd


def build_training_frame(split: str, trip_ids: list[str], strategy: str,
                         config: dict, cache) -> pd.DataFrame:
    """Build sampled training candidate frame for given trips and strategy."""
    frames = []
    for trip_id in trip_ids:
        trip_dir = RAW_ROOT / split / trip_id
        trip = load_inputs(trip_dir)
        gen = generate(trip, config["candidates"], trip_id)
        labels = build_label_table(split, trip_id)
        joined = join_labels_with_candidates(gen, labels)
        frames.append(joined)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    train_frame, _ = sample_observations(frame, trip_ids, strategy, config)
    return train_frame


def build_validation_frame(split: str, trip_ids: list[str], config: dict) -> pd.DataFrame:
    """Build full candidate frame for validation trips (all covered observations)."""
    frames = []
    for trip_id in trip_ids:
        trip_dir = RAW_ROOT / split / trip_id
        trip = load_inputs(trip_dir)
        gen = generate(trip, config["candidates"], trip_id)
        labels = build_label_table(split, trip_id)
        joined = join_labels_with_candidates(gen, labels)
        frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass
class TrainResult:
    model_name: str
    strategy: str
    run_id: str
    run_dir: Path
    metrics: dict
    train_seconds: float
    device: str
    skipped: bool = False


def train_model(model_name: str, strategy: str = DEFAULT_STRATEGY,
                cv: bool = False, skip_existing: bool = False,
                force: bool = False, resume: str | None = None,
                config: dict | None = None) -> TrainResult | list[TrainResult]:
    """Train one model.

    If cv=True: run trip-grouped CV and save fold models separately, then
    report fold metrics without selecting a final model.
    If cv=False: final training on the full permitted training partition.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name}. Available: {ALL_MODELS}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}. Available: {STRATEGIES}")
    if skip_existing and force:
        raise ValueError("Use either --skip-existing or --force, not both")
    config = config or load_run_config()
    model_cfg = config.get("models", {}).get(model_name, {})
    print(f"[pathsnap.train] model={model_name} strategy={strategy} "
          f"cv={cv} device_query", flush=True)

    if cv:
        return _train_cv(model_name, strategy, config, skip_existing, force)
    return _train_final(model_name, strategy, model_cfg, config,
                        skip_existing, force, resume)


def _train_final(model_name, strategy, model_cfg, config,
                 skip_existing, force, resume):
    train_ids = training_trips("train")
    from pathsnap.features import CandidateCache, dataset_identity_from_trips
    cache = CandidateCache()
    train_dirs = [RAW_ROOT / "train" / t for t in train_ids]
    ds_identity = dataset_identity_from_trips(train_dirs)
    key = cache.cache_key(ds_identity, config["candidates"],
                          "phase2a-frozen")
    if not cache.exists(key):
        print(f"[pathsnap.train] building candidate cache ({len(train_ids)} trips)",
              flush=True)
        frames = []
        for trip_id in train_ids:
            trip = load_inputs(RAW_ROOT / "train" / trip_id)
            frames.append(generate(trip, config["candidates"], trip_id))
        cache.write(key, pd.concat(frames, ignore_index=True))
    print(f"[pathsnap.train] cache key {key}", flush=True)

    # Check existing compatible checkpoint
    if skip_existing:
        for run in list_runs(model_name):
            rd = Path(run["run_dir"])
            if compatible_checkpoint(rd, model_name, config, strategy):
                print(f"[pathsnap.train] skip-existing: compatible run "
                      f"{run['run_id']} found", flush=True)
                return TrainResult(model_name, strategy, run["run_id"], rd,
                                   json.loads((rd / "metrics.json").read_text()),
                                   0.0, run.get("device", "CPU"), skipped=True)

    run_id = make_run_id(model_name, strategy, config)
    rd = run_dir(model_name, run_id)
    if rd.exists() and not force:
        raise FileExistsError(
            f"Run directory {rd} exists. Use --force to create a new run.")
    print(f"[pathsnap.train] final training {model_name} on {len(train_ids)} "
          f"trips, strategy={strategy}, run_id={run_id}", flush=True)

    train_frame = build_training_frame("train", train_ids, strategy, config, cache)
    X = np.ascontiguousarray(train_frame[config["feature_schema"]]
                             .to_numpy(dtype=np.float32))
    y = train_frame.weak_target.to_numpy(dtype=np.float64)
    covered = y >= 0
    X = X[covered]
    y = y[covered]
    print(f"[pathsnap.train] fitted rows: {len(X):,} "
          f"(positives {(y == 1).sum():,}, negatives {(y == 0).sum():,})",
          flush=True)

    wrapper_cls = MODEL_REGISTRY[model_name]
    wrapper = wrapper_cls(name=model_name, params=model_cfg,
                          n_features=X.shape[1],
                          feature_names=list(config["feature_schema"]),
                          random_state=int(config.get("seed", DEFAULT_SEED)))
    t0 = time.perf_counter()
    metrics = wrapper.fit(X, y)
    train_s = time.perf_counter() - t0

    data_manifest = dict(
        split="train", trips=train_ids,
        n_observations=int(train_frame.observation_index.nunique()),
        n_candidate_rows=int(len(train_frame)),
        n_fitted_rows=int(len(X)),
        n_positives=int((y == 1).sum()),
        n_negatives=int((y == 0).sum()),
        cache_key=key,
    )
    save_run_artifacts(RunInfo(model_name, strategy, run_id, rd, config,
                               time.time()),
                       wrapper, {**metrics, "train_seconds": train_s,
                                 "strategy": strategy},
                       data_manifest)
    print(f"[pathsnap.train] saved artifacts to {rd}", flush=True)
    mark_best(model_name, run_id, "train_rows", float(len(X)),
              higher_is_better=True)
    return TrainResult(model_name, strategy, run_id, rd,
                       {**metrics, "train_seconds": train_s},
                       train_s, wrapper.device)


def _train_cv(model_name, strategy, config, skip_existing, force):
    train_ids = training_trips("train")
    folds = make_folds(train_ids, n_splits=int(config.get("folds", DEFAULT_FOLDS)),
                       seed=int(config.get("seed", DEFAULT_SEED)),
                       save_path=ROOT / "experiments" / "phase2b" / "folds.json")
    print(f"[pathsnap.train] CV: {len(folds)} folds, seed {config.get('seed')}",
          flush=True)
    results = []
    from pathsnap.features import CandidateCache
    cache = CandidateCache()
    for fold in folds:
        print(f"[pathsnap.train] fold {fold.fold}: "
              f"{len(fold.train_trips)} train / {len(fold.validation_trips)} val",
              flush=True)
        train_frame = build_training_frame("train", fold.train_trips, strategy,
                                           config, cache)
        val_frame = build_validation_frame("train", fold.validation_trips,
                                           config)
        X = np.ascontiguousarray(train_frame[config["feature_schema"]]
                                 .to_numpy(dtype=np.float32))
        y = train_frame.weak_target.to_numpy(dtype=np.float64)
        covered = y >= 0
        X, y = X[covered], y[covered]
        wrapper_cls = MODEL_REGISTRY[model_name]
        wrapper = wrapper_cls(name=model_name, params=config.get("models", {}).get(model_name, {}),
                              n_features=X.shape[1],
                              feature_names=list(config["feature_schema"]),
                              random_state=int(config.get("seed", DEFAULT_SEED)))
        m = wrapper.fit(X, y)
        Xv = np.ascontiguousarray(val_frame[config["feature_schema"]]
                                  .to_numpy(dtype=np.float32))
        scores = wrapper.predict_proba(Xv)
        # Observation-level top-1
        val_frame = val_frame.copy()
        val_frame["candidate_score"] = scores
        from pathsnap.phase2b_metrics import evaluate_frame, top1_arc
        obs, cand = evaluate_frame(val_frame)
        fold_metrics = dict(fold=fold.fold, model=model_name, strategy=strategy,
                            val_observations=obs["observation_count"],
                            top1_agreement=obs["top1_agreement"],
                            candidate_macro_f1=cand["candidate_macro_f1"],
                            candidate_log_loss=cand["candidate_log_loss"],
                            **m)
        results.append(fold_metrics)
        print(f"[pathsnap.train] fold {fold.fold} Top-1 {obs['top1_agreement']:.4f} "
              f"MacroF1 {cand['candidate_macro_f1']:.4f}", flush=True)
    summary = dict(model=model_name, strategy=strategy, folds=results,
                   mean_top1=float(np.mean([r["top1_agreement"] for r in results])),
                   mean_macro_f1=float(np.mean([r["candidate_macro_f1"] for r in results])))
    out = ROOT / "experiments" / "phase2b" / f"cv_{model_name}_{strategy}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[pathsnap.train] CV summary saved to {out}", flush=True)
    return summary
