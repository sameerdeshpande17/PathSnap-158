"""Five-model candidate-ranking experiments with trip-grouped cross-validation.

Reference-route files are never read here; targets join offline weak labels.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .candidates import FEATURES, load_inputs
from .explore import ROOT, save_json
from .phase2b_data import (
    ARTIFACTS, DATA, MODELS, STRATEGIES, OUT, RAW, build, initialize,
    load_partition, sample_observations,
)
from .phase2b_metrics import (
    FEATURE_DISCLAIMER, candidate_metrics, evaluate_frame, metrics_summary,
    route_continuity,
)

CPU_THREADS = 4
torch.set_num_threads(CPU_THREADS)
np.random.seed(42)
torch.manual_seed(42)

DEVICE_CUDA = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class TrainResult:
    model_name: str
    device: str
    train_seconds: float
    predict_seconds_per_obs: float
    peak_rss_mb: float
    artifact_path: str
    artifact_bytes: int


class TorchMLP(nn.Module):
    def __init__(self, n_features: int, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = n_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(0.1)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def peak_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def fit_sklearn(model_name: str, params: dict, X: np.ndarray, y: np.ndarray):
    start = time.perf_counter()
    if model_name == "logistic":
        est = LogisticRegression(
            C=params.get("C", 1.0), max_iter=params.get("max_iter", 500),
            solver="lbfgs", class_weight=None, random_state=42)
    elif model_name == "knn":
        est = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 11),
            weights=params.get("weights", "distance"),
            algorithm=params.get("algorithm", "ball_tree"),
            leaf_size=params.get("leaf_size", 40),
            n_jobs=params.get("n_jobs", CPU_THREADS))
    elif model_name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=params.get("n_jobs", CPU_THREADS), random_state=42)
    elif model_name == "hist_gradient_boosting":
        est = HistGradientBoostingClassifier(
            max_iter=params.get("max_iter", 150), learning_rate=params.get("learning_rate", .08),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            l2_regularization=params.get("l2_regularization", 1.0),
            early_stopping=params.get("early_stopping", False), random_state=42)
    else:
        raise ValueError(model_name)
    est.fit(X, y)
    return est, time.perf_counter() - start


def fit_mlp(params: dict, X: np.ndarray, y: np.ndarray):
    """Train on CUDA when available; otherwise CPU. Returns module and seconds."""
    device = DEVICE_CUDA
    start = time.perf_counter()
    model = TorchMLP(X.shape[1], tuple(params.get("hidden", [64, 32]))).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("learning_rate", .001),
                                 weight_decay=params.get("weight_decay", .0001))
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    n = len(X_t)
    batch = int(params.get("batch_size", 4096))
    epochs = int(params.get("epochs", 15))
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        running = 0.
        for start_i in range(0, n, batch):
            idx = perm[start_i:start_i + batch]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss) * len(idx)
        if epoch % 5 == 0:
            print(f"  MLP epoch {epoch + 1}/{epochs} loss {running / n:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X_t[i:i + batch]) for i in range(0, n, batch)])
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return model, probs, time.perf_counter() - start


def predict_sklearn(est, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def predict_mlp(model, X: np.ndarray, device, batch=8192) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.cat([
            torch.sigmoid(model(X_t[i:i + batch])) for i in range(0, len(X_t), batch)
        ])
    return probs.cpu().numpy()


def observation_level(df: pd.DataFrame):
    """One score per observation: max candidate score; predictions grouped stably."""
    df = df.sort_values(["candidate_score", "candidate_rank"],
                        ascending=[False, True], kind="mergesort")
    top = df.groupby(["trip_id", "observation_index"], sort=True).head(1)
    return top.set_index(["trip_id", "observation_index"])[["arc_id", "candidate_score"]]


def run_fold(fold, train_keys, val_frame, strategy_cfg, config):
    """Train on sampled train keys; evaluate on full val observation candidate rows."""
    scaler = StandardScaler()
    results = {}
    fold_art = ARTIFACTS / f"fold_{fold['fold']}"
    fold_art.mkdir(parents=True, exist_ok=True)
    train_frame = train_keys.copy()
    X_tr = np.ascontiguousarray(train_frame[FEATURES].to_numpy(dtype=np.float32))
    y_tr = train_frame.weak_target.to_numpy(dtype=np.float64)
    scaler.fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    val_groups = [g for _, g in val_frame.groupby(["trip_id", "observation_index"],
                                                  sort=True)]
    arcs_lookup = {}
    for directory in sorted((RAW / "train").iterdir()):
        if directory.is_dir():
            try:
                from .data import load_trip
                arcs_lookup[directory.name] = load_trip(directory).arcs
            except Exception:  # pragma: no cover - diagnostics only
                pass
    for name in MODELS:
        params = config["models"][name]
        mem_before = peak_rss_mb()
        t0 = time.perf_counter()
        if name == "mlp":
            model, probs_tr, train_s = fit_mlp(params, X_tr_s, y_tr)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # validation prediction in observation-group chunks
            val_scores, val_arcs = [], []
            for group in val_groups:
                X_v = np.ascontiguousarray(group[FEATURES].to_numpy(dtype=np.float32))
                p = predict_mlp(model, X_v, device)
                best = int(np.argmax(p))
                val_scores.append(float(p[best]))
                val_arcs.append(int(group.arc_id.iloc[best]))
            pred_map = {(g.trip_id.iloc[0], g.observation_index.iloc[0]): (s, a)
                        for g, s, a in zip(val_groups, val_scores, val_arcs)}
            train_result = TrainResult(
                name, device, train_s,
                time.perf_counter() - t0 - train_s, peak_rss_mb() - mem_before,
                "", 0)
        else:
            est, train_s = fit_sklearn(name, params, X_tr_s, y_tr)
            device = "CPU"
            val_scores, val_arcs = [], []
            for group in val_groups:
                X_v = np.ascontiguousarray(group[FEATURES].to_numpy(dtype=np.float32))
                p = predict_sklearn(est, X_v)
                best = int(np.argmax(p))
                val_scores.append(float(p[best]))
                val_arcs.append(int(group.arc_id.iloc[best]))
            pred_map = {(g.trip_id.iloc[0], g.observation_index.iloc[0]): (s, a)
                        for g, s, a in zip(val_groups, val_scores, val_arcs)}
            path = fold_art / f"{name}.joblib"
            joblib.dump({"estimator": est, "scaler": scaler,
                         "features": FEATURES, "schema_version": "pathsnap-candidates-v1"},
                        path, compress=3)
            train_result = TrainResult(name, device, train_s, 0.,
                                       peak_rss_mb() - mem_before, str(path),
                                       path.stat().st_size)
        # Build val frame with scores aligned to row order.
        val_frame = val_frame.copy()
        keys = list(zip(val_frame.trip_id, val_frame.observation_index))
        try:
            val_frame["candidate_score"] = [pred_map[k][0] for k in keys]
            val_frame["top1_arc"] = [pred_map[k][1] for k in keys]
        except KeyError as exc:
            raise KeyError(f"Missing prediction for observation key {exc}") from exc
        obs, cand = evaluate_frame(val_frame)
        cont = route_continuity(val_frame, arcs_lookup)
        results[name] = dict(
            train=train_result, obs=metrics_summary(obs, cand, prefix="obs_"),
            cand=metrics_summary(obs, cand, prefix="cand_"),
            cont=cont.top1_direct_adjacency.mean() if len(cont) else np.nan,
            val_observations=obs["observation_count"],
        )
        print(f"  fold {fold['fold']} {name}: Top-1 {obs['top1_agreement']:.4f} "
              f"MacroF1 {cand['candidate_macro_f1']:.4f} "
              f"train {train_s:.2f}s", flush=True)
    return results


def main():
    initialize()
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text())
    folds = json.loads((OUT / "folds.json").read_text())
    train_partition = load_partition("train")
    all_results = []
    for fold in folds:
        train_trips = fold["train_trips"]
        val_trips = fold["validation_trips"]
        print(f"FOLD {fold['fold']}: train {len(train_trips)} trips, "
              f"val {len(val_trips)} trips", flush=True)
        val_base = train_partition[train_partition.trip_id.isin(val_trips)]
        for strategy in STRATEGIES:
            train_keys, _ = sample_observations(
                train_partition, train_trips, strategy, config)
            # Same consensus-evaluation stratum for both strategies; training data differs.
            val_frame = val_base[val_base.consensus_eligible].copy()
            print(f"  strategy {strategy}: train rows {len(train_keys):,}, "
                  f"val observations {val_frame.observation_index.nunique()}",
                  flush=True)
            fold_results = run_fold(fold, train_keys, val_frame, strategy, config)
            for name, res in fold_results.items():
                row = dict(fold=fold["fold"], strategy=strategy, model=name,
                           device=res["train"].device,
                           train_s=res["train"].train_seconds,
                           predict_s=res["train"].predict_seconds_per_obs,
                           peak_rss_mb=res["train"].peak_rss_mb,
                           artifact_bytes=res["train"].artifact_bytes,
                           val_observations=res["val_observations"],
                           cont_direct_adjacency=res["cont"])
                row.update(res["obs"]); row.update(res["cand"])
                all_results.append(row)
    cv = pd.DataFrame(all_results)
    cv.to_csv(ROOT / "experiments/model_comparison.csv", index=False)
    summary = cv.groupby("model")[["obs_top1_agreement", "cand_candidate_macro_f1",
                                   "cand_candidate_log_loss", "cand_candidate_accuracy",
                                   "obs_mrr", "obs_candidate_miss_rate",
                                   "cont_direct_adjacency", "train_s"]].agg(
        ["mean", "std"]).reset_index()
    save_json(OUT / "cv_summary.json", summary.to_dict("records"))
    print(summary.to_string(), flush=True)
    # Selection: primary = CV trip-macro observation Top-1 (mean over folds).
    best = cv.groupby("model").obs_top1_agreement.mean().idxmax()
    save_json(OUT / "selection.json", {
        "selected_model": best,
        "selection_metric": "CV mean observation Top-1 weak-label agreement",
        "selection_timestamp": time.time(),
        "config_sha256": __import__("hashlib").sha256(
            config_path.read_bytes()).hexdigest(),
        "models": {m: float(cv[cv.model == m].obs_top1_agreement.mean()) for m in MODELS},
    })
    print(f"SELECTED: {best}", flush=True)


if __name__ == "__main__":
    main()
