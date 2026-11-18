"""Final model training on all 80 trips and student-testing evaluation.

Runs only after CV selection (selection.json) is frozen.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .candidates import FEATURES
from .explore import ROOT, save_json
from .phase2b_data import (
    ARTIFACTS, DATA, MODELS, OUT, RAW, build, initialize,
    load_partition, sample_observations,
)
from .phase2b_metrics import evaluate_frame, metrics_summary, route_continuity
from .phase2b_train import (
    DEVICE_CUDA, CPU_THREADS, fit_mlp, fit_sklearn, predict_mlp,
    predict_sklearn, peak_rss_mb,
)

torch.set_num_threads(CPU_THREADS)
np.random.seed(42)
torch.manual_seed(42)


def load_saved(artifact_path):
    if artifact_path.suffix == ".joblib":
        bundle = joblib.load(artifact_path)
        return bundle, "sklearn"
    bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
    return bundle, "torch"


def evaluate_saved_model(model_name, strategy, split="student_test"):
    config = json.loads((OUT / "config.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["selected_model"] == model_name, (model_name, selection)
    partition = load_partition(split)
    frame = partition[partition.trip_id.isin(
        {p.name for p in (RAW / split).iterdir() if p.is_dir()})]
    # Student testing: evaluate every labeled observation (full development set).
    frame = frame[frame.weak_target >= 0].copy()
    art_dir = ARTIFACTS / "final"
    bundle, kind = load_saved(art_dir / f"{model_name}.joblib")
    scaler = bundle["scaler"]
    X = np.ascontiguousarray(frame[FEATURES].to_numpy(dtype=np.float32))
    Xs = scaler.transform(X)
    t0 = time.perf_counter()
    if kind == "sklearn":
        scores = predict_sklearn(bundle["estimator"], Xs)
    else:
        model = bundle["model"]
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(
                model(torch.tensor(Xs, dtype=torch.float32))).cpu().numpy()
    predict_s = time.perf_counter() - t0
    frame["candidate_score"] = scores
    obs, cand = evaluate_frame(frame)
    from .data import load_trip
    arcs_lookup = {}
    for directory in sorted((RAW / split).iterdir()):
        if directory.is_dir():
            try:
                arcs_lookup[directory.name] = load_trip(directory).arcs
            except Exception:  # pragma: no cover
                pass
    cont = route_continuity(frame, arcs_lookup)
    result = dict(
        model=model_name, strategy=strategy, split=split,
        device="cuda" if kind == "torch" and torch.cuda.is_available() else "CPU",
        predict_s=predict_s, predict_s_per_obs=predict_s / max(1, len(frame)),
        val_observations=int(obs["observation_count"]),
    )
    result.update(metrics_summary(obs, cand, prefix="obs_"))
    result.update(metrics_summary(obs, cand, prefix="cand_"))
    result["cont_direct_adjacency"] = float(
        cont.top1_direct_adjacency.mean()) if len(cont) else np.nan
    # Per-trip breakdown for audit.
    trip_rows = []
    for trip_id, group in frame.groupby("trip_id", sort=True):
        o, c = evaluate_frame(group)
        trip_rows.append(dict(trip_id=trip_id, observations=o["observation_count"],
                              top1_agreement=o["top1_agreement"], mrr=o["mrr"],
                              candidate_macro_f1=c["candidate_macro_f1"],
                              candidate_miss_rate=o["candidate_miss_rate"]))
    trips = pd.DataFrame(trip_rows)
    save_json(ARTIFACTS / "final" / f"student_eval_{model_name}.json", result)
    trips.to_csv(ARTIFACTS / "final" / f"student_eval_{model_name}_trips.csv", index=False)
    # Prediction export for the frontend contract: one row per observation Top-1.
    frame = frame.sort_values(["candidate_score", "candidate_rank"],
                              ascending=[False, True], kind="mergesort")
    top = frame.groupby(["trip_id", "observation_index"], sort=True).head(1)
    export = top[["trip_id", "observation_index", "longitude", "latitude",
                  "timestamp", "arc_id", "candidate_score", "candidate_rank"]]
    export = export.rename(columns={"arc_id": "top1_arc_id"})
    export.to_parquet(ARTIFACTS / "final" / f"student_predictions_{model_name}.parquet",
                      index=False)
    print(json.dumps(result, indent=2), flush=True)
    return result, trips


def main():
    initialize()
    selection = json.loads((OUT / "selection.json").read_text())
    model_name = selection["selected_model"]
    strategy = "consensus"  # primary supervision strategy per selection design
    config = json.loads((OUT / "config.json").read_text())
    art_dir = ARTIFACTS / "final"
    art_dir.mkdir(parents=True, exist_ok=True)
    train_partition = load_partition("train")
    train_keys, _ = sample_observations(
        train_partition,
        {p.name for p in (RAW / "train").iterdir() if p.is_dir()},
        strategy, config)
    scaler_path = art_dir / "scaler.joblib"
    from sklearn.preprocessing import StandardScaler
    X_tr = np.ascontiguousarray(train_keys[FEATURES].to_numpy(dtype=np.float32))
    y_tr = train_keys.weak_target.to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    mem0 = peak_rss_mb()
    t0 = time.perf_counter()
    if model_name == "mlp":
        params = config["models"]["mlp"]
        model, _, train_s = fit_mlp(params, X_tr_s, y_tr)
        bundle = {"model": model, "features": FEATURES,
                  "schema_version": "pathsnap-candidates-v1"}
        torch.save(bundle, art_dir / f"{model_name}.pt")
        artifact_path = art_dir / f"{model_name}.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        est, train_s = fit_sklearn(model_name, config["models"][model_name],
                                   X_tr_s, y_tr)
        bundle = {"estimator": est, "features": FEATURES,
                  "schema_version": "pathsnap-candidates-v1"}
        joblib.dump({"estimator": est, "scaler": scaler, "features": FEATURES,
                     "schema_version": "pathsnap-candidates-v1"},
                    art_dir / f"{model_name}.joblib", compress=3)
        artifact_path = art_dir / f"{model_name}.joblib"
        device = "CPU"
    train_s_total = time.perf_counter() - t0
    joblib.dump(scaler, scaler_path, compress=3)
    save_json(art_dir / "final_training.json", {
        "selected_model": model_name, "strategy": strategy,
        "train_rows": int(len(train_keys)),
        "train_seconds": train_s_total, "device": device,
        "peak_rss_mb": peak_rss_mb() - mem0,
        "artifact_path": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "config_sha256": selection["config_sha256"],
    })
    result, trips = evaluate_saved_model(model_name, strategy, "student_test")
    save_json(art_dir / "final_training.json",
              {**json.loads((art_dir / "final_training.json").read_text()), **result})
    print(f"FINAL {model_name} student-test Top-1 {result['obs_top1_agreement']:.4f}",
          flush=True)
    return result, trips


if __name__ == "__main__":
    main()
