"""Load trained models and evaluate — never triggers training automatically."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pathsnap.candidates import generate, load_inputs
from pathsnap.explore import ROOT
from pathsnap.labels import build_label_table, join_labels_with_candidates
from pathsnap.models import MODEL_REGISTRY
from pathsnap.training import (
    ARTIFACTS_ROOT, build_validation_frame, environment_snapshot, list_runs,
    load_model_from_run, model_dir, run_dir,
)
from pathsnap.phase2b_metrics import evaluate_frame, top1_arc

SPLITS = ["train", "student_test"]


def evaluate_model(model_name: str, split: str = "student_test",
                   run_id: str | None = None,
                   config: dict | None = None,
                   consensus_only: bool = True) -> dict[str, Any]:
    """Evaluate an existing trained model on a split. No retraining.

    Parameters
    ----------
    consensus_only : bool
        If True (default), observation-level metrics (Top-1, MRR) are computed
        on the B/C-consensus labeled population only, matching Phase 2A's
        recommended supervision (12,516 student-test observations). If False,
        the union of consensus + B-extra labels is used (12,862 observations).
        Candidate-level metrics always use all labeled candidates.
    """
    if split not in SPLITS:
        raise ValueError(f"Split must be one of {SPLITS}")
    wrapper, run_id, run_path = load_model_from_run(model_name, run_id)
    import json as _json
    cfg_path = ROOT / "experiments" / "phase2b" / "config.json"
    config = config or (_json.loads(cfg_path.read_text()) if cfg_path.exists()
                        else {})
    feature_schema = config.get("feature_schema", [])

    # Build full candidate frame for the split (all covered observations).
    raw = ROOT / "data" / "raw" / "Student folder" / split
    frames = []
    for trip_dir in sorted(raw.iterdir()):
        if not trip_dir.is_dir():
            continue
        trip = load_inputs(trip_dir)
        gen = generate(trip, config.get("candidates", {}), trip_dir.name)
        labels = build_label_table(split, trip_dir.name)
        frames.append(join_labels_with_candidates(gen, labels))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frame = frame[frame.weak_target >= 0].copy()

    X = np.ascontiguousarray(frame[feature_schema].to_numpy(dtype=np.float32))
    scores = wrapper.predict_proba(X)
    frame["candidate_score"] = scores

    t0 = time.time()
    obs_union, cand_union = evaluate_frame(frame, consensus_only=False)
    obs_consensus, cand_consensus = evaluate_frame(
        frame, consensus_only=True) if consensus_only else (obs_union, cand_union)
    eval_s = time.time() - t0

    # Use consensus observation metrics as headline (Phase 2A recommended supervision).
    obs = obs_consensus if consensus_only else obs_union
    cand = cand_union

    # Per-trip breakdown (consensus population).
    trip_rows = []
    consensus_frame = frame[frame.consensus_eligible] if consensus_only else frame
    for trip_id, group in consensus_frame.groupby("trip_id", sort=True):
        o, c = evaluate_frame(group, consensus_only=True)
        trip_rows.append(dict(
            trip_id=trip_id, observations=o["observation_count"],
            top1_agreement=o["top1_agreement"], mrr=o["mrr"],
            candidate_miss_rate=o["candidate_miss_rate"],
            candidate_macro_f1=c["candidate_macro_f1"],
        ))

    result = dict(
        model=model_name, run_id=run_id, run_dir=str(run_path),
        split=split, device=wrapper.device,
        eval_seconds=eval_s,
        population="consensus" if consensus_only else "consensus_plus_B",
        n_observations=int(obs["observation_count"]),
        n_observations_union=int(obs_union["observation_count"]),
        n_candidate_rows=int(len(frame)),
    )
    result.update({f"obs_{k}": v for k, v in obs.items() if k != "observation_count"})
    result.update({f"cand_{k}": v for k, v in cand.items()})
    result["top1_agreement"] = result.pop("obs_top1_agreement", None)
    result["candidate_macro_f1"] = result.pop("cand_candidate_macro_f1", None)
    result["candidate_log_loss"] = result.pop("cand_candidate_log_loss", None)
    result["candidate_miss_rate"] = result.pop("obs_candidate_miss_rate", None)
    result["mrr"] = result.pop("obs_mrr", None)
    result["environment"] = environment_snapshot()

    out_dir = ARTIFACTS_ROOT / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = out_dir / f"{model_name}_{split}_{stamp}.json"
    out.write_text(json.dumps(result, indent=2))
    pd.DataFrame(trip_rows).to_csv(
        out_dir / f"{model_name}_{split}_{stamp}_trips.csv", index=False)

    # Prediction export for frontend contract: Top-1 per observation.
    frame = frame.sort_values(["candidate_score", "candidate_rank"],
                              ascending=[False, True], kind="mergesort")
    top = frame.groupby(["trip_id", "observation_index"], sort=True).head(1)
    export = top[["trip_id", "observation_index", "longitude", "latitude",
                  "timestamp", "arc_id", "candidate_score", "candidate_rank"]]
    export = export.rename(columns={"arc_id": "top1_arc_id"})
    export.to_parquet(
        out_dir / f"{model_name}_{split}_{stamp}_predictions.parquet",
        index=False)

    print(json.dumps(result, indent=2))
    print(f"[pathsnap.evaluate] results saved to {out}", flush=True)
    return result


def evaluate_all(split: str = "student_test",
                 consensus_only: bool = True) -> list[dict[str, Any]]:
    results = []
    for model_name in sorted(MODEL_REGISTRY):
        best = __import__("pathsnap.training", fromlist=["load_best"]).load_best(model_name)
        if best is None:
            print(f"[pathsnap.evaluate] no checkpoint for {model_name}, skipping",
                  flush=True)
            continue
        try:
            evaluate_model(model_name, split=split, run_id=best["best_run_id"],
                           consensus_only=consensus_only)
        except Exception as exc:  # pragma: no cover - report and continue
            print(f"[pathsnap.evaluate] {model_name} failed: {exc}", flush=True)
    return results
