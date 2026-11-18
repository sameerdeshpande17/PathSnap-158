"""Load a checkpoint and predict a complete trip without accessing .route."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pathsnap.candidates import generate, load_inputs
from pathsnap.explore import ROOT
from pathsnap.training import (
    ARTIFACTS_ROOT, environment_snapshot, list_runs, load_model_from_run,
    model_dir, run_dir,
)


def predict_trip(model_name: str, trip_id: str,
                 run_id: str | None = None,
                 split: str = "student_test",
                 config: dict | None = None) -> dict[str, Any]:
    """Predict every observation of a trip using only track/nodes/arcs.

    Does not read .route. Returns per-candidate scores plus Top-1 export.
    """
    import json as _json
    cfg_path = ROOT / "experiments" / "phase2b" / "config.json"
    config = config or (_json.loads(cfg_path.read_text()) if cfg_path.exists()
                        else {})
    feature_schema = config.get("feature_schema", [])

    wrapper, run_id, run_path = load_model_from_run(model_name, run_id)
    trip_dir = ROOT / "data" / "raw" / "Student folder" / split / trip_id
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip not found: {trip_dir}")
    trip = load_inputs(trip_dir)
    gen = generate(trip, config.get("candidates", {}), trip_id)
    if len(gen) == 0:
        raise ValueError(f"No candidates generated for {trip_id}")

    X = np.ascontiguousarray(gen[feature_schema].to_numpy(dtype=np.float32))
    scores = wrapper.predict_proba(X)
    gen = gen.copy()
    gen["candidate_score"] = scores

    gen = gen.sort_values(["candidate_score", "candidate_rank"],
                          ascending=[False, True], kind="mergesort")
    top = gen.groupby("observation_index", sort=True).head(1)
    export = top[["observation_index", "longitude", "latitude",
                  "arc_id", "candidate_score", "candidate_rank"]]
    export = export.rename(columns={"arc_id": "top1_arc_id"})
    export.insert(0, "trip_id", trip_id)

    out_dir = ARTIFACTS_ROOT / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base = out_dir / f"{model_name}_{split}_{trip_id}_{stamp}"
    export.to_csv(f"{base}.csv", index=False)
    gen.to_parquet(f"{base}_all_candidates.parquet", index=False)

    result = dict(
        model=model_name, run_id=run_id, run_dir=str(run_path),
        trip_id=trip_id, split=split, device=wrapper.device,
        n_observations=int(gen.observation_index.nunique()),
        n_candidates=int(len(gen)),
        top1_export=str(f"{base}.csv"),
        all_candidates=str(f"{base}_all_candidates.parquet"),
    )
    print(json.dumps(result, indent=2))
    print(export.to_string(index=False))
    return result


def list_available_runs() -> None:
    for model_name in sorted(__import__("pathsnap.models", fromlist=["ALL_MODELS"]).ALL_MODELS):
        runs = list_runs(model_name)
        print(f"\n{model_name}:")
        if not runs:
            print("  (no runs)")
            continue
        for r in runs:
            mark = " BEST" if r.get("best") else ""
            print(f"  {r['run_id']} device={r.get('device')}{mark}")
