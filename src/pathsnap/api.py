"""FastAPI backend for PathSnap map matching.

Endpoints:
  GET  /health                      - Server status, model readiness
  GET  /api/trips                   - List available trips
  POST /api/predict                 - Predict route for a trip
  POST /api/evaluate                - Evaluate a trip (reference-aware)
  GET  /api/model                   - Model metadata

Coordinates use GeoJSON longitude-latitude order.
No reference-route data is used during prediction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pathsnap.candidates import FEATURES, generate, load_inputs
from pathsnap.data import project
from pathsnap.hmm import CandidateState, HMMSettings, MapMatchingHMM
from pathsnap.map_matching_pipeline import MapMatchingPipeline, PipelineConfig
from pathsnap.route_reconstruct import RouteReconstructor
from pathsnap.viterbi import ViterbiDecoder

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="GPS-to-road map matching backend using classifier + HMM/Viterbi",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------
MODEL_NAME: str | None = None
MODEL_DEVICE: str = "CPU"
MODEL_READY: bool = False
PIPELINE: MapMatchingPipeline | None = None
MODEL_METADATA: dict[str, Any] = {}

AVAILABLE_TRIPS: dict[str, dict[str, Any]] = {}
REQUEST_LOG: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    trip_id: str
    split: str = "student_test"
    radius_m: float = 50.0
    max_candidates: int = 32


class PredictResponse(BaseModel):
    trip_id: str
    split: str
    model: str
    device: str
    n_observations: int
    n_candidates: int
    classifier_top1_arcs: list[int]
    hmm_decoded_arcs: list[int]
    hmm_log_prob: float
    route_n_segments: int
    route_n_connectors: int
    route_disconnected_gaps: int
    route_total_length_m: float
    arc_id_sequence: list[int]
    timing_s: dict[str, float]


class EvaluateRequest(BaseModel):
    trip_id: str
    split: str = "student_test"
    radius_m: float = 50.0
    max_candidates: int = 32


class EvaluateResponse(BaseModel):
    trip_id: str
    split: str
    model: str
    n_observations: int
    n_candidates: int
    classifier_top1_agreement_rate: float
    hmm_agreement_rate: float
    hmm_log_prob: float
    route_precision: float
    route_recall: float
    route_f1: float
    route_connectivity: float
    route_edit_distance: int
    route_n_segments: int
    route_n_connectors: int
    route_disconnected_gaps: int
    route_repeated_traversals: int
    route_total_length_m: float
    timing_s: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_name: str | None
    model_device: str
    cuda_available: bool
    trips_available: int


class ModelInfoResponse(BaseModel):
    model_name: str
    device: str
    n_features: int
    feature_schema: list[str]
    strategy: str
    train_seconds: float | None
    best_cv_top1: float | None


class TripInfo(BaseModel):
    trip_id: str
    split: str
    n_observations: int
    n_nodes: int
    n_arcs: int


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    """Load the selected model checkpoint and index available trips."""
    global MODEL_NAME, MODEL_DEVICE, MODEL_READY, PIPELINE, MODEL_METADATA, AVAILABLE_TRIPS

    # Index available trips
    for split in ["train", "student_test"]:
        split_dir = ROOT / "data" / "raw" / "Student folder" / split
        if not split_dir.is_dir():
            continue
        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            try:
                trip = load_inputs(trip_dir)
                key = f"{split}/{trip_dir.name}"
                AVAILABLE_TRIPS[key] = {
                    "trip_id": trip_dir.name,
                    "split": split,
                    "n_observations": len(trip.track),
                    "n_nodes": len(trip.nodes),
                    "n_arcs": len(trip.arcs),
                }
            except Exception:
                continue

    # Load the selected model
    model_dir = ROOT / "artifacts" / "models" / "hist_gradient_boosting"
    if model_dir.exists():
        run_dirs = sorted(model_dir.iterdir())
        for run_dir in run_dirs:
            pipeline_path = run_dir / "pipeline.joblib"
            if pipeline_path.exists():
                try:
                    bundle = joblib.load(pipeline_path)
                    MODEL_NAME = bundle.get("model_name", "hist_gradient_boosting")
                    MODEL_DEVICE = bundle.get("device", "CPU")
                    PIPELINE = MapMatchingPipeline(
                        PipelineConfig(
                            model_name=str(MODEL_NAME),
                            model_run_dir=run_dir,
                            hmm_settings=HMMSettings(),
                        )
                    )
                    MODEL_READY = True
                    MODEL_METADATA = {
                        "model_name": str(MODEL_NAME),
                        "device": str(MODEL_DEVICE),
                        "n_features": int(bundle.get("n_features", 0)),
                        "feature_schema": list(bundle.get("feature_names", [])),
                        "strategy": "consensus",
                        "train_seconds": float(bundle.get("metrics", {}).get("train_seconds", 0.0)),
                    }
                    break
                except Exception as exc:
                    print(f"WARNING: Failed to load model from {pipeline_path}: {exc}")
                    continue

    if not MODEL_READY:
        print("WARNING: No model checkpoint found. API will return errors on predict.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Server status and model readiness."""
    import torch

    return HealthResponse(
        status="ready" if MODEL_READY else "not_ready",
        model_ready=MODEL_READY,
        model_name=MODEL_NAME,
        model_device=MODEL_DEVICE,
        cuda_available=bool(torch.cuda.is_available()),
        trips_available=len(AVAILABLE_TRIPS),
    )


@app.get("/api/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return selected model metadata and inference configuration."""
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name=str(MODEL_NAME),
        device=str(MODEL_DEVICE),
        n_features=int(MODEL_METADATA.get("n_features", 0)),
        feature_schema=list(MODEL_METADATA.get("feature_schema", [])),
        strategy=str(MODEL_METADATA.get("strategy", "consensus")),
        train_seconds=MODEL_METADATA.get("train_seconds"),
        best_cv_top1=None,
    )


@app.get("/api/trips", response_model=list[TripInfo])
def list_trips(
    split: str = Query("student_test", pattern="^(train|student_test)$"),
) -> list[TripInfo]:
    """List available trips for a split."""
    return [
        TripInfo(
            trip_id=v["trip_id"],
            split=v["split"],
            n_observations=v["n_observations"],
            n_nodes=v["n_nodes"],
            n_arcs=v["n_arcs"],
        )
        for v in AVAILABLE_TRIPS.values()
        if v["split"] == split
    ]


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Run the full map-matching pipeline on a trip.

    Uses GPS observations + road network + trained classifier only.
    No reference-route (.route) data is accessed during inference.
    """
    if not MODEL_READY or PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    key = f"{req.split}/{req.trip_id}"
    if key not in AVAILABLE_TRIPS:
        raise HTTPException(status_code=404, detail=f"Trip not found: {key}")

    req_start = time.perf_counter()

    # Temporarily override candidate config
    orig_radius = PIPELINE.config.candidate_radius_m
    orig_max_cand = PIPELINE.config.candidate_max_candidates
    PIPELINE.config.candidate_radius_m = req.radius_m
    PIPELINE.config.candidate_max_candidates = req.max_candidates

    try:
        pred = PIPELINE.decode_trip(req.trip_id, split=req.split)
    finally:
        PIPELINE.config.candidate_radius_m = orig_radius
        PIPELINE.config.candidate_max_candidates = orig_max_cand

    total_s = time.perf_counter() - req_start

    # Build arc_id sequence from route
    arc_sequence = []
    if pred.route is not None:
        arc_sequence = list(pred.route.arc_id_sequence)

    # Log request
    REQUEST_LOG.append({
        "endpoint": "/api/predict",
        "trip_id": req.trip_id,
        "split": req.split,
        "status": "success",
        "duration_s": total_s,
        "timestamp": time.time(),
    })

    return PredictResponse(
        trip_id=pred.trip_id,
        split=pred.split,
        model=str(MODEL_NAME),
        device=str(MODEL_DEVICE),
        n_observations=pred.n_observations,
        n_candidates=pred.n_candidates,
        classifier_top1_arcs=list(pred.classifier_top1_arc),
        hmm_decoded_arcs=list(pred.hmm_decoded_arcs),
        hmm_log_prob=float(pred.hmm_log_prob),
        route_n_segments=int(pred.route.n_route_segments) if pred.route else 0,
        route_n_connectors=int(pred.route.n_connector_segments) if pred.route else 0,
        route_disconnected_gaps=int(pred.route.n_disconnected_gaps) if pred.route else 0,
        route_total_length_m=float(pred.route.total_length_m) if pred.route else 0.0,
        arc_id_sequence=[int(a) for a in arc_sequence],
        timing_s={
            "candidate_gen": float(pred.candidate_gen_s),
            "classifier_score": float(pred.classifier_score_s),
            "viterbi": float(pred.viterbi_s),
            "route_reconstruct": float(pred.route_reconstruct_s),
            "total": float(total_s),
        },
    )


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Evaluate a trip using reference-route data (development only).

    Computes route-level metrics comparing classifier Top-1 vs HMM/Viterbi
    predictions against the supplied reference route. Reference data is used
    for evaluation only, never for inference.
    """
    if not MODEL_READY or PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    key = f"{req.split}/{req.trip_id}"
    if key not in AVAILABLE_TRIPS:
        raise HTTPException(status_code=404, detail=f"Trip not found: {key}")

    req_start = time.perf_counter()

    # Load trip with reference route
    raw = ROOT / "data" / "raw" / "Student folder" / req.split / req.trip_id
    trip = load_inputs(raw)

    # Read reference route
    prefix = raw / raw.name
    route_path = prefix.with_suffix(".route")
    if not route_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference route not found for {key}")
    reference_arcs = [int(x) for x in route_path.read_text().split()]

    # Run pipeline (no .route during inference)
    orig_radius = PIPELINE.config.candidate_radius_m
    orig_max_cand = PIPELINE.config.candidate_max_candidates
    PIPELINE.config.candidate_radius_m = req.radius_m
    PIPELINE.config.candidate_max_candidates = req.max_candidates

    try:
        pred = PIPELINE.decode_trip(req.trip_id, split=req.split)
    finally:
        PIPELINE.config.candidate_radius_m = orig_radius
        PIPELINE.config.candidate_max_candidates = orig_max_cand

    total_s = time.perf_counter() - req_start

    # --- Evaluation metrics ---
    n_obs = len(trip.track)

    # Classifier Top-1 vs reference (weak-label agreement)
    cls_match = sum(
        1 for i, a in enumerate(pred.classifier_top1_arc)
        if a >= 0 and i < len(reference_arcs) and a == reference_arcs[i]
    )
    cls_agreement = cls_match / max(n_obs, 1)

    # HMM/Viterbi vs reference
    hmm_match = sum(
        1 for i, a in enumerate(pred.hmm_decoded_arcs)
        if a >= 0 and i < len(reference_arcs) and a == reference_arcs[i]
    )
    hmm_agreement = hmm_match / max(n_obs, 1)

    # Route-level: route arc ID sequence vs reference arc ID sequence
    pred_arc_seq = [a for a in pred.hmm_decoded_arcs if a >= 0]
    ref_arc_seq = reference_arcs

    # Route precision: fraction of predicted arcs that appear in reference
    ref_set = set(ref_arc_seq)
    route_precision = (
        sum(1 for a in pred_arc_seq if a in ref_set) / max(len(pred_arc_seq), 1)
    )

    # Route recall: fraction of reference arcs that appear in prediction
    pred_set = set(pred_arc_seq)
    route_recall = (
        sum(1 for a in ref_arc_seq if a in pred_set) / max(len(ref_arc_seq), 1)
    )

    # Route F1
    route_f1 = (
        2 * route_precision * route_recall / max(route_precision + route_recall, 1e-9)
    )

    # Directed connectivity: fraction of consecutive prediction pairs
    # that are directly connected in the road graph
    arc_lookup = {int(a): (int(trip.arcs[a, 0]), int(trip.arcs[a, 1])) for a in range(len(trip.arcs))}
    connected = 0
    total_pairs = 0
    for i in range(len(pred_arc_seq) - 1):
        a1, a2 = pred_arc_seq[i], pred_arc_seq[i + 1]
        if a1 in arc_lookup and a2 in arc_lookup:
            total_pairs += 1
            if arc_lookup[a1][1] == arc_lookup[a2][0]:
                connected += 1
    route_connectivity = connected / max(total_pairs, 1)

    # Edit distance (Levenshtein) between predicted and reference arc sequences
    def _edit_distance(a: list[int], b: list[int]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost,  # substitution
                )
        return dp[m][n]

    edit_dist = _edit_distance(pred_arc_seq, ref_arc_seq)

    # Log request
    REQUEST_LOG.append({
        "endpoint": "/api/evaluate",
        "trip_id": req.trip_id,
        "split": req.split,
        "status": "success",
        "duration_s": total_s,
        "timestamp": time.time(),
    })

    return EvaluateResponse(
        trip_id=req.trip_id,
        split=req.split,
        model=str(MODEL_NAME),
        n_observations=n_obs,
        n_candidates=pred.n_candidates,
        classifier_top1_agreement_rate=float(cls_agreement),
        hmm_agreement_rate=float(hmm_agreement),
        hmm_log_prob=float(pred.hmm_log_prob),
        route_precision=float(route_precision),
        route_recall=float(route_recall),
        route_f1=float(route_f1),
        route_connectivity=float(route_connectivity),
        route_edit_distance=int(edit_dist),
        route_n_segments=int(pred.route.n_route_segments) if pred.route else 0,
        route_n_connectors=int(pred.route.n_connector_segments) if pred.route else 0,
        route_disconnected_gaps=int(pred.route.n_disconnected_gaps) if pred.route else 0,
        route_repeated_traversals=int(pred.route.n_repeated_arc_traversals) if pred.route else 0,
        route_total_length_m=float(pred.route.total_length_m) if pred.route else 0.0,
        timing_s={
            "candidate_gen": float(pred.candidate_gen_s),
            "classifier_score": float(pred.classifier_score_s),
            "viterbi": float(pred.viterbi_s),
            "route_reconstruct": float(pred.route_reconstruct_s),
            "total": float(total_s),
        },
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
