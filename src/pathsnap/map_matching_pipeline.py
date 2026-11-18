"""End-to-end map-matching pipeline: classifier + HMM + Viterbi + route reconstruction.

Loads a trained classifier checkpoint, generates road candidates, scores them,
decodes with HMM/Viterbi, and reconstructs a connected route.

Inference uses GPS observations + road network (nodes/arcs) + classifier only.
Reference-route files are never accessed during inference.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pathsnap.candidates import FEATURES, generate, load_inputs
from pathsnap.data import project
from pathsnap.explore import ROOT
from pathsnap.hmm import CandidateState, HMMSettings, MapMatchingHMM
from pathsnap.route_reconstruct import ReconstructedRoute, RouteReconstructor
from pathsnap.viterbi import ViterbiDecoder


@dataclass
class PipelineConfig:
    """Configuration for the full map-matching pipeline."""

    model_name: str
    model_run_dir: str | Path | None = None
    hmm_settings: HMMSettings | None = None
    candidate_radius_m: float = 50.0
    candidate_max_candidates: int = 32
    candidate_fallback: str = "nearest"
    gap_threshold_s: float = 5.0


@dataclass
class TripPrediction:
    """Complete prediction output for one trip."""

    trip_id: str
    split: str
    n_observations: int
    n_candidates: int
    # Classifier-only output (per observation)
    classifier_top1_arc: list[int] = field(default_factory=list)
    classifier_top1_prob: list[float] = field(default_factory=list)
    classifier_all_candidates: list[dict[str, Any]] = field(default_factory=list)
    # HMM/Viterbi output
    hmm_decoded_arcs: list[int] = field(default_factory=list)
    hmm_log_prob: float = 0.0
    hmm_n_connected: int = 0
    hmm_n_disconnected: int = 0
    hmm_n_dwells: int = 0
    # Reconstructed route
    route: ReconstructedRoute | None = None
    # Timing
    candidate_gen_s: float = 0.0
    classifier_score_s: float = 0.0
    viterbi_s: float = 0.0
    route_reconstruct_s: float = 0.0
    total_s: float = 0.0
    # Diagnostics
    device: str = "CPU"
    warnings: list[str] = field(default_factory=list)


class MapMatchingPipeline:
    """Loads a trained model and runs the full map-matching pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.device = "CPU"
        self._load_model()

    def _load_model(self) -> None:
        """Load a trained classifier checkpoint (joblib pipeline or PyTorch)."""
        run_dir = Path(self.config.model_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Model run directory not found: {run_dir}")

        pipeline_path = run_dir / "pipeline.joblib"
        pt_path = run_dir / "model.pt"

        if pipeline_path.exists():
            bundle = joblib.load(pipeline_path)
            self.model = bundle["estimator"]
            self.scaler = bundle["scaler"]
            self.feature_names = bundle["feature_names"]
            self.n_features = bundle["n_features"]
            self.device = bundle.get("device", "CPU")
            self.config.model_name = bundle.get("model_name", self.config.model_name)
        elif pt_path.exists():
            import torch

            metadata_path = run_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            self.device = metadata.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self.feature_names = metadata.get("feature_names", [])
            self.n_features = metadata.get("n_features", len(self.feature_names))

            class _MLPWrapper:
                def __init__(self, path, device, n_features):
                    import torch.nn as nn

                    class _MLP(nn.Module):
                        def __init__(self, nf):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(nf, 64), nn.ReLU(), nn.Dropout(0.1),
                                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                                nn.Linear(32, 1),
                            )

                        def forward(self, x):
                            return self.net(x).squeeze(1)

                    self.device = device
                    self.model = _MLP(n_features).to(device)
                    checkpoint = torch.load(path, map_location=device, weights_only=False)
                    self.model.load_state_dict(checkpoint["state_dict"])
                    self.model.eval()

                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    self.model.eval()
                    X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        probs = torch.sigmoid(self.model(X_t))
                    return probs.cpu().numpy()

            self.model = _MLPWrapper(pt_path, self.device, self.n_features)
        else:
            raise FileNotFoundError(
                f"No pipeline.joblib or model.pt found in {run_dir}"
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Classifier probability for each candidate row (positive class)."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)
        # sklearn predict_proba returns (n, 2) for binary; take class 1.
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba = proba[:, 1]
        return np.asarray(proba, dtype=np.float64)

    def decode_trip(
        self,
        trip_id: str,
        split: str = "student_test",
    ) -> TripPrediction:
        """Run the full pipeline on one trip.

        Does NOT access .route files. Uses track/nodes/arcs + classifier only.
        """
        t0 = time.perf_counter()
        raw = ROOT / "data" / "raw" / "Student folder" / split / trip_id
        if not raw.is_dir():
            raise FileNotFoundError(f"Trip not found: {raw}")

        # ---- Load trip data (track, nodes, arcs only) ----
        trip = load_inputs(raw)
        nodes = trip.nodes  # (N, 2) [lon, lat]
        arcs = trip.arcs    # (M, 2) [from_node, to_node]

        # ---- Build HMM network ----
        hmm = MapMatchingHMM(settings=self.config.hmm_settings)
        hmm.set_network(nodes, arcs)

        # ---- Generate candidates (inference-only) ----
        t1 = time.perf_counter()
        cand_config = {
            "radius_m": self.config.candidate_radius_m,
            "max_candidates": self.config.candidate_max_candidates,
            "fallback": self.config.candidate_fallback,
            "gap_s": self.config.gap_threshold_s,
        }
        candidates_df = generate(trip, cand_config, trip_id)
        t2 = time.perf_counter()
        candidate_gen_s = t2 - t1

        n_obs = len(trip.track)
        pred = TripPrediction(
            trip_id=trip_id,
            split=split,
            n_observations=n_obs,
            n_candidates=len(candidates_df),
            device=self.device,
        )

        if len(candidates_df) == 0:
            pred.warnings.append("No candidates generated for any observation")
            pred.total_s = time.perf_counter() - t0
            return pred

        # ---- Score candidates with classifier ----
        t3 = time.perf_counter()
        feature_cols = self.feature_names or FEATURES
        X = np.ascontiguousarray(candidates_df[feature_cols].to_numpy(dtype=np.float32))
        probs = self.predict_proba(X)
        t4 = time.perf_counter()
        classifier_score_s = t4 - t3

        candidates_df = candidates_df.copy()
        candidates_df["classifier_prob"] = probs

        # ---- Per-observation classifier Top-1 ----
        for obs_idx in range(n_obs):
            obs_cands = candidates_df[candidates_df.observation_index == obs_idx]
            if len(obs_cands) == 0:
                pred.classifier_top1_arc.append(-1)
                pred.classifier_top1_prob.append(0.0)
                continue
            best = obs_cands.loc[obs_cands.classifier_prob.idxmax()]
            pred.classifier_top1_arc.append(int(best.arc_id))
            pred.classifier_top1_prob.append(float(best.classifier_prob))

            pred.classifier_all_candidates.append(
                dict(
                    obs_index=obs_idx,
                    arc_id=int(best.arc_id),
                    probability=float(best.classifier_prob),
                    distance_m=float(best.distance_m),
                    n_candidates=len(obs_cands),
                )
            )

        # ---- Build HMM observation structures ----
        # Add node coords to candidates
        node_lon = nodes[:, 0]
        node_lat = nodes[:, 1]
        arc_from = arcs[:, 0]
        arc_to = arcs[:, 1]

        candidates_df["from_node"] = arc_from[candidates_df.arc_id.values]
        candidates_df["to_node"] = arc_to[candidates_df.arc_id.values]

        # GPS headings and displacements
        gps_xy, _ = project(trip.track, nodes)
        displacements = np.r_[0.0, np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)]
        headings = np.r_[0.0, np.arctan2(np.diff(gps_xy[:, 0]), np.diff(gps_xy[:, 1]))]
        dts = np.r_[0.0, np.diff(trip.track[:, 2])]

        # ---- Viterbi decoding ----
        decoder = ViterbiDecoder(hmm, settings=self.config.hmm_settings)

        # Build Viterbi observations, skipping those with no candidates
        viterbi_observations = []
        obs_idx_to_viterbi_idx = {}  # maps original obs_index -> viterbi position
        for obs_idx in range(n_obs):
            obs_cands = candidates_df[candidates_df.observation_index == obs_idx]
            if len(obs_cands) == 0:
                # No candidates - already handled in classifier Top-1 loop above
                continue
            obs_idx_to_viterbi_idx[obs_idx] = len(viterbi_observations)

            row_group = obs_cands.reset_index(drop=True)
            obs_probs = probs[candidates_df.observation_index == obs_idx]

            states = decoder.build_states(
                obs_index=obs_idx,
                gps_lon=float(trip.track[obs_idx, 0]),
                gps_lat=float(trip.track[obs_idx, 1]),
                gps_heading_rad=float(headings[obs_idx]),
                candidates_df_row_group=row_group,
                classifier_probs=obs_probs,
            )
            viterbi_observations.append(
                dict(
                    obs_index=obs_idx,
                    gps_lon=float(trip.track[obs_idx, 0]),
                    gps_lat=float(trip.track[obs_idx, 1]),
                    gps_heading_rad=float(headings[obs_idx]),
                    dt_s=float(dts[obs_idx]),
                    gps_displacement_m=float(displacements[obs_idx]),
                    states=states,
                )
            )

        t5 = time.perf_counter()
        viterbi_result = decoder.decode(viterbi_observations)
        t6 = time.perf_counter()
        viterbi_s = t6 - t5

        # Map Viterbi results back to original observation indices
        pred.hmm_decoded_arcs = [-1] * n_obs
        for v_pos, arc_id in enumerate(viterbi_result.decoded_arc_ids):
            orig_obs_idx = viterbi_observations[v_pos]["obs_index"]
            pred.hmm_decoded_arcs[orig_obs_idx] = arc_id
        pred.hmm_log_prob = viterbi_result.total_log_prob
        pred.hmm_n_connected = viterbi_result.n_connected_transitions
        pred.hmm_n_disconnected = viterbi_result.n_disconnected_transitions
        pred.hmm_n_dwells = viterbi_result.n_same_arc_dwells

        # ---- Route reconstruction ----
        t7 = time.perf_counter()
        arc_lookup = {int(a): (int(arcs[a, 0]), int(arcs[a, 1])) for a in range(len(arcs))}
        node_coords = {i: (float(nodes[i, 0]), float(nodes[i, 1])) for i in range(len(nodes))}

        reconstructor = RouteReconstructor(hmm)
        route = reconstructor.reconstruct(
            trip_id=trip_id,
            viterbi_result=viterbi_result,
            observation_details=viterbi_result.observation_details,
            arc_lookup=arc_lookup,
            node_coords=node_coords,
        )
        pred.route = route
        t8 = time.perf_counter()
        route_reconstruct_s = t8 - t7

        pred.total_s = time.perf_counter() - t0
        pred.candidate_gen_s = candidate_gen_s
        pred.classifier_score_s = classifier_score_s

        return pred
