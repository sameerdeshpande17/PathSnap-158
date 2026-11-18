"""Candidate road-segment generation with content-addressed caching.

Inference-only: reads track/nodes/arcs. Never opens .route files.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

from pathsnap.data import read_table, project

FEATURES = [
    "distance_m",
    "log_distance",
    "relative_distance_m",
    "distance_ratio",
    "candidate_rank",
    "candidate_count",
    "nearby_arc_count",
    "distance_tie_count",
    "segment_length_m",
    "log_segment_length",
    "road_heading_sin",
    "road_heading_cos",
    "gps_heading_sin",
    "gps_heading_cos",
    "heading_difference_rad",
    "directional_agreement",
    "speed_mps",
    "dt_s",
    "movement_m",
    "heading_valid",
    "road_heading_valid",
    "large_gap",
    "time_valid",
    "projection_fraction",
    "endpoint_distance_m",
    "from_outdegree",
    "to_indegree",
    "fallback_used",
]
SCHEMA_VERSION = "pathsnap-candidates-v1"

DEFAULT_CANDIDATE_CONFIG = {
    "radius_m": 50.0,
    "max_candidates": 12,
    "fallback": "nearest",
    "gap_s": 5.0,
}


@dataclass
class InferenceTrip:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def load_inputs(directory) -> InferenceTrip:
    """Read exactly track/nodes/arcs; the reference route is neither required nor opened."""
    directory = Path(directory)
    prefix = directory / directory.name
    trip = InferenceTrip(
        read_table(prefix.with_suffix(".track"), 3),
        read_table(prefix.with_suffix(".nodes"), 2),
        read_table(prefix.with_suffix(".arcs"), 2, True),
    )
    validate_inputs(trip)
    return trip


def validate_inputs(trip: InferenceTrip) -> None:
    for array, cols, name in [(trip.track, 3, "track"), (trip.nodes, 2, "nodes"),
                              (trip.arcs, 2, "arcs")]:
        if array.ndim != 2 or array.shape[1] != cols or len(array) == 0:
            raise ValueError(f"Invalid or empty {name}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {name}")
    for coords in [trip.track[:, :2], trip.nodes]:
        if (np.abs(coords[:, 0]) > 180).any() or (np.abs(coords[:, 1]) > 90).any():
            raise ValueError("Invalid longitude/latitude")
    if not np.issubdtype(trip.arcs.dtype, np.integer):
        raise ValueError("Noninteger arc IDs")
    if (trip.arcs < 0).any() or (trip.arcs >= len(trip.nodes)).any():
        raise ValueError("Arc endpoint out of range")


def movement_features(track, gps_xy, gap_s=5.0):
    dt = np.r_[0.0, np.diff(track[:, 2])]
    delta = np.vstack([np.zeros(2), np.diff(gps_xy, axis=0)])
    step = np.linalg.norm(delta, axis=1)
    time_valid = dt > 0
    valid = time_valid & (dt <= gap_s) & (step >= 1.0)
    heading = np.arctan2(delta[:, 0], delta[:, 1])
    heading[~valid] = 0.0
    speed = np.divide(step, dt, out=np.zeros_like(step), where=time_valid)
    return dt, step, heading, speed, valid, time_valid


def generate(trip: InferenceTrip, config: dict | None = None,
             trip_id: str = "unseen") -> pd.DataFrame:
    """Generate inference-safe candidate rows for every GPS observation.

    Returns columns: trip_id, observation_index, arc_id, then FEATURES.
    Observations with no nearby candidates use the global nearest arc when
    fallback="nearest", or are omitted when fallback="none".
    """
    cfg = dict(DEFAULT_CANDIDATE_CONFIG)
    cfg.update(config or {})
    validate_inputs(trip)
    radius = float(cfg["radius_m"])
    cap = int(cfg["max_candidates"])
    if radius <= 0 or cap < 1 or cfg["fallback"] not in ("nearest", "none"):
        raise ValueError("Invalid candidate configuration")
    gps, nodes = project(trip.track, trip.nodes)
    segments = shapely.linestrings(nodes[trip.arcs])
    points = shapely.points(gps)
    tree = STRtree(segments)
    pairs = tree.query(points, predicate="dwithin", distance=radius)
    obs, arc = pairs
    count = np.bincount(obs, minlength=len(gps))
    fallback = count == 0
    if fallback.any() and cfg["fallback"] == "nearest":
        missing = np.flatnonzero(fallback)
        near = tree.query_nearest(points[missing], all_matches=True)
        obs = np.r_[obs, missing[near[0]]]
        arc = np.r_[arc, near[1]]
    if not len(obs):
        return pd.DataFrame(columns=["trip_id", "observation_index", "arc_id"] + FEATURES)
    distance = shapely.distance(points[obs], segments[arc])
    order = np.lexsort((arc, distance, obs))
    obs, arc, distance = obs[order], arc[order], distance[order]
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    rank = np.arange(len(obs)) - np.repeat(
        starts, np.diff(np.r_[starts, len(obs)]))
    keep = rank < cap
    obs, arc, distance, rank = obs[keep], arc[keep], distance[keep], rank[keep]
    selected_count = np.bincount(obs, minlength=len(gps))
    minimum = np.full(len(gps), np.inf)
    np.minimum.at(minimum, obs, distance)
    near_ties = np.bincount(
        obs[distance <= minimum[obs] + 1.0], minlength=len(gps))
    vector = nodes[trip.arcs[arc, 1]] - nodes[trip.arcs[arc, 0]]
    length = np.linalg.norm(vector, axis=1)
    road_valid = length > 0
    road_heading = np.arctan2(vector[:, 0], vector[:, 1])
    dt, step, heading, speed, heading_valid, time_valid = movement_features(
        trip.track, gps, cfg["gap_s"])
    heading_difference = np.abs(np.arctan2(
        np.sin(road_heading - heading[obs]),
        np.cos(road_heading - heading[obs])))
    both_valid = heading_valid[obs] & road_valid
    heading_difference[~both_valid] = 0.0
    agreement = np.where(both_valid, np.cos(heading_difference), 0.0)
    displacement = gps[obs] - nodes[trip.arcs[arc, 0]]
    fraction = np.clip(
        np.divide(
            np.sum(displacement * vector, axis=1),
            length ** 2,
            out=np.zeros_like(length),
            where=length > 0,
        ),
        0,
        1,
    )
    outdegree = np.bincount(trip.arcs[:, 0], minlength=len(nodes))
    indegree = np.bincount(trip.arcs[:, 1], minlength=len(nodes))
    frame = pd.DataFrame({
        "trip_id": trip_id,
        "observation_index": obs.astype(np.int32),
        "arc_id": arc.astype(np.int32),
        "distance_m": distance,
        "log_distance": np.log1p(distance),
        "relative_distance_m": distance - minimum[obs],
        "distance_ratio": distance / (minimum[obs] + 1.0),
        "candidate_rank": rank,
        "candidate_count": selected_count[obs],
        "nearby_arc_count": count[obs],
        "distance_tie_count": near_ties[obs],
        "segment_length_m": length,
        "log_segment_length": np.log1p(length),
        "road_heading_sin": np.where(road_valid, np.sin(road_heading), 0.0),
        "road_heading_cos": np.where(road_valid, np.cos(road_heading), 0.0),
        "gps_heading_sin": np.where(heading_valid[obs], np.sin(heading[obs]), 0.0),
        "gps_heading_cos": np.where(heading_valid[obs], np.cos(heading[obs]), 0.0),
        "heading_difference_rad": heading_difference,
        "directional_agreement": agreement,
        "speed_mps": np.clip(speed[obs], 0, 100),
        "dt_s": np.clip(dt[obs], 0, 300),
        "movement_m": np.clip(step[obs], 0, 1000),
        "heading_valid": heading_valid[obs],
        "road_heading_valid": road_valid,
        "large_gap": dt[obs] > cfg["gap_s"],
        "time_valid": time_valid[obs],
        "projection_fraction": fraction,
        "endpoint_distance_m": np.minimum(fraction, 1 - fraction) * length,
        "from_outdegree": outdegree[trip.arcs[arc, 0]],
        "to_indegree": indegree[trip.arcs[arc, 1]],
        "fallback_used": fallback[obs],
    })
    frame[FEATURES] = frame[FEATURES].astype(np.float32)
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Nonfinite generated features")
    return frame


def feature_matrix(frame: pd.DataFrame, names: list[str] | None = None) -> np.ndarray:
    names = FEATURES if names is None else names
    if list(names) != FEATURES:
        raise ValueError("Feature schema mismatch")
    return np.ascontiguousarray(frame[FEATURES].to_numpy(dtype=np.float32))


def cache_key(dataset_identity: str, config: dict, alignment_version: str) -> str:
    """Content-addressed cache key: dataset + feature config + alignment version."""
    payload = json.dumps({
        "dataset_identity": dataset_identity,
        "candidate_config": config,
        "feature_schema": FEATURES,
        "schema_version": SCHEMA_VERSION,
        "alignment_version": alignment_version,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def dataset_identity_from_trips(trip_dirs: list[Path]) -> str:
    """Hash of all track/nodes/arcs files for cache invalidation."""
    h = hashlib.sha256()
    for d in sorted(trip_dirs):
        for suffix in (".track", ".nodes", ".arcs"):
            f = d / f"{d.name}{suffix}"
            if f.exists():
                h.update(f.read_bytes())
                h.update(suffix.encode())
    return h.hexdigest()[:24]


class CandidateCache:
    """On-disk parquet cache keyed by dataset/config/alignment hash."""

    def __init__(self, root: Path | str = "data/candidates/cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.parquet"

    def cache_key(self, dataset_identity: str, config: dict,
                  alignment_version: str) -> str:
        return cache_key(dataset_identity, config, alignment_version)

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def read(self, key: str) -> pd.DataFrame:
        p = self.path(key)
        if not p.exists():
            raise FileNotFoundError(f"Cache miss: {key}")
        return pd.read_parquet(p)

    def write(self, key: str, frame: pd.DataFrame) -> None:
        self.path(key).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.path(key), index=False)

    def manifest(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.parquet")):
            out.append(dict(key=p.stem, bytes=p.stat().st_size,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        return out
