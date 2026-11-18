"""Weak-label dataset preparation from Phase 2A alignment outputs.

Joins inference-time candidate features with frozen Phase 2A weak labels.
Reference-route identifiers and occurrence indices are metadata only — never
prediction features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pathsnap.explore import ROOT

LABELS_ROOT = ROOT / "data/alignments"
CANDIDATES_ROOT = ROOT / "data/candidates"
STRATEGIES = ["consensus", "consensus_plus_B"]

CONSENSUS_BUDGET = 20000
B_EXTRA_BUDGET = 5000
SEED = 42


def load_alignment_labels(split: str, trip_id: str) -> pd.DataFrame:
    """Load Phase 2A alignment outputs for one trip.

    Returns per-observation: consensus_eligible, B_extra_eligible,
    weak_arc_id, plus quality diagnostics. Reads B and C files only.
    """
    root = LABELS_ROOT / split
    b = pd.read_csv(root / f"{trip_id}_B.csv", dtype={"trip_id": str})
    c = pd.read_csv(root / f"{trip_id}_C.csv", dtype={"trip_id": str})
    consensus = c.recommended_supervision_use.to_numpy(dtype=bool)
    b_good = (b.quality_eligible.to_numpy(dtype=bool) &
              ~b.quality_flags.fillna("").str.contains("geometric_occurrence_ambiguity"))
    # C quality eligibility restricted to consensus columns in export; reconstruct:
    c_eligible = c.quality_eligible.to_numpy(dtype=bool)
    c_ambig = c.quality_flags.fillna("").str.contains("geometric_occurrence_ambiguity")
    consensus = consensus & c_eligible & ~c_ambig
    assert (b_good[consensus]).all(), "Consensus rows must be B-eligible"
    np.testing.assert_array_equal(
        b.assigned_arc_id[consensus].to_numpy(),
        c.assigned_arc_id[consensus].to_numpy(),
    )
    return pd.DataFrame({
        "trip_id": trip_id,
        "observation_index": b.observation_index.to_numpy(),
        "consensus_eligible": consensus,
        "B_extra_eligible": (b_good.to_numpy(dtype=bool) & ~consensus),
        "weak_arc_id": b.assigned_arc_id.to_numpy(),
        "b_distance_m": b.projection_distance_m.to_numpy(),
        "c_distance_m": c.projection_distance_m.to_numpy(),
    })


def build_label_table(split: str, trip_id: str) -> pd.DataFrame:
    """Public wrapper around load_alignment_labels."""
    return load_alignment_labels(split, trip_id)


def join_labels_with_candidates(candidates_frame: pd.DataFrame,
                                labels_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach weak labels to candidate rows by (trip_id, observation_index).

    weak_target = 1 when candidate arc == weak label arc AND observation is
    labeled (consensus or B-extra), else 0 for candidates of labeled
    observations, else -1 (unlabeled).
    positive_in_candidates = True when the weak label arc appears among
    candidates for that observation.
    """
    obs = candidates_frame.observation_index.to_numpy()
    weak = labels_frame.set_index("observation_index")
    labeled = labels_frame.consensus_eligible | labels_frame.B_extra_eligible
    weak_arc = weak.weak_arc_id.to_numpy()
    weak_target = np.where(
        labeled.to_numpy()[obs],
        (candidates_frame.arc_id.to_numpy() == weak_arc[obs]).astype(np.int8),
        -1,
    )
    covered = np.zeros(len(labels_frame), dtype=bool)
    covered[obs[weak_target == 1]] = True
    frame = candidates_frame.copy()
    frame["weak_target"] = weak_target
    frame["weak_arc_id"] = weak_arc[obs]
    frame["consensus_eligible"] = labels_frame.consensus_eligible.to_numpy()[obs]
    frame["B_extra_eligible"] = labels_frame.B_extra_eligible.to_numpy()[obs]
    frame["positive_in_candidates"] = covered[obs]
    return frame


def sample_observations(frame: pd.DataFrame, trip_ids: list[str], strategy: str,
                        config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete candidate groups for sampled observations.

    Never samples rows independently — whole observation candidate groups are
    selected. Round-robin across trips for balance without dependence on
    candidate multiplicity.
    """
    available = frame[frame.trip_id.isin(trip_ids) &
                      frame.positive_in_candidates]
    unique = available[["trip_id", "observation_index",
                        "consensus_eligible", "B_extra_eligible"]].drop_duplicates()
    selected = []
    for mask_name, budget in [
        ("consensus_eligible", int(config.get("consensus_observation_budget",
                                               CONSENSUS_BUDGET))),
        ("B_extra_eligible", int(config.get("B_extra_observation_budget",
                                            B_EXTRA_BUDGET)
                                    if strategy == "consensus_plus_B" else 0)),
    ]:
        if budget <= 0:
            continue
        groups = unique[unique[mask_name]].groupby("trip_id", sort=True)
        rng = np.random.default_rng(int(config.get("seed", SEED)))
        queues = [g.iloc[rng.permutation(len(g))] for _, g in groups]
        parts = []
        for g in queues:
            g = g.copy()
            g["_round"] = np.arange(len(g))
            parts.append(g)
        if parts:
            sample = pd.concat(parts).sort_values(
                ["_round", "trip_id"]).head(budget)
            selected.append(sample[["trip_id", "observation_index"]])
    if not selected:
        raise ValueError("No supervised fitting observations")
    keys = pd.concat(selected, ignore_index=True).drop_duplicates(
        ["trip_id", "observation_index"])
    result = available.merge(keys, on=["trip_id", "observation_index"],
                             how="inner")
    result = result.sort_values(
        ["trip_id", "observation_index", "candidate_rank"]).reset_index(drop=True)
    return result, keys


def available_trips(split: str) -> list[str]:
    raw = ROOT / "data/raw/Student folder" / split
    return sorted(p.name for p in raw.iterdir() if p.is_dir())
