"""Genuine weak-label agreement metrics; never claim ground-truth accuracy."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

FEATURE_DISCLAIMER = "Computed against frozen Phase 2A weak labels, not verified ground truth."


@dataclass
class ObsMetrics:
    covered: int
    missed: int
    miss_rate: float
    top1_agreement: float
    mrr: float
    selected_score_median: float
    coverage_score: float


def observation_scores(frame):
    """Highest predicted probability per observation; keep top arc and rank."""
    grouped = frame.sort_values(["candidate_rank"], kind="stable")
    best = grouped.groupby(["trip_id", "observation_index"], sort=True).tail(1)
    return best.set_index(["trip_id", "observation_index"])["candidate_score"]


def top1_arc(frame):
    grouped = frame.sort_values(["candidate_score", "candidate_rank"],
                                ascending=[False, True], kind="mergesort")
    return grouped.groupby(["trip_id", "observation_index"], sort=True).head(1).set_index(
        ["trip_id", "observation_index"])["arc_id"]


def reciprocal_ranks(frame):
    """Reciprocal rank of the first weak-positive candidate per observation.

    Candidates within each observation are ordered by predicted score
    (descending) then distance rank (ascending). Returns 1 / rank of the first
    candidate whose weak_target > 0; returns 0.0 when the positive arc is absent
    from the candidate set (a candidate miss). Indexed by
    (trip_id, observation_index).
    """
    ordered = frame.sort_values(
        ["candidate_score", "candidate_rank"], ascending=[False, True],
        kind="mergesort")
    rr = {}
    for key, group in ordered.groupby(
            ["trip_id", "observation_index"], sort=True):
        positives = group.weak_target.to_numpy()
        if not (positives > 0).any():
            rr[key] = 0.0
            continue
        hit = int(np.flatnonzero(positives > 0).min()) + 1  # 1-based rank
        rr[key] = 1.0 / hit
    return pd.Series(rr)


def candidate_metrics(y_true, y_score, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    macro_f1 = float(f1.mean())
    try:
        ll = float(log_loss(y_true, y_score, labels=[0, 1]))
    except ValueError:
        ll = float("nan")
    try:
        ap = float(average_precision_score(y_true, y_score))
    except ValueError:
        ap = float("nan")
    return dict(
        candidate_precision_class0=float(precision[0]),
        candidate_recall_class0=float(recall[0]),
        candidate_f1_class0=float(f1[0]),
        candidate_support_class0=int(support[0]),
        candidate_precision_class1=float(precision[1]),
        candidate_recall_class1=float(recall[1]),
        candidate_f1_class1=float(f1[1]),
        candidate_support_class1=int(support[1]),
        candidate_macro_f1=macro_f1,
        candidate_log_loss=ll,
        candidate_average_precision=ap,
        candidate_accuracy=float(np.mean(y_pred == y_true)),
    )


def evaluate_frame(frame, threshold=.5, consensus_only=False):
    """Returns per-observation and candidate-level metrics for one prediction frame.

    Parameters
    ----------
    frame : DataFrame
        Candidate rows with scores and weak labels (``weak_target``: 1 positive,
        0 negative, -1 unlabeled).
    threshold : float
        Score threshold for candidate-level binary predictions.
    consensus_only : bool
        If True, restrict observation-level metrics to B/C-consensus labeled
        observations (excluding B-extra supplementary labels). Candidate-level
        metrics still use all labeled candidates.
    """
    frame = frame.sort_values(["trip_id", "observation_index", "candidate_rank"],
                              kind="mergesort")
    covered = frame[frame.weak_target >= 0]
    if consensus_only:
        covered = covered[covered.consensus_eligible].copy()
    obs_index = covered[["trip_id", "observation_index"]].drop_duplicates()
    obs_count = len(obs_index)

    y_true = covered.weak_target.to_numpy(dtype=np.int8)
    y_score = covered.candidate_score.to_numpy(dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int8)
    cand = candidate_metrics(y_true, y_score, y_pred)

    # Observation level: one top candidate per observation.
    best_arc = top1_arc(frame)
    weak = covered.set_index(["trip_id", "observation_index"]).weak_arc_id
    all_keys = list(zip(obs_index.trip_id, obs_index.observation_index))
    top_arc_map = dict(zip(best_arc.index, best_arc.to_numpy()))
    weak_map = dict(zip(weak.index, weak.to_numpy()))
    obs_agreement = np.array([
        key in top_arc_map and top_arc_map[key] == weak_map[key]
        for key in all_keys])
    pos_dict = dict(zip(zip(covered.trip_id, covered.observation_index),
                        covered.positive_in_candidates.to_numpy(dtype=bool)))
    obs_miss = np.array([not bool(pos_dict.get(key, False)) for key in all_keys])
    # Reciprocal MRR, keyed by observation (fixes the earlier row-based zip
    # truncation that dropped most observations and returned rank position
    # instead of 1/rank).
    rr = reciprocal_ranks(covered)
    obs_mrr = np.array([float(rr.get(key, 0.0)) for key in all_keys])
    obs_scores = observation_scores(frame)
    obs_score_map = dict(zip(obs_scores.index, obs_scores.to_numpy()))
    obs_selected_score = np.array([obs_score_map.get(key, np.nan) for key in all_keys])

    obs = dict(
        observation_count=int(obs_count),
        candidate_miss_count=int(obs_miss.sum()),
        candidate_miss_rate=float(obs_miss.mean()) if obs_count else 0.0,
        top1_agreement=float(obs_agreement.mean()) if obs_count else 0.0,
        top1_agreement_covered_only=float(
            obs_agreement[~obs_miss].mean()) if (~obs_miss).any() else 0.0,
        mrr=float(obs_mrr.mean()) if obs_count else 0.0,
        selected_score_median=float(np.nanmedian(obs_selected_score))
        if np.isfinite(obs_selected_score).any() else np.nan,
    )
    return obs, cand


def trip_macro(frame):
    rows = []
    for trip_id, group in frame.groupby("trip_id", sort=True):
        o, c = evaluate_frame(group)
        rows.append({"trip_id": trip_id, "observations": o["observation_count"],
                     "top1_agreement": o["top1_agreement"], "mrr": o["mrr"],
                     "candidate_macro_f1": c["candidate_macro_f1"],
                     "candidate_log_loss": c["candidate_log_loss"],
                     "candidate_accuracy": c["candidate_accuracy"],
                     "candidate_recall_class1": c["candidate_recall_class1"],
                     "candidate_miss_rate": o["candidate_miss_rate"]})
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in df.columns if k != "trip_id"}, df


def route_continuity(frame, arcs_lookup):
    """Direct endpoint adjacency of Top-1 arc predictions per observation pair.

    arcs_lookup: dict trip_id -> directed arcs array. Not a weak-label input;
    used only to report prediction-route coherence diagnostics.
    """
    rows = []
    for trip_id, group in frame.groupby("trip_id", sort=True):
        arcs = arcs_lookup.get(trip_id)
        if arcs is None or len(arcs) == 0:
            continue
        best = top1_arc(group).to_numpy()
        if len(best) < 2:
            continue
        same = best[:-1] == best[1:]
        adjacent = arcs[best[:-1], 1] == arcs[best[1:], 0]
        rows.append(dict(
            trip_id=trip_id, observations=len(best),
            top1_direct_adjacency=float(np.mean(same | adjacent)),
            top1_same_arc=float(np.mean(same)),
            top1_endpoint_adjacent=float(np.mean(adjacent)),
        ))
    return pd.DataFrame(rows)


def metrics_summary(obs, cand, prefix=""):
    out = {f"{prefix}{k}": v for k, v in obs.items()}
    out.update({f"{prefix}{k}": v for k, v in cand.items()})
    return out
