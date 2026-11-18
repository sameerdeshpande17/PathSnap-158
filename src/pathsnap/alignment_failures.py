"""Training-only diagnosis of severe C residuals; does not tune frozen methods."""
import json

import numpy as np
import pandas as pd

from .alignment import geometry
from .alignment_run import OUT, RAW, LABELS
from .data import load_trip
from .explore import ROOT


def main():
    cfg = json.loads((OUT / "frozen_methods.json").read_text())["config"]
    rows = []
    for path in sorted((LABELS / "train").glob("*_C.csv")):
        trip_id = path.name.split("_")[0]
        c = pd.read_csv(path)
        b = pd.read_csv(path.with_name(f"{trip_id}_B.csv"))
        severe = (c.projection_distance_m > 1000) & (b.projection_distance_m < 100)
        if not severe.any():
            continue
        trip = load_trip(RAW / "train" / trip_id)
        g = geometry(trip)
        terminal = int(c.assigned_occurrence_index.iloc[-1])
        suffix_start = int(np.flatnonzero(c.assigned_occurrence_index == terminal)[0])
        next_occ = terminal + 1
        following_admissible = 0
        following_nearest = 0
        if next_occ < len(trip.route):
            indices = np.arange(max(suffix_start, 1), len(c))
            delta = g.progress[indices, next_occ] - g.progress[indices - 1, terminal]
            following_admissible = int((delta <= cfg["max_speed_mps"] * g.dt[indices] +
                                         cfg["speed_slack_m"]).sum())
            # Rank by stable occurrence ordering to match candidate selection.
            near = np.argsort(g.distances[indices], axis=1, kind="stable")[:, :cfg["candidate_count"]]
            following_nearest = int((near == next_occ).any(axis=1).sum())
        rows.append(dict(
            split="train", trip_id=trip_id,
            observations=len(c),
            observations_C_over_1000m_B_under_100m=int(severe.sum()),
            first_severe_observation=int(np.flatnonzero(severe)[0]),
            terminal_C_occurrence=terminal,
            terminal_B_occurrence=int(b.assigned_occurrence_index.iloc[-1]),
            total_reference_occurrences=len(trip.route),
            constant_C_suffix_start=suffix_start,
            constant_C_suffix_observations=len(c) - suffix_start,
            C_residual_p95_m=float(c.projection_distance_m.quantile(.95)),
            B_residual_p95_m=float(b.projection_distance_m.quantile(.95)),
            subsequent_occurrence_speed_admissible_steps=following_admissible,
            subsequent_occurrence_in_nearest_candidates_steps=following_nearest,
            severe_observations_selected_for_supervision=int(
                (severe & c.recommended_supervision_use).sum()),
        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "training_failure_cases.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
