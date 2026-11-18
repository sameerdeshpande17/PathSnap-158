"""Additional original-archive, projection and anomaly-location verification."""
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
from pyproj import Geod

from .data import load_trip, project
from .explore import ROOT, save_json


def main():
    stats = pd.read_csv(ROOT / "experiments/trip_statistics.csv", dtype={"trip_id": str})
    inventory = pd.read_csv(ROOT / "experiments/file_inventory.csv", dtype={"trip_id": str})
    archive = ROOT / "data/original/Student folder.zip"
    verified = 0
    with zipfile.ZipFile(archive) as z:
        for row in inventory.itertuples():
            member = row.file.removeprefix("data/raw/")
            assert hashlib.sha256(z.read(member)).hexdigest() == row.sha256
            verified += 1
    geod = Geod(ellps="WGS84")
    projections, anomalies = [], []
    for row in stats.itertuples():
        trip = load_trip(ROOT / "data/raw/Student folder" / row.split / row.trip_id)
        t, n, a, r = trip.track, trip.nodes, trip.arcs, trip.route
        _, xy = project(t, n)
        _, _, geodesic = geod.inv(n[a[:, 0], 0], n[a[:, 0], 1],
                                  n[a[:, 1], 0], n[a[:, 1], 1])
        projected = np.linalg.norm(xy[a[:, 1]] - xy[a[:, 0]], axis=1)
        valid = geodesic > .01
        relative = np.abs(projected[valid] / geodesic[valid] - 1)
        projections.append(dict(split=row.split, trip_id=row.trip_id,
                                checked_arcs=int(valid.sum()),
                                max_relative_error=float(relative.max()),
                                p99_relative_error=float(np.quantile(relative, .99))))
        features = pd.read_csv(ROOT / "data/processed" / row.split /
                              f"{row.trip_id}_observations.csv")
        baseline = pd.read_csv(ROOT / "data/processed" / row.split /
                              f"{row.trip_id}_baseline.csv")
        assert len(features) == len(baseline) == row.gps_count
        assert baseline.arc_id.between(0, len(a) - 1).all()
        for kind, values, flag in [
            ("gps_gap_s", features.dt_s, features.dt_s > 5),
            ("gps_speed_mps", features.speed_mps, features.speed_mps > 55),
            ("nearest_distance_m", baseline.distance_m, baseline.distance_m > 100),
        ]:
            for i in np.flatnonzero(flag):
                anomalies.append(dict(split=row.split, trip_id=row.trip_id,
                                      kind=kind, row_index=int(i), value=float(values.iloc[i])))
        for i in np.flatnonzero(a[r[:-1], 1] != a[r[1:], 0]):
            anomalies.append(dict(split=row.split, trip_id=row.trip_id,
                                  kind="reference_adjacency_break_after_route_row",
                                  row_index=int(i), value=int(r[i])))
    save_json(ROOT / "experiments/projection_validation.json", projections)
    pd.DataFrame(anomalies).to_csv(ROOT / "experiments/anomaly_locations.csv", index=False)
    assert stats.groupby("split").size().to_dict() == {"student_test": 10, "train": 80}
    assert verified == 360
    save_json(ROOT / "experiments/integration_checks.json", {
        "archive_members_hash_verified": verified,
        "trip_output_row_counts_verified": len(stats),
        "baseline_arc_bounds_verified": len(stats),
        "expected_split_counts_verified": True,
        "max_projection_relative_arc_length_error": max(p["max_relative_error"] for p in projections),
    })
    lines = [
        "# Measured data-quality findings", "",
        "All counts are computed from original supplied files; node/arc totals count",
        "trip-local records, not globally unique physical roads.", "",
    ]
    for split, group in stats.groupby("split"):
        lines.extend([
            f"## {split}", "",
            f"- {len(group)} trips; {group.gps_count.sum():,} GPS observations.",
            f"- Zero missing/nonfinite values and invalid IDs in successfully parsed files.",
            f"- {group.nonpositive_dt.sum()} nonpositive time deltas; "
            f"{group.gaps_over_5s.sum()} gaps >5s; maximum {group.max_gap_s.max():g}s.",
            f"- {group.route_breaks.sum()} reference adjacency breaks in "
            f"{(group.route_breaks > 0).sum()} trips.",
            f"- {group.zero_length_arcs.sum()} zero-length arcs; "
            f"{group.duplicate_arcs.sum()} duplicate directed arc rows; "
            f"{group.self_loops.sum()} self-loops.",
            f"- {group.duplicate_node_coordinates.sum()} duplicate node-coordinate rows; "
            f"{group.repeated_gps_positions.sum()} stationary consecutive GPS pairs.",
            f"- {(group.weak_components > 1).sum()} networks have multiple weak components; "
            f"{group.isolated_nodes.sum()} isolated nodes.",
            f"- {group.speed_over_55mps.sum()} observed speeds >55m/s (flags, not confirmed errors).",
            f"- {group.nearest_over_100m.sum()} GPS-to-nearest-road distances >100m.",
            f"- {group.geometric_tie_observations.sum()} observations have tied nearest arcs.",
            f"- Longitude range {group.lon_min.min():.6f} to {group.lon_max.max():.6f}; "
            f"latitude {group.lat_min.min():.6f} to {group.lat_max.max():.6f}.", "",
        ])
    lines.extend([
        "## Interpretation", "",
        "No trip has equal route-row and GPS-row counts. Pointwise reference labels",
        "are not supplied. Connectivity breaks and direction ties require explicit",
        "handling in later alignment and decoding. Networks span diverse regions;",
        "a single fixed UTM zone would be unsuitable for the entire dataset.",
        "No duplicate track SHA-256 hashes or overlapping trip IDs were found",
        "between train and student testing; this does not establish geographic independence.",
        "See anomaly_locations.csv for zero-based observation/route row locations.",
        "Projection checks compare every nontrivial arc's projected length with its",
        "WGS84 geodesic endpoint distance; they do not prove exact geodesic nearest-road ordering.",
    ])
    (ROOT / "docs/data_quality.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
