"""Run with PYTHONPATH=src ~/ml312/bin/python -m pathsnap.explore."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer, CRS
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .data import load_trip, project
from .baseline import nearest_road, adjacency_rate

ROOT = Path(__file__).resolve().parents[2]
GEOD = Geod(ellps="WGS84")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    import torch
    packages = dict(sorted(
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    ))
    save_json(ROOT / "experiments/environment.json", {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": "PYTHONPATH=src ~/ml312/bin/python -m pathsnap.explore",
    })
    (ROOT / "experiments/environment.freeze.txt").write_text(
        "".join(f"{k}=={v}\n" for k, v in packages.items())
    )
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    (ROOT / "experiments/nvidia-smi.txt").write_text(result.stdout + result.stderr)


def analyze(directory, split):
    started = time.perf_counter()
    trip = load_trip(directory)
    t, n, a, r = trip.track, trip.nodes, trip.arcs, trip.route
    dt = np.diff(t[:, 2])
    bearing, _, step = GEOD.inv(t[:-1, 0], t[:-1, 1], t[1:, 0], t[1:, 1])
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=dt > 0)
    gps_xy, nodes_xy = project(t, n)
    graph = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])),
                       shape=(len(n), len(n))).tocsr()
    weak_count, weak_labels = connected_components(graph, directed=True, connection="weak")
    strong_count, _ = connected_components(graph, directed=True, connection="strong")
    degree = np.bincount(a.ravel(), minlength=len(n))
    lengths = np.linalg.norm(nodes_xy[a[:, 1]] - nodes_xy[a[:, 0]], axis=1)
    route_breaks = a[r[:-1], 1] != a[r[1:], 0]
    route_lengths = lengths[r]
    baseline_start = time.perf_counter()
    pred, distance, ties, snapped = nearest_road(gps_xy, nodes_xy, a)
    baseline_seconds = time.perf_counter() - baseline_start
    folder = ROOT / "data/processed" / split
    folder.mkdir(parents=True, exist_ok=True)
    features = pd.DataFrame({
        "observation_index": np.arange(len(t)), "longitude": t[:, 0], "latitude": t[:, 1],
        "timestamp_s": t[:, 2], "dt_s": np.r_[np.nan, dt],
        "step_m": np.r_[np.nan, step], "speed_mps": np.r_[np.nan, speed],
        "heading_deg": np.r_[np.nan, (bearing + 360) % 360],
    })
    features.to_csv(folder / f"{directory.name}_observations.csv", index=False)
    pd.DataFrame({
        "observation_index": np.arange(len(t)), "arc_id": pred,
        "distance_m": distance, "equally_near_arcs": ties,
        "snapped_x_m": snapped[:, 0], "snapped_y_m": snapped[:, 1],
    }).to_csv(folder / f"{directory.name}_baseline.csv", index=False)
    hashes = {}
    inventory = []
    for extension in ["track", "nodes", "arcs", "route"]:
        f = directory / f"{directory.name}.{extension}"
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        hashes[extension] = digest
        inventory.append(dict(split=split, trip_id=directory.name, file=str(f.relative_to(ROOT)),
                              bytes=f.stat().st_size, sha256=digest,
                              rows=len(getattr(trip, extension))))
    stats = dict(
        split=split, trip_id=directory.name, gps_count=len(t), node_count=len(n),
        arc_count=len(a), route_count=len(r), missing_values=0, invalid_values=0,
        lon_min=float(t[:, 0].min()), lon_max=float(t[:, 0].max()),
        lat_min=float(t[:, 1].min()), lat_max=float(t[:, 1].max()),
        nodes_lon_min=float(n[:, 0].min()), nodes_lon_max=float(n[:, 0].max()),
        nodes_lat_min=float(n[:, 1].min()), nodes_lat_max=float(n[:, 1].max()),
        timestamp_start_s=float(t[0, 2]), timestamp_end_s=float(t[-1, 2]),
        duration_s=float(t[-1, 2] - t[0, 2]), dt_median_s=float(np.median(dt)),
        max_gap_s=float(dt.max()), dt_steps=len(dt), steps_1s=int((dt == 1).sum()),
        nonpositive_dt=int((dt <= 0).sum()), gaps_over_5s=int((dt > 5).sum()),
        repeated_gps_positions=int((step == 0).sum()),
        duplicate_track_rows=int(len(t) - len(np.unique(t, axis=0))),
        gps_path_m=float(step.sum()), max_speed_mps=float(np.nanmax(speed)),
        speed_over_55mps=int((speed > 55).sum()),
        weak_components=int(weak_count), strong_components=int(strong_count),
        largest_weak_fraction=float(np.bincount(weak_labels).max() / len(n)),
        isolated_nodes=int((degree == 0).sum()), self_loops=int((a[:, 0] == a[:, 1]).sum()),
        duplicate_arcs=int(len(a) - len(np.unique(a, axis=0))),
        duplicate_node_coordinates=int(len(n) - len(np.unique(n, axis=0))),
        zero_length_arcs=int((lengths == 0).sum()), route_breaks=int(route_breaks.sum()),
        route_repeated_arc_occurrences=int(len(r) - len(np.unique(r))),
        route_length_m=float(route_lengths.sum()),
        route_rows_equal_gps=bool(len(r) == len(t)),
        nearest_distance_median_m=float(np.median(distance)),
        nearest_distance_p95_m=float(np.quantile(distance, .95)),
        nearest_distance_max_m=float(distance.max()),
        nearest_over_100m=int((distance > 100).sum()),
        geometric_tie_observations=int((ties > 1).sum()),
        baseline_direct_adjacency_rate=adjacency_rate(a, pred),
        baseline_reference_membership=float(np.isin(pred, r).mean()),
        baseline_seconds=baseline_seconds,
        processing_seconds=time.perf_counter() - started,
        track_sha256=hashes["track"],
    )
    return stats, inventory, dt, speed, distance


def visualize(directory, split, reason):
    trip = load_trip(directory)
    t, n, a, r = trip.track, trip.nodes, trip.arcs, trip.route
    pred = pd.read_csv(ROOT / "data/processed" / split /
                       f"{directory.name}_baseline.csv").arc_id.to_numpy()
    # Display only arcs whose bounding boxes intersect the GPS/reference extent.
    relevant = np.vstack([t[:, :2], n[a[r].ravel()]])
    low, high = relevant.min(axis=0) - .002, relevant.max(axis=0) + .002
    segments = n[a]
    mask = ((segments.max(axis=1) >= low) & (segments.min(axis=1) <= high)).all(axis=1)
    roads = segments[mask]
    path = ROOT / "visualizations" / f"{split}_{directory.name}"
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.add_collection(LineCollection(roads, colors="#bbbbbb", linewidths=.4, label="Road network"))
    ax.add_collection(LineCollection(n[a[r]], colors="green", linewidths=2, label="Reference arcs"))
    ax.add_collection(LineCollection(n[a[np.unique(pred)]], colors="darkorange", linewidths=.8,
                                     label="Nearest-road arcs (unordered union)"))
    ax.scatter(t[:, 0], t[:, 1], s=3, color="blue", label="GPS observations", zorder=4)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
           xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
           title=f"{split} {directory.name}: {reason}")
    ax.set_aspect(1 / np.cos(np.deg2rad(t[:, 1].mean())))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    m = folium.Map(location=[t[:, 1].mean(), t[:, 0].mean()], tiles=None,
                   prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="OSM basemap (internet required)").add_to(m)
    for name, coords, color, weight in [
        ("Road network (extent clipped)", roads, "#999999", 1),
        ("Reference route arcs", n[a[r]], "green", 6),
        ("Geometric baseline arcs (not reconstructed route)",
         n[a[np.unique(pred)]], "darkorange", 2),
    ]:
        group = folium.FeatureGroup(name=name)
        folium.PolyLine(coords[:, :, ::-1].tolist(), color=color, weight=weight,
                        opacity=.7, tooltip=name).add_to(group)
        group.add_to(m)
    group = folium.FeatureGroup(name="GPS observations")
    for i, (lon, lat, timestamp) in enumerate(t):
        folium.CircleMarker([lat, lon], radius=1, weight=.5, opacity=.6,
                            color="blue", fill=True, fill_opacity=.45,
                            tooltip=f"Observation {i}; time {timestamp:g}s").add_to(group)
    group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([low[::-1].tolist(), high[::-1].tolist()])
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;'
        'background:white;padding:10px;font:14px sans-serif">'
        f'{split} {directory.name} — {reason}<br>'
        '<span style="color:blue">Blue: GPS</span> | '
        '<span style="color:green">Green: reference</span> | '
        '<span style="color:darkorange">Orange: geometric baseline</span><br>'
        'Grey: roads. Baseline arc union is not a continuous reconstructed route.</div>'
    ))
    m.save(str(path.with_suffix(".html")))
    return dict(split=split, trip_id=directory.name, reason=reason,
                displayed_road_arcs=int(mask.sum()), total_road_arcs=len(a))


def main():
    environment()
    start = time.perf_counter()
    rows, files, errors, distributions = [], [], [], {}
    raw = ROOT / "data/raw/Student folder"
    for split in ["train", "student_test"]:
        all_dt, all_speed, all_distance = [], [], []
        for directory in sorted((raw / split).iterdir()):
            if not directory.is_dir():
                continue
            try:
                stats, inventory, dt, speed, distance = analyze(directory, split)
            except Exception as exc:
                errors.append(dict(split=split, trip_id=directory.name, error=str(exc)))
                continue
            rows.append(stats)
            files.extend(inventory)
            all_dt.extend(dt)
            all_speed.extend(speed[np.isfinite(speed)])
            all_distance.extend(distance)
            print(f"{split}/{directory.name}: {stats['gps_count']} GPS; "
                  f"{stats['route_breaks']} reference breaks", flush=True)
        distributions[split] = (np.array(all_dt), np.array(all_speed), np.array(all_distance))
    save_json(ROOT / "experiments/parsing_errors.json", errors)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "experiments/trip_statistics.csv", index=False)
    pd.DataFrame(files).to_csv(ROOT / "experiments/file_inventory.csv", index=False)
    summary = {}
    for split, group in frame.groupby("split"):
        dt, speed, distance = distributions[split]
        summary[split] = {
            "trips": len(group),
            **{col: int(group[col].sum()) for col in [
                "gps_count", "node_count", "arc_count", "route_count", "missing_values",
                "invalid_values", "nonpositive_dt", "gaps_over_5s", "route_breaks",
                "isolated_nodes", "zero_length_arcs", "geometric_tie_observations",
                "nearest_over_100m", "speed_over_55mps"]},
            "gps_per_trip_min": int(group.gps_count.min()),
            "gps_per_trip_median": float(group.gps_count.median()),
            "gps_per_trip_max": int(group.gps_count.max()),
            "sampling_1s_fraction": float((dt == 1).mean()),
            "max_gap_s": float(dt.max()),
            "trips_with_reference_breaks": int((group.route_breaks > 0).sum()),
            "trips_with_multiple_weak_components": int((group.weak_components > 1).sum()),
            "nearest_distance_median_m": float(np.median(distance)),
            "nearest_distance_p95_m": float(np.quantile(distance, .95)),
            "baseline_direct_adjacency_macro": float(group.baseline_direct_adjacency_rate.mean()),
            "baseline_direct_adjacency_micro": float(np.average(
                group.baseline_direct_adjacency_rate, weights=group.gps_count - 1)),
            "baseline_reference_membership_micro": float(np.average(
                group.baseline_reference_membership, weights=group.gps_count)),
            "baseline_total_seconds": float(group.baseline_seconds.sum()),
        }
    summary["validation"] = {
        "parsing_errors": len(errors),
        "overlapping_trip_ids": sorted(set(frame[frame.split == "train"].trip_id) &
                                      set(frame[frame.split == "student_test"].trip_id)),
        "duplicate_track_hashes": frame[frame.track_sha256.duplicated(False)][
            ["split", "trip_id", "track_sha256"]].to_dict("records"),
        "evaluator_data_accessed": False,
    }
    save_json(ROOT / "experiments/summary.json", summary)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for split, group in frame.groupby("split"):
        dt, speed, distance = distributions[split]
        axes[0, 0].hist(group.gps_count, bins=20, alpha=.5, label=split)
        axes[0, 1].hist(dt, bins=np.geomspace(.5, max(2, dt.max() + 1), 30), alpha=.5, label=split)
        axes[0, 2].scatter(group.node_count, group.arc_count, alpha=.6, label=split)
        axes[1, 0].hist(speed, bins=40, alpha=.5, label=split)
        axes[1, 1].hist(distance, bins=50, alpha=.5, label=split)
        axes[1, 2].scatter(group.route_count, group.gps_count, alpha=.6, label=split)
    for ax, title, xlabel in zip(axes.ravel(),
            ["Trip observations", "Sampling intervals", "Network size",
             "Observed displacement / dt", "Geometric baseline distance",
             "Reference sequence versus GPS count"],
            ["GPS count", "seconds (log scale)", "Nodes", "m/s", "metres", "Reference arcs"]):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend()
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 2].set_ylabel("Arcs")
    axes[1, 2].set_ylabel("GPS observations")
    fig.tight_layout()
    fig.savefig(ROOT / "visualizations/dataset_overview.png", dpi=150)
    plt.close(fig)
    training = frame[frame.split == "train"]
    selections = [
        (training.loc[(training.gps_count - training.gps_count.median()).abs().idxmin()], "Near-median GPS count"),
        (training.loc[training.max_gap_s.idxmax()], "Largest training timestamp gap"),
        (training.loc[training.route_breaks.idxmax()], "Most reference adjacency breaks"),
    ]
    maps, seen = [], set()
    for row, reason in selections:
        if row.trip_id in seen:
            continue
        seen.add(row.trip_id)
        maps.append(visualize(raw / "train" / row.trip_id, "train", reason))
    save_json(ROOT / "experiments/representative_maps.json", maps)
    save_json(ROOT / "experiments/run.json", {
        "elapsed_seconds": time.perf_counter() - start, "completed_trips": len(frame),
        "failed_trips": len(errors), "models_trained": 0,
    })
    if errors:
        raise SystemExit("Parsing failures: inspect experiments/parsing_errors.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
