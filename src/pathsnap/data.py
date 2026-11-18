from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer


@dataclass
class Trip:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray


def read_table(path, columns, integer=False):
    """Strict headerless parser: reject ragged, nonfinite and fractional IDs."""
    rows = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            values = line.split()
            if len(values) != columns:
                raise ValueError(f"{path}:{number}: expected {columns} columns")
            row = [float(v) for v in values]
            if not np.isfinite(row).all():
                raise ValueError(f"{path}:{number}: nonfinite value")
            if integer and any(v != int(v) for v in row):
                raise ValueError(f"{path}:{number}: fractional ID")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty table")
    return np.asarray(rows, dtype=np.int64 if integer else np.float64)


def load_trip(directory):
    directory = Path(directory)
    stem = directory / directory.name
    trip = Trip(
        read_table(stem.with_suffix(".track"), 3),
        read_table(stem.with_suffix(".nodes"), 2),
        read_table(stem.with_suffix(".arcs"), 2, True),
        read_table(stem.with_suffix(".route"), 1, True).ravel(),
    )
    for name, coordinates in [("track", trip.track[:, :2]), ("nodes", trip.nodes)]:
        if ((np.abs(coordinates[:, 0]) > 180) |
                (np.abs(coordinates[:, 1]) > 90)).any():
            raise ValueError(f"{directory}: invalid {name} longitude/latitude")
    if (trip.arcs < 0).any() or (trip.arcs >= len(trip.nodes)).any():
        raise ValueError(f"{directory}: out-of-range node ID")
    if (trip.route < 0).any() or (trip.route >= len(trip.arcs)).any():
        raise ValueError(f"{directory}: out-of-range route arc ID")
    return trip


def project(track, nodes):
    """Local WGS84 azimuthal equidistant metres, always longitude first."""
    lon, lat = np.mean(track[:, :2], axis=0)
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    )
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    gps_xy = np.column_stack(transform.transform(track[:, 0], track[:, 1]))
    nodes_xy = np.column_stack(transform.transform(nodes[:, 0], nodes[:, 1]))
    return gps_xy, nodes_xy
