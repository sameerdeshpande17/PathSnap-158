"""Investigate each broken reference boundary without editing any route."""
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .data import load_trip, project
from .explore import ROOT, save_json


def directed_graph(nodes_xy, arcs):
    unique, inverse = np.unique(arcs, axis=0, return_inverse=True)
    lengths = np.linalg.norm(nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]], axis=1)
    weights = np.full(len(unique), np.inf)
    np.minimum.at(weights, inverse, lengths)
    # Explicit zero weights are preserved by sparse csgraph.
    return csr_matrix((weights, (unique[:, 0], unique[:, 1])),
                      shape=(len(nodes_xy), len(nodes_xy)))


def connection(graph, source, target):
    distances, predecessors = dijkstra(graph, directed=True, indices=source,
                                       return_predecessors=True)
    if not np.isfinite(distances[target]):
        return None, []
    path = [int(target)]
    while path[-1] != source:
        parent = int(predecessors[path[-1]])
        if parent < 0 or len(path) > graph.shape[0]:
            raise RuntimeError("Invalid shortest-path predecessor chain")
        path.append(parent)
    return float(distances[target]), path[::-1]


def main():
    rows, examples = [], []
    geod = Geod(ellps="WGS84")
    for split in ["train", "student_test"]:
        for directory in sorted((ROOT / "data/raw/Student folder" / split).iterdir()):
            trip = load_trip(directory)
            a, r, n = trip.arcs, trip.route, trip.nodes
            repeated = np.flatnonzero(pd.Series(r).duplicated().to_numpy())
            if len(repeated):
                j = int(repeated[0])
                examples.append(dict(split=split, trip_id=directory.name, arc_id=int(r[j]),
                                     occurrences=np.flatnonzero(r == r[j]).tolist()))
            indices = np.flatnonzero(a[r[:-1], 1] != a[r[1:], 0])
            if not len(indices):
                continue
            _, xy = project(trip.track, n)
            graph = directed_graph(xy, a)
            _, components = connected_components(graph, directed=True, connection="weak")
            for j in indices:
                first, second = int(r[j]), int(r[j + 1])
                u, v = int(a[first, 1]), int(a[second, 0])
                _, _, separation = geod.inv(*n[u], *n[v])
                length, path = connection(graph, u, v)
                reverse_length, reverse_path = connection(graph, v, u)
                nearest = []
                for endpoint in [u, v]:
                    _, _, gps_distance = geod.inv(
                        np.full(len(trip.track), n[endpoint, 0]),
                        np.full(len(trip.track), n[endpoint, 1]),
                        trip.track[:, 0], trip.track[:, 1])
                    observation = int(np.argmin(gps_distance))
                    nearest.append((observation, float(gps_distance[observation]),
                                    float(trip.track[observation, 2])))
                if components[u] != components[v]:
                    finding = "different_weak_components"
                elif length is None:
                    finding = "same_weak_component_but_no_directed_path"
                elif separation < .01:
                    finding = "distinct_node_ids_same_coordinates_with_directed_path"
                else:
                    finding = "directed_connecting_path_exists_but_is_omitted_from_reference"
                rows.append(dict(
                    split=split, trip_id=directory.name,
                    occurrence_before=int(j), occurrence_after=int(j + 1),
                    arc_before=first, arc_after=second,
                    before_from=int(a[first, 0]), before_to=u,
                    after_from=v, after_to=int(a[second, 1]),
                    before_endpoint_lon=n[u, 0], before_endpoint_lat=n[u, 1],
                    after_endpoint_lon=n[v, 0], after_endpoint_lat=n[v, 1],
                    separation_m=float(separation),
                    same_weak_component=bool(components[u] == components[v]),
                    directed_path_exists=length is not None,
                    directed_path_m=length, directed_path_edges=max(0, len(path) - 1),
                    path_nodes=" ".join(map(str, path)),
                    reverse_path_exists=reverse_length is not None,
                    reverse_path_m=reverse_length,
                    nearest_before_observation=nearest[0][0],
                    nearest_before_gps_distance_m=nearest[0][1],
                    nearest_before_timestamp_s=nearest[0][2],
                    nearest_after_observation=nearest[1][0],
                    nearest_after_gps_distance_m=nearest[1][1],
                    nearest_after_timestamp_s=nearest[1][2],
                    nearest_endpoint_observation_time_difference_s=nearest[1][2] - nearest[0][2],
                    finding=finding,
                ))
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "experiments/reference_adjacency_breaks.csv", index=False)
    save_json(ROOT / "experiments/alignment/repeated_arc_examples.json", examples)
    assert len(frame) == 14
    print(frame[["split", "trip_id", "occurrence_before", "separation_m", "finding"]].to_string(index=False))


if __name__ == "__main__":
    main()
