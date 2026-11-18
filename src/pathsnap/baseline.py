"""Geometry only: no fitting, reference route, or learned parameters."""
import numpy as np
import shapely
from shapely.strtree import STRtree


def nearest_road(gps_xy, nodes_xy, arcs):
    """Exact projected point-to-segment query; lowest arc ID breaks ties.

    Opposite directed arcs often share geometry. This method cannot resolve
    direction and deliberately exposes the number of equally near arcs.
    """
    lines = shapely.linestrings(nodes_xy[arcs])
    pairs, distances = STRtree(lines).query_nearest(
        shapely.points(gps_xy), all_matches=True, return_distance=True
    )
    order = np.lexsort((pairs[1], pairs[0]))
    obs, arc = pairs[:, order]
    first = np.r_[True, np.diff(obs) != 0]
    selected = arc[first]
    distance = distances[order][first]
    ties = np.bincount(obs, minlength=len(gps_xy))
    snapped = shapely.line_interpolate_point(
        lines[selected],
        shapely.line_locate_point(lines[selected], shapely.points(gps_xy)),
    )
    return selected, distance, ties, shapely.get_coordinates(snapped)


def adjacency_rate(arcs, predictions):
    """Same arc or directly endpoint-adjacent; not full path reachability."""
    if len(predictions) < 2:
        return None
    a, b = predictions[:-1], predictions[1:]
    return float(np.mean((a == b) | (arcs[a, 1] == arcs[b, 0])))
