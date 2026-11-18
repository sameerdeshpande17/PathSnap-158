"""Route reconstruction from Viterbi-decoded candidate sequences.

Converts a sequence of decoded road-arc IDs into a connected, directed route
by filling gaps between consecutive decoded arcs with directed shortest paths.

Preserves original arc IDs and traversal order. Distinguishes consecutive
observations on the same arc (dwelling) from legitimate repeated traversal
of the same road at different times. Handles disconnected road-network
components explicitly instead of inventing connecting roads.

No reference-route (.route) data is used during reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pathsnap.hmm import MapMatchingHMM


@dataclass
class RouteSegment:
    """One segment of the reconstructed route."""

    arc_id: int
    from_node: int
    to_node: int
    length_m: float
    # Why this arc is in the route
    kind: str  # 'decoded', 'connector', 'dwell'
    # GPS observations matched to this arc
    observation_indices: list[int] = field(default_factory=list)
    # For connectors: the decoded arcs being connected
    connects_from: int | None = None
    connects_to: int | None = None


@dataclass
class ReconstructedRoute:
    """Complete reconstructed route for one trip."""

    trip_id: str
    n_observations: int
    n_decoded_arcs: int
    n_route_segments: int
    n_connector_segments: int
    total_length_m: float
    segments: list[RouteSegment] = field(default_factory=list)
    # Arc ID sequence (traversal order, includes connectors)
    arc_id_sequence: list[int] = field(default_factory=list)
    # Diagnostics
    n_disconnected_gaps: int = 0
    n_repeated_arc_traversals: int = 0  # same arc ID appearing non-consecutively
    n_dwell_transitions: int = 0  # consecutive observations on same arc
    # Per-observation mapping
    observation_to_arc: list[dict[str, Any]] = field(default_factory=list)


class RouteReconstructor:
    """Builds a connected route from Viterbi-decoded arc assignments."""

    def __init__(self, hmm: MapMatchingHMM):
        self.hmm = hmm

    def reconstruct(
        self,
        trip_id: str,
        viterbi_result,
        observation_details: list[dict[str, Any]],
        arc_lookup: dict[int, tuple[int, int]],
        node_coords: dict[int, tuple[float, float]],
    ) -> ReconstructedRoute:
        """Reconstruct a connected route from decoded Viterbi output.

        arc_lookup: arc_id -> (from_node, to_node)
        node_coords: node_id -> (lon, lat)
        """
        route = ReconstructedRoute(
            trip_id=trip_id,
            n_observations=viterbi_result.n_observations,
            n_decoded_arcs=len(viterbi_result.decoded_arc_ids),
        )

        decoded_arcs = viterbi_result.decoded_arc_ids
        if len(decoded_arcs) == 0:
            return route

        # Build observation -> arc mapping (decoded)
        obs_to_arc = {}
        for i, detail in enumerate(observation_details):
            obs_to_arc[i] = detail["decoded_arc_id"]

        # ---- Step 1: Collapse consecutive decoded arcs into runs ----
        # A "run" is a maximal sequence of consecutive observations on the same arc.
        runs = []  # list of (arc_id, start_obs, end_obs)
        current_arc = decoded_arcs[0]
        start_obs = 0
        for t in range(1, len(decoded_arcs)):
            if decoded_arcs[t] != current_arc:
                runs.append((current_arc, start_obs, t - 1))
                route.n_dwell_transitions += t - 1 - start_obs  # dwells within run
                current_arc = decoded_arcs[t]
                start_obs = t
        runs.append((current_arc, start_obs, len(decoded_arcs) - 1))
        route.n_dwell_transitions += len(decoded_arcs) - 1 - start_obs

        # ---- Step 2: Connect consecutive runs with shortest paths ----
        route_segments = []
        arc_sequence = []

        for run_idx, (arc_id, obs_start, obs_end) in enumerate(runs):
            from_node, to_node = arc_lookup[arc_id]
            length_m = self._segment_length_m(arc_id, arc_lookup, node_coords)

            seg = RouteSegment(
                arc_id=arc_id,
                from_node=from_node,
                to_node=to_node,
                length_m=length_m,
                kind="decoded",
                observation_indices=list(range(obs_start, obs_end + 1)),
            )
            route_segments.append(seg)
            arc_sequence.append(arc_id)

            # If not the last run, find connector to next run's arc
            if run_idx < len(runs) - 1:
                next_arc_id = runs[run_idx + 1][0]
                next_from_node = arc_lookup[next_arc_id][0]

                connector_path = self.hmm._bounded_directed_path(
                    to_node, next_from_node, max_edges=100
                )

                if connector_path is None:
                    # Disconnected gap - record explicitly
                    route.n_disconnected_gaps += 1
                    gap_seg = RouteSegment(
                        arc_id=-1,
                        from_node=to_node,
                        to_node=next_from_node,
                        length_m=0.0,
                        kind="disconnected_gap",
                        observation_indices=[],
                        connects_from=arc_id,
                        connects_to=next_arc_id,
                    )
                    route_segments.append(gap_seg)
                    arc_sequence.append(-1)  # sentinel for gap
                else:
                    # Add connector arcs
                    conn_length = 0.0
                    for conn_arc in connector_path:
                        c_from, c_to = arc_lookup[conn_arc]
                        c_len = self._segment_length_m(conn_arc, arc_lookup, node_coords)
                        conn_length += c_len
                        conn_seg = RouteSegment(
                            arc_id=conn_arc,
                            from_node=c_from,
                            to_node=c_to,
                            length_m=c_len,
                            kind="connector",
                            observation_indices=[],
                            connects_from=arc_id,
                            connects_to=next_arc_id,
                        )
                        route_segments.append(conn_seg)
                        arc_sequence.append(conn_arc)
                    route.n_connector_segments += 1

        route.segments = route_segments
        route.arc_id_sequence = arc_sequence
        route.n_route_segments = len(route_segments)

        # ---- Step 3: Detect repeated arc traversals ----
        # Same arc ID appearing in non-consecutive runs = repeated traversal
        arc_run_indices = {}
        for run_idx, (arc_id, _, _) in enumerate(runs):
            arc_run_indices.setdefault(arc_id, []).append(run_idx)
        route.n_repeated_arc_traversals = sum(
            1 for indices in arc_run_indices.values() if len(indices) > 1
        )

        # ---- Step 4: Total length ----
        route.total_length_m = sum(s.length_m for s in route_segments if s.length_m > 0)

        # ---- Step 5: Observation-to-arc mapping ----
        route.observation_to_arc = [
            dict(obs_index=i, arc_id=obs_to_arc[i]) for i in range(viterbi_result.n_observations)
        ]

        return route

    def _segment_length_m(
        self, arc_id: int, arc_lookup: dict[int, tuple[int, int]],
        node_coords: dict[int, tuple[float, float]],
    ) -> float:
        from_node, to_node = arc_lookup[arc_id]
        lon1, lat1 = node_coords[from_node]
        lon2, lat2 = node_coords[to_node]
        return self.hmm._haversine_m(lat1, lon1, lat2, lon2)
