"""Hidden Markov Model for GPS-to-road map matching.

Hidden states: candidate road segments (arc_id) for each GPS observation.
Observations: GPS points with classifier emission probabilities.

All inference uses GPS coordinates, road-network nodes/arcs, and classifier
probabilities only. Reference-route files are never read during
inference or decoding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pathsnap.candidates import generate, load_inputs
from pathsnap.data import project


@dataclass
class HMMSettings:
    """Configuration for HMM emissions and transitions."""

    # Emission: classifier probability temperature
    emission_temperature: float = 1.0
    # Floor for emission probabilities (avoid log(0))
    emission_floor: float = 1e-9
    # Floor for transition probabilities
    transition_floor: float = 1e-9
    # Maximum speed for transition plausibility (m/s)
    max_speed_mps: float = 35.0
    # Speed penalty coefficient (higher = stronger penalty for implausible speed)
    speed_penalty_coeff: float = 3.0
    # Distance decay scale for transitions (metres)
    distance_scale_m: float = 100.0
    # Heading agreement weight (0-1, higher = more weight on direction match)
    heading_weight: float = 0.3
    # Self-transition (stay on same arc) bonus
    stay_arc_bonus: float = 1.0
    # Penalty for crossing a directed connectivity break
    break_penalty: float = 5.0
    # Penalty for switching to a parallel/opposite-direction arc
    direction_mismatch_penalty: float = 2.0
    # Beam width for Viterbi pruning (None = no pruning)
    beam_width: int | None = 64
    # Max candidates per observation to consider (None = all)
    max_candidates: int | None = 32


@dataclass
class CandidateState:
    """A hidden state: one candidate road segment for one observation."""

    obs_index: int
    arc_id: int
    from_node: int
    to_node: int
    segment_length_m: float
    segment_heading_rad: float
    # Projection of GPS point onto this segment (0..1)
    projection_fraction: float
    # Distance from GPS point to segment (metres)
    distance_m: float
    # Classifier probability that this arc is the positive match
    classifier_prob: float
    # Emission probability (derived from classifier_prob)
    emission_prob: float = 0.0


@dataclass
class TransitionScore:
    """Log transition score from one state to the next."""

    from_arc: int
    to_arc: int
    log_prob: float
    is_connected: bool
    path_length_m: float | None = None
    path_edges: list[int] | None = None
    reason: str = ""


class MapMatchingHMM:
    """HMM for map matching using classifier emissions and network transitions."""

    def __init__(self, settings: HMMSettings | None = None):
        self.settings = settings or HMMSettings()
        self._arc_geometry: dict[int, tuple[float, float]] = {}  # arc_id -> (lon, lat) midpoint
        self._node_coords: dict[int, tuple[float, float]] = {}
        self._adjacency: dict[int, set[int]] = {}  # from_node -> set of to_node
        self._reverse_adjacency: dict[int, set[int]] = {}  # to_node -> set of from_node
        self._directed_arcs: dict[int, tuple[int, int]] = {}  # arc_id -> (from_node, to_node)
        self._connected_components: dict[int, int] = {}  # node -> component id
        self._component_arcs: dict[int, set[int]] = {}  # component -> set of arc_ids

    # ------------------------------------------------------------------
    # Network setup (inference-only: nodes + arcs, never .route)
    # ------------------------------------------------------------------
    def set_network(self, nodes: np.ndarray, arcs: np.ndarray) -> None:
        """Build directed graph from nodes/arcs arrays.

        nodes: (N, 2) array of [lon, lat]
        arcs: (M, 2) array of [from_node_id, to_node_id]
        """
        self._node_coords = {i: (float(nodes[i, 0]), float(nodes[i, 1])) for i in range(len(nodes))}
        self._directed_arcs = {int(a[0]): (int(a[0]), int(a[1])) for a in arcs}
        # Actually arcs rows are [from_node, to_node]
        self._directed_arcs = {i: (int(arcs[i, 0]), int(arcs[i, 1])) for i in range(len(arcs))}

        self._adjacency = {}
        self._reverse_adjacency = {}
        for i in range(len(arcs)):
            f, t = int(arcs[i, 0]), int(arcs[i, 1])
            self._adjacency.setdefault(f, set()).add(t)
            self._reverse_adjacency.setdefault(t, set()).add(f)

        # Weakly connected components (undirected)
        self._compute_components(arcs)

    def _compute_components(self, arcs: np.ndarray) -> None:
        """Union-find weakly connected components over nodes."""
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(arcs)):
            f, t = int(arcs[i, 0]), int(arcs[i, 1])
            union(f, t)

        for i in range(len(arcs)):
            f, t = int(arcs[i, 0]), int(arcs[i, 1])
            c = find(f)
            self._connected_components[f] = c
            self._connected_components[t] = c

        # Map component -> arcs
        self._component_arcs = {}
        for i in range(len(arcs)):
            c = self._connected_components.get(int(arcs[i, 0]), -1)
            self._component_arcs.setdefault(c, set()).add(i)

    def same_component(self, node_a: int, node_b: int) -> bool:
        return self._connected_components.get(node_a, -1) == self._connected_components.get(
            node_b, -2
        )

    # ------------------------------------------------------------------
    # Emission probabilities from classifier scores
    # ------------------------------------------------------------------
    def compute_emissions(
        self, obs_index: int, candidates: list[CandidateState], gps_latlon: tuple[float, float]
    ) -> None:
        """Set emission_prob for each candidate state.

        Combines classifier probability with geometric distance to segment.
        Temperature scales the classifier logit; distance provides a
        geometry-based floor that prevents overconfident wrong-arc selection.
        """
        temp = max(self.settings.emission_temperature, 1e-6)
        for c in candidates:
            # Classifier-derived emission (already a probability 0..1)
            p_class = float(np.clip(c.classifier_prob, 0.0, 1.0))
            # Temperature scaling: sharpen or soften
            logit = math.log((p_class + 1e-9) / (1.0 - p_class + 1e-9))
            p_scaled = 1.0 / (1.0 + math.exp(-logit / temp))
            # Geometry factor: closer = higher (Gaussian-ish on distance)
            sigma = max(self.settings.distance_scale_m * 0.3, 1.0)
            geo_factor = math.exp(-0.5 * (c.distance_m / sigma) ** 2)
            # Combined emission
            p = 0.7 * p_scaled + 0.3 * geo_factor
            c.emission_prob = float(np.clip(p, self.settings.emission_floor, 1.0))

    # ------------------------------------------------------------------
    # Transition scores between consecutive observation states
    # ------------------------------------------------------------------
    def compute_transition(
        self, prev_state: CandidateState, curr_state: CandidateState,
        dt_s: float, gps_displacement_m: float, gps_heading_rad: float
    ) -> TransitionScore:
        """Log transition probability from prev_state to curr_state.

        Factors:
        - Directed connectivity (is there a path from prev.to_node to curr.from_node?)
        - Network path distance vs GPS displacement
        - Implied speed vs max_speed_mps
        - Heading agreement between road segment and GPS motion
        - Same-arc stay (dwelling)
        - Component mismatch
        """
        s = self.settings

        # Same arc: dwelling is always allowed
        if prev_state.arc_id == curr_state.arc_id:
            return TransitionScore(
                from_arc=prev_state.arc_id,
                to_arc=curr_state.arc_id,
                log_prob=math.log(max(stay_arc_bonus := s.stay_arc_bonus, s.transition_floor)),
                is_connected=True,
                path_length_m=0.0,
                path_edges=[prev_state.arc_id],
                reason="same_arc_dwell",
            )

        from_node = prev_state.to_node
        to_node = curr_state.from_node

        # Component check
        if not self.same_component(from_node, to_node):
            return TransitionScore(
                from_arc=prev_state.arc_id,
                to_arc=curr_state.arc_id,
                log_prob=math.log(s.transition_floor),
                is_connected=False,
                reason="different_components",
            )

        # Directed path search (bounded BFS)
        path = self._bounded_directed_path(from_node, to_node, max_edges=50)
        if path is None:
            return TransitionScore(
                from_arc=prev_state.arc_id,
                to_arc=curr_state.arc_id,
                log_prob=math.log(s.transition_floor),
                is_connected=False,
                reason="no_directed_path",
            )

        path_length_m = self._path_length_m(path)
        path_edges = list(path)

        # Implied speed
        implied_speed = path_length_m / max(dt_s, 1e-6) if dt_s > 0 else float("inf")

        # Speed plausibility
        speed_ok = implied_speed <= s.max_speed_mps
        speed_penalty = 0.0
        if not speed_ok:
            speed_penalty = s.speed_penalty_coeff * math.log(
                1.0 + (implied_speed - s.max_speed_mps) / max(s.max_speed_mps, 1.0)
            )

        # Distance decay: long network path vs short GPS displacement is suspicious
        distance_ratio = path_length_m / max(gps_displacement_m, 1.0)
        distance_penalty = math.log(1.0 + max(0.0, distance_ratio - 1.0) / s.distance_scale_m)

        # Heading agreement
        road_heading = curr_state.segment_heading_rad
        heading_diff = abs(math.atan2(math.sin(gps_heading_rad - road_heading),
                                       math.cos(gps_heading_rad - road_heading)))
        heading_agreement = math.cos(heading_diff)  # 1 = aligned, -1 = opposite
        heading_penalty = s.heading_weight * max(0.0, -heading_agreement)

        # Combine into log probability
        base = 0.0  # neutral
        log_prob = base - speed_penalty - distance_penalty - heading_penalty

        # Floor
        log_prob = max(log_prob, math.log(s.transition_floor))

        return TransitionScore(
            from_arc=prev_state.arc_id,
            to_arc=curr_state.arc_id,
            log_prob=log_prob,
            is_connected=True,
            path_length_m=path_length_m,
            path_edges=path_edges,
            reason="connected",
        )

    # ------------------------------------------------------------------
    # Directed path search (bounded BFS on the road graph)
    # ------------------------------------------------------------------
    def _bounded_directed_path(self, start: int, goal: int, max_edges: int = 50) -> list[int] | None:
        """Find a directed path of arc IDs from start node to goal node.

        Bounded BFS to avoid unbounded search on large networks.
        Returns list of arc IDs, or None if no path within max_edges.
        """
        if start == goal:
            return []

        # BFS over nodes, tracking arc IDs used
        from collections import deque

        visited = {start}
        queue = deque([(start, [])])  # (node, path_of_arc_ids)

        while queue:
            node, path = queue.popleft()
            if len(path) >= max_edges:
                continue
            for to_node in self._adjacency.get(node, ()):
                if to_node in visited:
                    continue
                # Find arc(s) from node to to_node
                arc_id = self._find_arc(node, to_node)
                if arc_id is None:
                    continue
                new_path = path + [arc_id]
                if to_node == goal:
                    return new_path
                visited.add(to_node)
                queue.append((to_node, new_path))

        return None

    def _find_arc(self, from_node: int, to_node: int) -> int | None:
        """Find an arc ID from from_node to to_node (first match)."""
        for arc_id, (f, t) in self._directed_arcs.items():
            if f == from_node and t == to_node:
                return arc_id
        return None

    def _path_length_m(self, path: list[int]) -> float:
        """Sum of segment lengths for a list of arc IDs."""
        # We need segment lengths; stored via candidate states or compute here.
        # For simplicity, estimate from node coordinates.
        total = 0.0
        for arc_id in path:
            f, t = self._directed_arcs[arc_id]
            lon1, lat1 = self._node_coords[f]
            lon2, lat2 = self._node_coords[t]
            total += self._haversine_m(lat1, lon1, lat2, lon2)
        return total

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2) -> float:
        R = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))
