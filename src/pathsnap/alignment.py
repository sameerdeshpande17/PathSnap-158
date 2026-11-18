"""Reference-assisted WEAK labels. Never import into prediction feature generation."""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .data import project


@dataclass
class Geometry:
    distances: np.ndarray
    progress: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    break_prefix: np.ndarray
    steps: np.ndarray
    dt: np.ndarray
    gps_xy: np.ndarray
    nodes_xy: np.ndarray


def geometry(trip):
    """All occurrence geometries; duplicates intentionally retained."""
    if len(trip.route) == 0:
        raise ValueError("Empty reference route")
    if not np.issubdtype(trip.route.dtype, np.integer):
        raise ValueError("Route IDs must be integers")
    if (trip.route < 0).any() or (trip.route >= len(trip.arcs)).any():
        raise ValueError("Invalid reference arc ID")
    if len(trip.track) == 0 or not np.isfinite(trip.track).all():
        raise ValueError("Invalid observations")
    dt = np.r_[0., np.diff(trip.track[:, 2])]
    if (dt[1:] <= 0).any():
        raise ValueError("Timestamps must strictly increase")
    gps, nodes = project(trip.track, trip.nodes)
    arcs = trip.arcs[trip.route]
    start = nodes[arcs[:, 0]]
    vector = nodes[arcs[:, 1]] - start
    length = np.linalg.norm(vector, axis=1)
    # Broken boundaries are NOT repaired; progress omits unknown connecting path.
    # Such transitions are separately penalized and flagged.
    cumulative = np.r_[0., np.cumsum(length[:-1])]
    distances = np.empty((len(gps), len(arcs)))
    progress = np.empty_like(distances)
    denominator = np.sum(vector * vector, axis=1)
    for offset in range(0, len(gps), 512):
        displacement = gps[offset:offset + 512, None, :] - start
        fraction = np.clip(np.divide(
            np.sum(displacement * vector, axis=2), denominator,
            out=np.zeros(displacement.shape[:2]), where=denominator > 0), 0, 1)
        distances[offset:offset + 512] = np.linalg.norm(
            displacement - fraction[:, :, None] * vector, axis=2)
        progress[offset:offset + 512] = cumulative + fraction * length
    breaks = arcs[:-1, 1] != arcs[1:, 0]
    return Geometry(distances, progress, cumulative, length,
                    np.r_[0, np.cumsum(breaks)],
                    np.r_[0., np.linalg.norm(np.diff(gps, axis=0), axis=1)],
                    dt, gps, nodes)


def emission_cost(distances, method, config):
    if method == "B":
        return distances / config["sigma_m"]
    if method == "C":
        return .5 * (distances / config["sigma_m"]) ** 2
    raise ValueError(method)


def transition_cost(g, i, previous, current, method, config):
    """Matrix [previous, current]; nondecreasing occurrence indices required."""
    delta = g.progress[i, current][None, :] - g.progress[i - 1, previous][:, None]
    dt = g.dt[i]
    beta = config["beta_m"] * (1 + np.sqrt(min(max(dt - 1, 0), 300)))
    excess = np.maximum(delta - (config["max_speed_mps"] * dt + config["speed_slack_m"]), 0)
    crossed = g.break_prefix[current][None, :] - g.break_prefix[previous][:, None]
    # Across a documented break, path-distance consistency is unknowable.
    # Retain the transition with an explicit penalty and later exclude its label.
    mismatch = np.abs(delta - g.steps[i]) / beta
    mismatch = np.where(crossed > 0, 0., mismatch)
    cost = (config["progress_weight"] * mismatch +
            config["break_penalty"] * np.maximum(crossed, 0))
    cost[previous[:, None] > current[None, :]] = np.inf
    if method == "B":
        return cost + config["speed_penalty"] * excess / config["speed_slack_m"]
    if method != "C":
        raise ValueError(method)
    # A break has unknown connecting distance; do not pretend to enforce its speed.
    # Within-occurrence position is a projection, not a latent motion state.
    # Permit self-occurrence transitions even when noisy projections jump.
    # Diagnostics still flag the excess; this guarantees a finite path.
    cost[(excess > 0) & (crossed == 0) &
         (previous[:, None] != current[None, :])] = np.inf
    # Conditional Markov transition probabilities on this candidate lattice.
    normalizer = logsumexp(-cost, axis=1)
    feasible = np.isfinite(normalizer)
    cost[feasible] += normalizer[feasible, None]
    return cost


def solve(g, method, config, progress_callback=None):
    """Beam-pruned Viterbi DP; exact when beam and candidate counts >= route size.

    Current states are nearest K occurrences UNION retained previous states.
    Retaining previous states guarantees a nondecreasing occurrence path exists.
    Large gaps relax progression uncertainty, never reset route position.
    """
    if method == "A":
        return np.argmin(g.distances, axis=1), {"objective": None, "peak_states": g.distances.shape[1]}
    n, m = g.distances.shape
    beam = min(config["beam_width"], m)
    k = min(config["candidate_count"], m)
    # Stable tie handling retains earliest occurrences, not unique arc IDs.
    nearest = np.argsort(g.distances, axis=1, kind="stable")[:, :k]
    states = np.full((n, beam), -1, dtype=np.int32)
    parents = np.full((n, beam), -1, dtype=np.int32)
    initial = emission_cost(g.distances[0], method, config)
    selected = np.argsort(initial, kind="stable")[:beam]
    states[0, :len(selected)] = selected
    scores = initial[selected]
    offset = float(scores.min())
    scores = scores - offset
    previous = selected
    peak = len(previous)
    for i in range(1, n):
        current = np.union1d(nearest[i], previous)
        transition = transition_cost(g, i, previous, current, method, config)
        total = scores[:, None] + transition
        parent = np.argmin(total, axis=0)
        values = total[parent, np.arange(len(current))]
        values += emission_cost(g.distances[i, current], method, config)
        selected = np.argsort(values, kind="stable")[:beam]
        selected = selected[np.isfinite(values[selected])]
        if len(selected) == 0:
            raise RuntimeError(f"No feasible alignment at observation {i}")
        states[i, :len(selected)] = current[selected]
        parents[i, :len(selected)] = parent[selected]
        scores = values[selected]
        minimum = float(scores.min())
        scores -= minimum
        offset += minimum
        previous = current[selected]
        peak = max(peak, len(current))
        if progress_callback and i % 5000 == 0:
            progress_callback(i, n)
    path = np.empty(n, dtype=np.int32)
    state = int(np.argmin(scores))
    for i in range(n - 1, -1, -1):
        path[i] = states[i, state]
        state = parents[i, state]
    return path, {"objective": offset + float(scores.min()), "peak_states": peak}


def diagnostics(g, path, config):
    distance = g.distances[np.arange(len(path)), path]
    progress = g.progress[np.arange(len(path)), path]
    delta = np.r_[0., np.diff(progress)]
    reverse = np.r_[False, np.diff(path) < 0]
    crossing = np.r_[False, np.diff(g.break_prefix[path]) != 0]
    # Same definition for every method, independent of its enforced constraints.
    jump = (delta > config["max_speed_mps"] * g.dt + config["speed_slack_m"]) & ~crossing
    backwards = delta < -config["speed_slack_m"]
    ambiguous = (g.distances <= g.distances.min(axis=1)[:, None] +
                 config["ambiguity_m"]).sum(axis=1) > 1
    gap = g.dt > config["gap_s"]
    gap_adjacent = gap | np.r_[gap[1:], False]
    high = distance > config["residual_limit_m"]
    zero = g.lengths[path] == 0
    flags = {
        "high_residual": high, "geometric_occurrence_ambiguity": ambiguous,
        "route_order_violation": reverse, "unexplained_jump": jump,
        "backward_progress": backwards, "reference_break_crossing": crossing,
        "large_gap_adjacent": gap_adjacent, "zero_length_arc": zero,
    }
    # Geometry ambiguity is retained as a warning; consensus screening later
    # requires B/C occurrence agreement and excludes all other flags.
    eligible = ~np.logical_or.reduce([v for k, v in flags.items()
                                     if k != "geometric_occurrence_ambiguity"])
    return distance, progress, flags, eligible
