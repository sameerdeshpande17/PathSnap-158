# PathSnap Phase 3B — HMM/Viterbi Route Reconstruction Methodology

## Overview

PathSnap uses a Hidden Markov Model (HMM) with Viterbi decoding to convert
noisy GPS observations into a connected sequence of directed road arc IDs.
The pipeline runs entirely on inference-time data: GPS observations, road
network nodes/arcs, and a trained classifier. Reference-route (`.route`) files
are never accessed during inference.

## Hidden states

Each hidden state corresponds to a candidate road-segment assignment for one
GPS observation. A state is a tuple:

- `arc_id`: directed road segment identifier
- `from_node`, `to_node`: directed endpoint node IDs
- `segment_length_m`: arc length in metres
- `segment_heading_rad`: directed bearing from `from_node` to `to_node`
- `projection_fraction`: where along the segment the GPS point projects (0..1)
- `distance_m`: perpendicular distance from GPS point to segment
- `classifier_prob`: probability from the trained classifier
- `emission_prob`: combined emission probability (see below)

## Emission probabilities

Emission probability combines the classifier's probability with geometric
distance:

```
p_class = clip(classifier_prob, 0, 1)
logit = log((p_class + eps) / (1 - p_class + eps))
p_scaled = sigmoid(logit / temperature)
geo_factor = exp(-0.5 * (distance / sigma)^2)   # sigma = distance_scale * 0.3
emission = clip(0.7 * p_scaled + 0.3 * geo_factor, emission_floor, 1)
```

- `temperature`: sharpens or softens classifier confidence (default 1.0)
- `distance_scale`: controls how quickly geometric distance penalizes
  far-away candidates (default 100m)
- `emission_floor`: prevents log(0) (default 1e-9)

## Transition probabilities

Transition probability from state `s_prev` to state `s_curr` accounts for:

1. **Directed connectivity**: Is there a directed path from `s_prev.to_node`
   to `s_curr.from_node` in the road graph? Computed via bounded BFS
   (max 50 edges). If no path exists, the transition is heavily penalized.
2. **Self-transition (dwelling)**: Staying on the same arc is always allowed
   with a small bonus (`stay_arc_bonus`, default 1.0).
3. **Speed plausibility**: Implied speed = network path length / elapsed time.
   Exceeding `max_speed_mps` (default 35 m/s ≈ 126 km/h) incurs a penalty.
4. **Distance decay**: Long network path vs short GPS displacement is
   suspicious; penalized proportionally.
5. **Heading agreement**: Cosine of the angle between GPS heading and road
   heading. Misalignment is penalized by `heading_weight` (default 0.3).

All penalties are combined in log space:

```
log_prob = -speed_penalty - distance_penalty - heading_penalty
log_prob = max(log_prob, log(transition_floor))   # floor = 1e-9
```

## Viterbi decoding

Log-space Viterbi finds the most likely sequence of hidden states:

- **Initialization**: First observation states weighted by emission only
  (uniform prior over states).
- **Recursion**: For each observation, consider transitions from beam-retained
  previous states. Beam width defaults to 64.
- **Termination**: Select the final state with highest accumulated log prob.
- **Backtracking**: Reconstruct the full state sequence via backpointers.

Beam pruning keeps computation tractable: only the top-K previous states by
accumulated log prob are considered at each step.

## Route reconstruction

The decoded sequence of arc IDs is converted into a connected route:

1. **Collapse runs**: Consecutive observations on the same arc form a "run"
   (dwelling). Runs are maximal sequences of identical arc IDs.
2. **Connect runs**: Between consecutive runs, find the shortest directed
   path in the road graph using bounded BFS. If no path exists, the gap is
   flagged as `disconnected_gap` (not silently repaired).
3. **Detect repeats**: The same arc ID appearing in non-consecutive runs
   indicates a repeated road traversal.
4. **Output**: Route segments include both decoded arcs and connector arcs,
   with observation-to-arc mapping and diagnostics.

No reference-route data is used during reconstruction.

## Performance

The original implementation performed a linear scan over all arcs (~27,000
per trip) for every edge exploration in BFS, making `_bounded_directed_path`
take ~200ms per call. Optimizations:

- **O(1) arc lookup**: `(from_node, to_node) -> arc_id` dictionary
- **Path caching**: Results cached per `(start, goal)` pair
- **Precomputed arc lengths**: Avoid recomputing haversine distances

Measured speedup on `_bounded_directed_path`: ~10,000x (202ms → 0.02ms per
call). Full trip decode (3,958 GPS obs): ~10.5 seconds end-to-end.

## Limitations

- The HMM assumes the classifier provides well-calibrated emission
  probabilities. With weak labels, calibration may be imperfect.
- Transition penalties (speed, distance, heading) are heuristic and not
  learned from data.
- Bounded BFS (max 50 edges) may miss long connector paths between distant
  arcs.
- All results are weak-label agreement, not verified ground-truth accuracy.