# Phase 2B feature engineering

## Design principle

Every feature is computable at inference time from a GPS observation and the
road network alone: observation coordinates/timestamp, node coordinates, and
directed arcs. No feature uses `.route`, reference-route membership,
reference-route occurrence indices, reference progress, weak labels, or any
target-derived information. The candidate-generation module
(`src/pathsnap/candidates.py`) opens only `track`, `nodes`, and `arcs`.

All distances use per-trip WGS84 azimuthal-equidistant local metres centred on
the trip's mean GPS position (longitude-first, PyProj `always_xy=True`), the
same projection as Phase 1. Headings are bearings from north.

## Persisted schema

Feature definitions, order, dtype (`float32`), and schema version
`pathsnap-candidates-v1` are recorded in:

- `src/pathsnap/candidates.py` constants `FEATURES` / `SCHEMA_VERSION`
- `experiments/phase2b/config.json` (`feature_schema`, `feature_version`)
- every saved model artifact (`features`, `schema_version`)
- `docs/model_evaluation_methodology.md` preprocessing section

Preprocessing (StandardScaler) is fitted **within each training fold only**
and serialized with the model. No preprocessing parameter is learned from
validation or student-testing data.

## Feature list

| # | Feature | Type | Purpose / inference availability |
|---|---|---|---|
| 1 | `distance_m` | continuous | Metre distance from GPS point to candidate segment (STRtree exact query). Core matching evidence. |
| 2 | `log_distance` | continuous | `log1p(distance)`; compresses long-tail distances. |
| 3 | `relative_distance_m` | continuous | Distance minus the observation's minimum candidate distance. Isolates near-field competition. |
| 4 | `distance_ratio` | continuous | Distance / (min distance + 1). Scale-relative nearness. |
| 5 | `candidate_rank` | ordinal | Rank by distance within the observation (0 = nearest). |
| 6 | `candidate_count` | count | Number of candidates generated for the observation (radius/cap/fallback dependent). |
| 7 | `nearby_arc_count` | count | All arcs within the search radius before ranking cap; local density. |
| 8 | `distance_tie_count` | count | Candidates within 1 m of the minimum distance; flags geometric ambiguity. |
| 9 | `segment_length_m` | continuous | Directed arc length in metres. |
| 10 | `log_segment_length` | continuous | `log1p(segment_length)`. |
| 11 | `road_heading_sin` / `road_heading_cos` | circular | Directed segment bearing (from_node → to_node). Available at inference from nodes/arcs only. |
| 12 | `gps_heading_sin` / `gps_heading_cos` | circular | Bearing from previous to current GPS observation. Zero when displacement/heading undefined (first observation, zero movement, or invalid dt). |
| 13 | `heading_difference_rad` | continuous | Wrapped absolute difference between road and GPS headings; zero when either is invalid. |
| 14 | `directional_agreement` | continuous | `cos(heading difference)` when both headings valid, else 0. Rewards same-direction travel on directed arcs. |
| 15 | `speed_mps` | continuous | Estimated GPS speed `step_m / dt_s`, clipped to [0, 100]; zero when dt invalid. |
| 16 | `dt_s` | continuous | Seconds since previous observation, clipped to [0, 300]; first observation = 0. |
| 17 | `movement_m` | continuous | Geodesic/projected displacement since previous observation, clipped [0, 1000]. |
| 18 | `heading_valid` | binary | 1 when GPS heading is defined. |
| 19 | `road_heading_valid` | binary | 1 when segment length > 0. |
| 20 | `large_gap` | binary | 1 when `dt_s > 5 s`. |
| 21 | `time_valid` | binary | 1 when `dt_s > 0`. |
| 22 | `projection_fraction` | continuous | Along-segment projection of the GPS point, clamped to [0, 1]; 0/1 mark endpoints. |
| 23 | `endpoint_distance_m` | continuous | Distance to the nearer arc endpoint along the segment. |
| 24 | `from_outdegree` / `to_indegree` | counts | Directed graph degree of the segment endpoints (local topology). |
| 25 | `fallback_used` | binary | 1 when the observation had no arc within radius and fell back to the global nearest arc. |

## Safe handling of degenerate data

- Zero-length movement: heading marked invalid (zeros, flag `heading_valid=0`) rather than dividing by zero.
- Duplicate coordinates: produce zero movement; handled as above.
- First observation: `dt_s = 0`, movement 0, heading invalid; `time_valid = 0`.
- Large time gaps: `dt_s` still reported and flagged `large_gap` (>5 s); models see the flag and raw value.
- Missing/nonpositive `dt`: cannot occur after strict Phase 1 parsing (timestamps strictly increase), but defensive clipping keeps values bounded.
- Zero-length arcs: `road_heading_valid = 0`, heading/trig features zeroed, endpoint distance 0.
- No candidates within radius: fallback to single nearest arc with `fallback_used = 1`; or zero candidates when `fallback = none`.
- Parallel/duplicate geometry: each directed arc is a distinct candidate row with its own arc ID.

## Deliberately excluded features

- Reference-route arc IDs / occurrence indices: not available at inference and would leak the target.
- Reference progress, residual, or alignment confidence: label-construction outputs only.
- Any statistic computed from weak labels or route membership.

The weak label is joined strictly after inference-time feature generation, in an offline labeling step, and exists only in training/evaluation tables.
