# Measured data-quality findings

All counts are computed from original supplied files; node/arc totals count
trip-local records, not globally unique physical roads.

## student_test

- 10 trips; 19,160 GPS observations.
- Zero missing/nonfinite values and invalid IDs in successfully parsed files.
- 0 nonpositive time deltas; 44 gaps >5s; maximum 210s.
- 1 reference adjacency breaks in 1 trips.
- 6 zero-length arcs; 128 duplicate directed arc rows; 2 self-loops.
- 3 duplicate node-coordinate rows; 0 stationary consecutive GPS pairs.
- 10 networks have multiple weak components; 0 isolated nodes.
- 4 observed speeds >55m/s (flags, not confirmed errors).
- 0 GPS-to-nearest-road distances >100m.
- 6265 observations have tied nearest arcs.
- Longitude range -93.424337 to 77.271115; latitude 42.645529 to 61.699590.

## train

- 80 trips; 208,417 GPS observations.
- Zero missing/nonfinite values and invalid IDs in successfully parsed files.
- 0 nonpositive time deltas; 1216 gaps >5s; maximum 153064s.
- 13 reference adjacency breaks in 10 trips.
- 100 zero-length arcs; 1775 duplicate directed arc rows; 22 self-loops.
- 89 duplicate node-coordinate rows; 6 stationary consecutive GPS pairs.
- 79 networks have multiple weak components; 0 isolated nodes.
- 991 observed speeds >55m/s (flags, not confirmed errors).
- 1417 GPS-to-nearest-road distances >100m.
- 78583 observations have tied nearest arcs.
- Longitude range -100.418538 to 137.237029; latitude -34.640291 to 61.267258.

## Interpretation

No trip has equal route-row and GPS-row counts. Pointwise reference labels
are not supplied. Connectivity breaks and direction ties require explicit
handling in later alignment and decoding. Networks span diverse regions;
a single fixed UTM zone would be unsuitable for the entire dataset.
No duplicate track SHA-256 hashes or overlapping trip IDs were found
between train and student testing; this does not establish geographic independence.
See anomaly_locations.csv for zero-based observation/route row locations.
Projection checks compare every nontrivial arc's projected length with its
WGS84 geodesic endpoint distance; they do not prove exact geodesic nearest-road ordering.
