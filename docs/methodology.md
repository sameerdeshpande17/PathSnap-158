# Phase 1 methodology and limitations

## Evidence and scope

The five PDFs under `docs/official/` were read, including the assessment and
plagiarism policy. Text extractions accompany them. Original ZIPs are preserved;
`data/original/manifest.json` records their SHA-256 hashes and archive membership.
The analysis independently reads original extracted files. No precomputed
statistics or supplied scripts are trusted. A separately supplied Phase 1 package
was not located in the searched Downloads/Documents locations: its inspection and
comparison remain blocked until its path is provided.

The official documents mention a first evaluation before lunch, but do not give a
separate exhaustive Phase 1 rubric. This checkpoint covers the dataset exploration
requirements and a geometric baseline. It is not a claim of final submission
completeness.

## Parsing and preprocessing

All four file types are whitespace-delimited, headerless. `.track` has longitude,
latitude, timestamp in seconds; `.nodes` has longitude, latitude; `.arcs` has
integer source and destination node row IDs; `.route` has one integer arc row ID.
IDs are zero-based and local to a trip. Do not learn global arc IDs as classes.

Strict parsing checks every row's width, numeric finiteness, integer IDs and
index bounds. Nonfinite values, blank rows, malformed records and out-of-bounds
coordinates or IDs cause a recorded parsing error, not silent imputation.
Valid input rows are preserved in original order. No timestamps are sorted,
no GPS points are deleted, and no reference discontinuities are repaired.
Missing/invalid counts of zero are valid only for fully parsed trips; failures
are recorded separately and make the run fail.

Longitude-first is established by the official schema, coordinate ranges and
plausible mapped locations; range tests alone cannot always distinguish swapped
coordinates. Pyproj uses `always_xy=True`; Folium gets latitude first.
Observed timestamps are treated as seconds relative to an unspecified origin;
no calendar date or UTC epoch is inferred.

WGS84 geodesics measure consecutive GPS displacements and headings. Per-trip
azimuthal equidistant projection centred on the mean GPS position provides
metres for segment geometry. This is a local approximation away from the
projection centre, not a globally distortion-free metric. Quantified projection
distance checks are retained in `experiments/projection_validation.json`.
Arcs are straight segments between supplied nodes; there is no additional
road curvature, speed limit, road class or traffic metadata in this schema.

Connectivity uses directed sparse graphs: weak and strong components, isolated
nodes, self-loops and duplicate arcs. Reference continuity means the destination
node of one route arc equals the source of the next. A break does not prove the
vehicle teleported: it indicates limitations in the supplied representation.

Thresholds (>5 seconds, >55 m/s, >100 metres) are transparent exploratory flags,
not tuned thresholds or reasons to discard records. Dataset-quality counts
include duplicate coordinates, repeated positions, timestamp ordering, zero
length segments and reference-route repetitions.

## Derived observation features

`data/processed/{train,student_test}/*_observations.csv` contains timestamp,
delta time, geodesic step length, inferred speed and heading. First-observation
differences are undefined and deliberately blank; speed is undefined when
delta time is nonpositive. Heading at zero displacement must later be masked.
These features describe sampling, motion and direction, not supervised labels.
No learned scaling or imputation was fitted. Candidate-relative distance,
heading agreement, endpoint position, network path distance and gap flags are
planned features; their usefulness still requires training-only validation.

## Reference route and supervised-label investigation

`.route` is the chronological sequence of traversed arc IDs. Its length generally
differs from GPS length. Arc repetition is meaningful; a set discards order and
revisits. The files contain no explicit timestamp-to-reference-arc alignment.
Consequently, row-wise zipping, resizing, or declaring each nearest road the true
label is unsupported.

A possible future weak-label construction:

1. Preserve each reference occurrence as `(route_position, arc_id)`.
2. Project GPS observations onto the reference segments, retaining competing
   occurrences and residual distances.
3. Align monotonically in route position using a dynamic program that permits
   dwelling and skipped segments, with geodesic displacement and elapsed time
   constraints. Split or abstain around reference discontinuities and long gaps.
4. Retain uncertainty, runner-up margins and abstention flags. Audit sampled
   alignments visually; measure sensitivity to alignment assumptions.
5. Ask organisers whether an official alignment/evaluator is available.

These would be **pseudo-labels**, not independently verified pointwise ground
truth. Never label every route-member candidate positive at every observation.
Never align evaluation truth inside inference. The label-construction policy
must be fixed before reporting comparable observation-level metrics.

## Geometric baseline

`nearest_road` takes only GPS coordinates, road nodes and directed arcs.
Shapely STRtree finds exact nearest point-to-straight-segment distances in the
local projection. Equidistant arcs are retained for tie counting; smallest
arc ID deterministically breaks ties. Reverse-direction arcs often share
geometry, so this baseline cannot reliably infer travel direction.

All observations in both permitted partitions are processed. Results are
per-observation CSVs. This is not a trained model, candidate classifier or HMM.
No accuracy, precision, recall, F1 or log loss is reported because aligned
pointwise reference labels are unavailable.

Reported diagnostics:

- Nearest distance: GPS to the selected road segment, not route accuracy.
- Direct adjacency: consecutive predictions are identical or the first arc's
  destination equals the next arc's source. Skipping intermediate arcs can fail
  this test despite reachable roads; passing it does not prove forward motion
  within an arc. Macro averages weight trips equally; micro weights transitions.
- Reference membership: fraction of predicted arc IDs appearing anywhere in
  the trip's route; ignores temporal order and is **not pointwise accuracy**.
- Baseline time includes geometry construction, indexing, queries and snapping;
  excludes file loading, projection and writing. CPU wall clock, single run.

## Visual evidence

Select training trips by near-median observation count, maximum timestamp gap,
and most reference adjacency breaks (deduplicate selected trips). Static plots
and Folium layers show GPS, extent-clipped road geometry, supplied reference
arcs and the union of predicted nearest arcs. An unordered arc union is not a
continuous reconstructed route. Reference arcs are drawn individually to avoid
inventing connectors across breaks. HTML maps use external Leaflet assets and
optional OpenStreetMap tiles. Generated screenshots are actual browser renders.

## Subsequent experiment design (not yet executed)

- Candidate generation: train-validated radius/top-k directed segments, stable
  `(trip, observation, arc)` correspondence, null-state policy, candidate recall.
- Train five distinct families: logistic regression, k-nearest neighbours,
  random forest, histogram gradient boosting and a small MLP are provisional.
  Final configurations require experiments, not copied or assumed results.
- GroupKFold by trip on training data only, with fold-local preprocessing and
  calibration; inspect geographic overlap and consider region-group sensitivity.
  Student testing remains development evaluation, not an unbiased final holdout.
- Record seeds, folds, configurations, candidate coverage, Macro F1, Top-1,
  log loss where meaningful, confusion diagnostics, timing and route metrics.
  Decide primary selection metric after establishing defensible labels.
- Bayesian optimisation on training group-CV; persist trials, search rationale,
  objective and costs; compare untuned and tuned on the same student-test trips.
- HMM hidden states are candidate arc occurrences; model probabilities become
  emissions. Directed shortest paths and observed motion inform transitions.
  Log-space Viterbi handles invalid transitions, missing candidates and gaps.
  Tune decoder only on permitted development data and compare classifier-only.
- Artifacts include preprocessing, model, feature schema, candidate settings,
  decoder configuration, package versions and provenance.
- FastAPI loads artifacts once, excludes references from inference, computes
  metrics separately server-side, and logs request IDs/timings. See API contract.

## Integrity and assistance

This checkpoint was implemented with AI coding assistance and executed locally.
Team members must review and understand the code and measured findings before
submission. No trained artifacts, code or results from another team were used.
No publication or push was performed. Evaluator data is absent and was not used.
