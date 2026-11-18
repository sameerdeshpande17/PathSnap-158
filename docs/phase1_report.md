# PathSnap - Phase 1 Evidence Report

Dataset exploration, reference-format investigation and geometric baseline

Checkpoint generated from local experiment artifacts. No trained classifier, Bayesian optimisation, HMM, FastAPI deployment or UI demonstration is claimed.

AI assistance: code and documentation were developed with an AI coding agent. Experiments were executed locally; team review and understanding remain required.

## 1. Scope, inputs and reproducibility

All five original official guideline PDFs were read, including assessment and plagiarism policy. See docs/official for originals and extracted text. The documents mention a pre-lunch first evaluation without a separate detailed Phase 1 rubric.

Input: PathSnap_student_materials.zip; SHA-256: db368793f68c76baa8eaa30d783d376df50841287c29c43f6911feb98cab4367

Input: Student folder.zip; SHA-256: 4804b0399d1d56f4844b9c74e2f998aae0a281fb5947a1f47ad1bd013214d248

Python: 3.12.12 (main, Feb  3 2026, 22:51:04) [Clang 21.1.4 ]; executable: /home/rohitpujari/ml312/bin/python.

PyTorch: 2.11.0+cu128; CUDA available: True; GPU: NVIDIA GeForce RTX 4060 Laptop GPU.

Exploration wall time: 28.997s for 90 trips; failed trips: 0; models trained: 0.

Reproduce from the project root using README.md. Commands and raw logs are retained under experiments/. Node and arc counts are sums of trip-local records, not globally unique roads.

## 2. Dataset understanding and exploratory data analysis

Measured quantity | Training | Student testing

Trips | 80 | 10

GPS observations | 208,417 | 19,160

Node records | 1,933,485 | 90,689

Directed arc records | 3,876,904 | 185,041

Reference arc occurrences | 31,336 | 3,477

Missing numeric values | 0 | 0

Nonpositive time deltas | 0 | 0

Gaps >5 seconds | 1,216 | 44

Maximum gap (seconds) | 153,064.0 | 210.0

Reference adjacency breaks | 13 | 1

Zero-length arcs | 100 | 6

Exactly-one-second step fractions: training 96.5013%; student testing 98.6214%. The evaluator partition was not supplied and was not accessed.

Figure 1. All available trips; partitions shown separately. Speed is inferred from GPS.

![Figure 1. All available trips; partitions shown separately. Speed is inferred from GPS.](../visualizations/dataset_overview.png)

### Measured data-quality findings

All counts are computed from original supplied files; node/arc totals count trip-local records, not globally unique physical roads.

### student_test

- 10 trips; 19,160 GPS observations. - Zero missing/nonfinite values and invalid IDs in successfully parsed files. - 0 nonpositive time deltas; 44 gaps >5s; maximum 210s. - 1 reference adjacency breaks in 1 trips. - 6 zero-length arcs; 128 duplicate directed arc rows; 2 self-loops. - 3 duplicate node-coordinate rows; 0 stationary consecutive GPS pairs. - 10 networks have multiple weak components; 0 isolated nodes. - 4 observed speeds >55m/s (flags, not confirmed errors). - 0 GPS-to-nearest-road distances >100m. - 6265 observations have tied nearest arcs. - Longitude range -93.424337 to 77.271115; latitude 42.645529 to 61.699590.

### train

- 80 trips; 208,417 GPS observations. - Zero missing/nonfinite values and invalid IDs in successfully parsed files. - 0 nonpositive time deltas; 1216 gaps >5s; maximum 153064s. - 13 reference adjacency breaks in 10 trips. - 100 zero-length arcs; 1775 duplicate directed arc rows; 22 self-loops. - 89 duplicate node-coordinate rows; 6 stationary consecutive GPS pairs. - 79 networks have multiple weak components; 0 isolated nodes. - 991 observed speeds >55m/s (flags, not confirmed errors). - 1417 GPS-to-nearest-road distances >100m. - 78583 observations have tied nearest arcs. - Longitude range -100.418538 to 137.237029; latitude -34.640291 to 61.267258.

### Interpretation

No trip has equal route-row and GPS-row counts. Pointwise reference labels are not supplied. Connectivity breaks and direction ties require explicit handling in later alignment and decoding. Networks span diverse regions; a single fixed UTM zone would be unsuitable for the entire dataset. No duplicate track SHA-256 hashes or overlapping trip IDs were found between train and student testing; this does not establish geographic independence. See anomaly_locations.csv for zero-based observation/route row locations. Projection checks compare every nontrivial arc's projected length with its WGS84 geodesic endpoint distance; they do not prove exact geodesic nearest-road ordering.

## 2.1 Methodology, feature rationale and label limitations

### Phase 1 methodology and limitations

### Evidence and scope

The five PDFs under `docs/official/` were read, including the assessment and plagiarism policy. Text extractions accompany them. Original ZIPs are preserved; `data/original/manifest.json` records their SHA-256 hashes and archive membership. The analysis independently reads original extracted files. No precomputed statistics or supplied scripts are trusted. A separately supplied Phase 1 package was not located in the searched Downloads/Documents locations: its inspection and comparison remain blocked until its path is provided.

The official documents mention a first evaluation before lunch, but do not give a separate exhaustive Phase 1 rubric. This checkpoint covers the dataset exploration requirements and a geometric baseline. It is not a claim of final submission completeness.

### Parsing and preprocessing

All four file types are whitespace-delimited, headerless. `.track` has longitude, latitude, timestamp in seconds; `.nodes` has longitude, latitude; `.arcs` has integer source and destination node row IDs; `.route` has one integer arc row ID. IDs are zero-based and local to a trip. Do not learn global arc IDs as classes.

Strict parsing checks every row's width, numeric finiteness, integer IDs and index bounds. Nonfinite values, blank rows, malformed records and out-of-bounds coordinates or IDs cause a recorded parsing error, not silent imputation. Valid input rows are preserved in original order. No timestamps are sorted, no GPS points are deleted, and no reference discontinuities are repaired. Missing/invalid counts of zero are valid only for fully parsed trips; failures are recorded separately and make the run fail.

Longitude-first is established by the official schema, coordinate ranges and plausible mapped locations; range tests alone cannot always distinguish swapped coordinates. Pyproj uses `always_xy=True`; Folium gets latitude first. Observed timestamps are treated as seconds relative to an unspecified origin; no calendar date or UTC epoch is inferred.

WGS84 geodesics measure consecutive GPS displacements and headings. Per-trip azimuthal equidistant projection centred on the mean GPS position provides metres for segment geometry. This is a local approximation away from the projection centre, not a globally distortion-free metric. Quantified projection distance checks are retained in `experiments/projection_validation.json`. Arcs are straight segments between supplied nodes; there is no additional road curvature, speed limit, road class or traffic metadata in this schema.

Connectivity uses directed sparse graphs: weak and strong components, isolated nodes, self-loops and duplicate arcs. Reference continuity means the destination node of one route arc equals the source of the next. A break does not prove the vehicle teleported: it indicates limitations in the supplied representation.

Thresholds (>5 seconds, >55 m/s, >100 metres) are transparent exploratory flags, not tuned thresholds or reasons to discard records. Dataset-quality counts include duplicate coordinates, repeated positions, timestamp ordering, zero length segments and reference-route repetitions.

### Derived observation features

`data/processed/{train,student_test}/*_observations.csv` contains timestamp, delta time, geodesic step length, inferred speed and heading. First-observation differences are undefined and deliberately blank; speed is undefined when delta time is nonpositive. Heading at zero displacement must later be masked. These features describe sampling, motion and direction, not supervised labels. No learned scaling or imputation was fitted. Candidate-relative distance, heading agreement, endpoint position, network path distance and gap flags are planned features; their usefulness still requires training-only validation.

### Reference route and supervised-label investigation

`.route` is the chronological sequence of traversed arc IDs. Its length generally differs from GPS length. Arc repetition is meaningful; a set discards order and revisits. The files contain no explicit timestamp-to-reference-arc alignment. Consequently, row-wise zipping, resizing, or declaring each nearest road the true label is unsupported.

A possible future weak-label construction:

1. Preserve each reference occurrence as `(route_position, arc_id)`. 2. Project GPS observations onto the reference segments, retaining competing occurrences and residual distances. 3. Align monotonically in route position using a dynamic program that permits dwelling and skipped segments, with geodesic displacement and elapsed time constraints. Split or abstain around reference discontinuities and long gaps. 4. Retain uncertainty, runner-up margins and abstention flags. Audit sampled alignments visually; measure sensitivity to alignment assumptions. 5. Ask organisers whether an official alignment/evaluator is available.

These would be **pseudo-labels**, not independently verified pointwise ground truth. Never label every route-member candidate positive at every observation. Never align evaluation truth inside inference. The label-construction policy must be fixed before reporting comparable observation-level metrics.

### Geometric baseline

`nearest_road` takes only GPS coordinates, road nodes and directed arcs. Shapely STRtree finds exact nearest point-to-straight-segment distances in the local projection. Equidistant arcs are retained for tie counting; smallest arc ID deterministically breaks ties. Reverse-direction arcs often share geometry, so this baseline cannot reliably infer travel direction.

All observations in both permitted partitions are processed. Results are per-observation CSVs. This is not a trained model, candidate classifier or HMM. No accuracy, precision, recall, F1 or log loss is reported because aligned pointwise reference labels are unavailable.

Reported diagnostics:

- Nearest distance: GPS to the selected road segment, not route accuracy. - Direct adjacency: consecutive predictions are identical or the first arc's destination equals the next arc's source. Skipping intermediate arcs can fail this test despite reachable roads; passing it does not prove forward motion within an arc. Macro averages weight trips equally; micro weights transitions. - Reference membership: fraction of predicted arc IDs appearing anywhere in the trip's route; ignores temporal order and is **not pointwise accuracy**. - Baseline time includes geometry construction, indexing, queries and snapping; excludes file loading, projection and writing. CPU wall clock, single run.

### Visual evidence

Select training trips by near-median observation count, maximum timestamp gap, and most reference adjacency breaks (deduplicate selected trips). Static plots and Folium layers show GPS, extent-clipped road geometry, supplied reference arcs and the union of predicted nearest arcs. An unordered arc union is not a continuous reconstructed route. Reference arcs are drawn individually to avoid inventing connectors across breaks. HTML maps use external Leaflet assets and optional OpenStreetMap tiles. Generated screenshots are actual browser renders.

### Subsequent experiment design (not yet executed)

- Candidate generation: train-validated radius/top-k directed segments, stable `(trip, observation, arc)` correspondence, null-state policy, candidate recall. - Train five distinct families: logistic regression, k-nearest neighbours, random forest, histogram gradient boosting and a small MLP are provisional. Final configurations require experiments, not copied or assumed results. - GroupKFold by trip on training data only, with fold-local preprocessing and calibration; inspect geographic overlap and consider region-group sensitivity. Student testing remains development evaluation, not an unbiased final holdout. - Record seeds, folds, configurations, candidate coverage, Macro F1, Top-1, log loss where meaningful, confusion diagnostics, timing and route metrics. Decide primary selection metric after establishing defensible labels. - Bayesian optimisation on training group-CV; persist trials, search rationale, objective and costs; compare untuned and tuned on the same student-test trips. - HMM hidden states are candidate arc occurrences; model probabilities become emissions. Directed shortest paths and observed motion inform transitions. Log-space Viterbi handles invalid transitions, missing candidates and gaps. Tune decoder only on permitted development data and compare classifier-only. - Artifacts include preprocessing, model, feature schema, candidate settings, decoder configuration, package versions and provenance. - FastAPI loads artifacts once, excludes references from inference, computes metrics separately server-side, and logs request IDs/timings. See API contract.

### Integrity and assistance

This checkpoint was implemented with AI coding assistance and executed locally. Team members must review and understand the code and measured findings before submission. No trained artifacts, code or results from another team were used. No publication or push was performed. Evaluator data is absent and was not used.

## 3. Geometric baseline (not model training)

Diagnostic | Training | Student testing

Nearest distance median (m) | 4.9710 | 4.7600

Nearest distance 95th percentile (m) | 83.4535 | 16.0029

Direct adjacency, micro fraction | 0.8300 | 0.8254

Reference membership, NOT accuracy | 0.5415 | 0.6010

Baseline computation total (seconds) | 2.3331 | 0.2260

No observation-level Top-1, Macro F1, precision, recall or log loss is claimed: the supplied route sequence has no explicit timestamp alignment. Reference membership ignores route order; direct adjacency is not full route reachability.

### 3.1 Five-model experimentation and comparison - pending

No final model was trained, per Phase 1 scope. Later work must record every model's rationale, configuration, objective, fold and student-test results, overfitting evidence, timings and genuine comparison plots.

## 4. Bayesian hyperparameter optimisation - pending

No trials performed. Use training-only group-aware cross-validation; retain search space rationale, objective, trials, best parameters, cost and tuned/untuned comparison.

## 5. HMM and Viterbi route reconstruction - pending

No HMM fitted or decoded. Candidate-state mapping, emission calibration, directed transitions, gap handling and classifier-only comparisons are planned.

## 6. Geographical visualisations

### Training trip 00000083: Near-median GPS count

Actual Chromium render of the generated Folium vector map. Blue GPS, green reference arcs, orange nearest-road baseline, grey network. Offline view has no raster basemap; full HTML provides optional OpenStreetMap tiles.

![Actual Chromium render of the generated Folium vector map. Blue GPS, green reference arcs, orange nearest-road baseline, grey network. Offline view has no raster basemap; full HTML provides optional OpenStreetMap tiles.](../visualizations/train_00000083_folium.png)

Displayed road arcs: 16,742 of 25,393; clipped to GPS/reference extent. Interactive HTML: visualizations/train_00000083.html

Companion static longitude/latitude plot of the same supplied geometry.

![Companion static longitude/latitude plot of the same supplied geometry.](../visualizations/train_00000083.png)

### Training trip 00000090: Largest training timestamp gap

Actual Chromium render of the generated Folium vector map. Blue GPS, green reference arcs, orange nearest-road baseline, grey network. Offline view has no raster basemap; full HTML provides optional OpenStreetMap tiles.

![Actual Chromium render of the generated Folium vector map. Blue GPS, green reference arcs, orange nearest-road baseline, grey network. Offline view has no raster basemap; full HTML provides optional OpenStreetMap tiles.](../visualizations/train_00000090_folium.png)

Displayed road arcs: 3,884 of 7,942; clipped to GPS/reference extent. Interactive HTML: visualizations/train_00000090.html

Companion static longitude/latitude plot of the same supplied geometry.

![Companion static longitude/latitude plot of the same supplied geometry.](../visualizations/train_00000090.png)

### Training trip 00000019: Most reference adjacency breaks

Actual Chromium render of the generated Folium vector map. Blue GPS, green reference arcs, orange nearest-road baseline, grey network. Offline view has no raster basemap; full HTML provides optional OpenStreetMap tiles.

![Actual Chromium render of the generated Folium vector map. Blue GPS, green reference arcs, orange nearest-road baseline, grey network. Offline view has no raster basemap; full HTML provides optional OpenStreetMap tiles.](../visualizations/train_00000019_folium.png)

Displayed road arcs: 91,932 of 108,517; clipped to GPS/reference extent. Interactive HTML: visualizations/train_00000019.html

Companion static longitude/latitude plot of the same supplied geometry.

![Companion static longitude/latitude plot of the same supplied geometry.](../visualizations/train_00000019.png)

## 7. API, UI and deployment - contract only

docs/api_contract.md defines the two-laptop interface, schemas, failure cases, server-side evaluation isolation and integration acceptance criteria. No working API/UI screenshots are presented because these components are not implemented.

## 8. Verification and incomplete tasks

test_directed_adjacency_is_not_undirected (test_phase1.BaselineTests.test_directed_adjacency_is_not_undirected) ... ok
test_matches_exhaustive_distance_oracle (test_phase1.BaselineTests.test_matches_exhaustive_distance_oracle) ... ok
test_segment_interior_and_deterministic_direction_tie (test_phase1.BaselineTests.test_segment_interior_and_deterministic_direction_tie) ... ok
test_arc_and_route_id_bounds (test_phase1.ParsingTests.test_arc_and_route_id_bounds) ... ok
test_projection_order_and_metres (test_phase1.ParsingTests.test_projection_order_and_metres) ... ok
test_rejects_ragged_nonfinite_and_fractional_ids (test_phase1.ParsingTests.test_rejects_ragged_nonfinite_and_fractional_ids) ... ok
test_whitespace_and_single_row (test_phase1.ParsingTests.test_whitespace_and_single_row) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.027s

OK


Integration checks: 360 archive member hashes; 90 output row-count checks; 90 baseline arc-bound checks.

Maximum relative projected/geodesic arc-length discrepancy across checked nontrivial arcs: 0.00080556%. This measures endpoint-distance distortion, not nearest-road geodesic equivalence.

Blockers: supplied Phase 1 package not located; observation-to-reference alignment not supplied; evaluator dataset absent. Pending by design: five models, group-CV, Bayesian search, HMM, trained artifacts, API/UI and cross-laptop demonstration. Team review/viva preparation and eventual authorised GitHub publication remain outstanding.

### 8.1 Requirements coverage

Official requirement | Stage/status | Evidence or next step

Main Guidelines 1-4: Understand challenge and both laptop responsibilities | Phase 1: complete | docs/methodology.md and docs/api_contract.md

Dataset Exploration 3-4: Verify schema IDs availability sizes and counts | Phase 1: complete | src/pathsnap/data.py and experiments/file_inventory.csv

Dataset Exploration 4: Inspect all permitted trips and GPS/node/arc/route counts | Phase 1: complete | experiments/trip_statistics.csv

Dataset Exploration 4: Missing invalid duplicate coordinates and timestamp anomalies | Phase 1: complete | experiments/trip_statistics.csv and parsing_errors.json

Dataset Exploration 4-5: Coordinate order extents geodesic distances and projection choice | Phase 1: complete | docs/methodology.md and experiments/projection_validation.json

Dataset Exploration 4: Directed connectivity isolated nodes and reference continuity | Phase 1: complete | experiments/trip_statistics.csv

Dataset Exploration 6: Investigate reference format and uncertainty | Phase 1: complete | docs/methodology.md; aligned labels still unresolved

Dataset Exploration 7: Separate train student-test and evaluator data | All: complete | explicit split allowlist; no evaluator data present

Dataset Exploration 8: Inventory statistics plots and Folium maps | Phase 1: complete | experiments/ and visualizations/

Deliverables 2.2: Preprocessing patterns missing-value analysis and feature rationale | Phase 1: complete | docs/phase1_report.pdf and docs/methodology.md

Deliverables 1: Organized reproducible repository and dependencies | Phase 1: complete | README.md requirements.txt requirements.lock.txt

Deliverables 1: Public GitHub complete source artifacts and evaluator access | Final: pending | Requires user publication instruction

Deliverables 1-2: One consolidated PDF with sections 2-7 and actual evidence | Final: partial | Phase 1 PDF only; final sections pending experiments

Models and Training Guidelines 2-6: Train and compare at least five models across families | Later: pending | Do not train final models in Phase 1

Models and Training Guidelines 4: Record configurations seeds folds timing and experiment logs | Later: planned | docs/methodology.md

Models and Training Guidelines 5-6: Report applicable classification candidate Top-1 coverage and sequence metrics | Later: blocked | Defensible observation alignment needed

Deliverables 3.1: Consolidated five-model comparison CV and student-test table | Later: pending | Requires genuine training results

Deliverables 3.2: Each model suitability limitations hyperparameters objective performance and fit diagnosis | Later: pending | Requires genuine training results

Deliverables 3.3: CV Macro F1 student-test Top-1/F1 log-loss comparison plots | Later: pending | Only actual results; confusion matrices optional

Deliverables 3.3: Epoch histories only for models that produce them | Later: pending | No fabricated curves

Models and Training Guidelines 7: Bayesian search rationale objective training group-CV and trial history | Later: pending | docs/methodology.md design

Deliverables 4: Best parameters refit cost and tuned-versus-untuned comparison | Later: pending | Same metrics and partitions required

Models and Training Guidelines 8-9: States candidate mapping emissions transitions Viterbi gaps and invalid transitions | Later: pending | docs/methodology.md design

Deliverables 5: Classifier-only versus HMM pointwise sequence continuity comparison | Later: pending | Geometric baseline is not a classifier

Deliverables 6: Distinguishable GPS road reference and prediction map layers legend extent controls | Phase 1: complete | visualizations/train_*.html; baseline explicitly labelled

Deliverables 6: Representative geographical map screenshot in PDF and reproducible map code | Phase 1: complete | docs/phase1_report.pdf and src/pathsnap/render_maps.py

Models and Training Guidelines 5: Final trained model preprocessing decoder and inference assets | Later: pending | artifacts/ reserved

Main Guidelines 3: Two-laptop FastAPI/UI integration | Later: planned | docs/api_contract.md

Models and Training Guidelines 6 Final Evaluation: UI lists all ten evaluator trips and sends selected ID through API | Final: pending | Evaluator partition not supplied; frontend on second laptop

Models and Training Guidelines 6 Final Evaluation: Server-side scoring no reference leakage no hardcoded predictions API logs | Final: planned | docs/api_contract.md

Deliverables 7: Working UI and receiving-laptop API screenshots matching submitted code | Final: pending | No working API/UI claimed

Assessment and Plagiarism Policy 3-7: Independent experiments reproducible metrics no sharing or leakage | All: complete | Local execution provenance; ongoing obligation

Assessment and Plagiarism Policy 5-6: Disclose AI assistance and prepare technical viva | All: partial | Disclosure in methodology; team review and viva preparation required

User request Phase 1 package: Inspect supplied package and independently verify findings | Phase 1: blocked | Package not located; provide exact path

## Appendix: all-trip measurements

Split / trip | GPS | Nodes | Arcs | Max gap s | Route breaks

train / 00000001 | 1315 | 88809 | 165790 | 16 | 0

train / 00000002 | 7243 | 7936 | 16139 | 160 | 1

train / 00000003 | 821 | 9138 | 18028 | 2 | 0

train / 00000004 | 1064 | 7350 | 15499 | 51 | 0

train / 00000005 | 1045 | 898 | 1579 | 1 | 0

train / 00000006 | 1450 | 3259 | 6757 | 301 | 0

train / 00000008 | 3093 | 4995 | 10421 | 2 | 0

train / 00000010 | 2711 | 8502 | 15715 | 1 | 0

train / 00000011 | 2828 | 2649 | 5393 | 909 | 0

train / 00000012 | 1608 | 16376 | 38599 | 2 | 0

train / 00000013 | 2580 | 155003 | 307392 | 8 | 0

train / 00000014 | 448 | 4067 | 8288 | 72 | 0

train / 00000016 | 1305 | 8205 | 15098 | 1 | 0

train / 00000018 | 379 | 5980 | 11545 | 3 | 0

train / 00000019 | 2354 | 52694 | 108517 | 1 | 3

train / 00000020 | 2353 | 2739 | 5958 | 1 | 0

train / 00000021 | 15783 | 732 | 1510 | 2161 | 0

train / 00000022 | 474 | 9607 | 18982 | 22 | 0

train / 00000024 | 801 | 25782 | 51609 | 458 | 0

train / 00000025 | 1179 | 38035 | 85639 | 2 | 0

train / 00000026 | 361 | 2580 | 5043 | 17 | 0

train / 00000027 | 694 | 2756 | 5588 | 2 | 0

train / 00000028 | 3565 | 60628 | 127216 | 196 | 0

train / 00000029 | 733 | 10825 | 20868 | 1 | 0

train / 00000031 | 4245 | 59059 | 125831 | 1 | 0

train / 00000032 | 2464 | 124540 | 259437 | 41 | 0

train / 00000033 | 5844 | 3340 | 6360 | 3 | 1

train / 00000034 | 3614 | 19597 | 40971 | 15 | 0

train / 00000035 | 1643 | 21126 | 35212 | 1 | 0

train / 00000036 | 1368 | 4528 | 9619 | 1 | 0

train / 00000037 | 677 | 9472 | 21744 | 52 | 0

train / 00000038 | 1287 | 21046 | 42026 | 20 | 0

train / 00000039 | 363 | 1185 | 2388 | 10 | 0

train / 00000041 | 1019 | 7837 | 14317 | 89 | 1

train / 00000042 | 7739 | 49376 | 101128 | 670 | 1

train / 00000043 | 2322 | 109070 | 215620 | 230 | 0

train / 00000044 | 546 | 4386 | 9209 | 341 | 0

train / 00000045 | 640 | 2034 | 4335 | 1 | 0

train / 00000046 | 1401 | 299 | 649 | 22 | 0

train / 00000047 | 1272 | 18388 | 39747 | 1 | 0

train / 00000048 | 1240 | 1308 | 2592 | 4 | 0

train / 00000050 | 518 | 2206 | 4576 | 1 | 0

train / 00000051 | 984 | 4181 | 8494 | 2 | 0

train / 00000053 | 1304 | 77905 | 148707 | 10 | 0

train / 00000054 | 19313 | 63822 | 116916 | 79 | 0

train / 00000055 | 2472 | 98511 | 198961 | 184 | 0

train / 00000057 | 1645 | 8929 | 18777 | 13 | 0

train / 00000058 | 599 | 22367 | 39383 | 16 | 0

train / 00000060 | 854 | 15128 | 30077 | 27 | 0

train / 00000061 | 1430 | 4947 | 10339 | 3 | 0

train / 00000062 | 3378 | 30613 | 62836 | 56 | 0

train / 00000063 | 2781 | 2617 | 5553 | 2 | 0

train / 00000064 | 2093 | 28258 | 56986 | 4 | 0

train / 00000066 | 4325 | 9609 | 20071 | 1 | 0

train / 00000068 | 747 | 3537 | 7058 | 12 | 0

train / 00000069 | 2067 | 8556 | 17193 | 2658 | 1

train / 00000070 | 2076 | 1799 | 3776 | 152 | 0

train / 00000071 | 850 | 6265 | 14868 | 6 | 0

train / 00000072 | 969 | 3275 | 7063 | 1 | 0

train / 00000074 | 1929 | 34654 | 61055 | 24 | 0

train / 00000075 | 1628 | 1022 | 2124 | 2 | 2

train / 00000078 | 2909 | 75230 | 151458 | 52810 | 0

train / 00000079 | 689 | 39031 | 72116 | 1 | 0

train / 00000080 | 2510 | 10014 | 18746 | 59 | 1

train / 00000081 | 3224 | 2791 | 5048 | 536 | 0

train / 00000082 | 4268 | 15696 | 31754 | 1 | 0

train / 00000083 | 1507 | 12907 | 25393 | 1 | 0

train / 00000084 | 16031 | 24898 | 50888 | 312 | 0

train / 00000085 | 1092 | 690 | 1554 | 70 | 0

train / 00000086 | 4143 | 114868 | 237656 | 1 | 1

train / 00000088 | 876 | 11373 | 24376 | 232 | 0

train / 00000089 | 1228 | 15972 | 32823 | 193 | 1

train / 00000090 | 536 | 4010 | 7942 | 153064 | 0

train / 00000092 | 7892 | 26468 | 53348 | 1 | 0

train / 00000093 | 1211 | 45161 | 85122 | 6 | 0

train / 00000095 | 1551 | 1888 | 3723 | 2 | 0

train / 00000096 | 3271 | 16893 | 32276 | 1 | 0

train / 00000097 | 1740 | 23566 | 48503 | 2 | 0

train / 00000098 | 5584 | 43440 | 88137 | 50 | 0

train / 00000099 | 2322 | 34252 | 72870 | 25 | 0

student_test / 00000000 | 2503 | 8542 | 18196 | 2 | 0

student_test / 00000007 | 1354 | 2061 | 4369 | 1 | 0

student_test / 00000009 | 1049 | 6745 | 13106 | 2 | 0

student_test / 00000040 | 2995 | 12945 | 27026 | 1 | 0

student_test / 00000052 | 262 | 1770 | 3577 | 2 | 0

student_test / 00000056 | 3958 | 14759 | 27631 | 3 | 0

student_test / 00000065 | 768 | 6448 | 13184 | 210 | 0

student_test / 00000076 | 1486 | 26635 | 56027 | 32 | 1

student_test / 00000077 | 3911 | 4424 | 9622 | 1 | 0

student_test / 00000091 | 874 | 6360 | 12303 | 1 | 0
