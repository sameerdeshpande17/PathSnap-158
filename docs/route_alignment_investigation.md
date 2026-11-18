# Reference-route alignment investigation

## Official meaning and missing alignment rules
Dataset Exploration.pdf section 3 defines `.route` as a chronological sequence
of arc IDs traversed by the vehicle; arc IDs and node IDs are zero-based row
indices. Sections 3 and 6 explicitly warn against direct row correspondence
and treating supplied reference information as error-free physical truth.
Models and Training Guidelines.pdf section 5 asks for corresponding reference
segments for Top-1 scoring, but provides no timestamp-to-occurrence alignment
algorithm. Searches of all five official text extractions found no explicit
alignment rule or per-observation label file. Approximate alignment is therefore
an independently implemented weak-label policy, not recovered official labels.

## Ordered directed traversal and repeated occurrences
Across 90 trips, no route-row count equals GPS-row count. There are 34,813 supplied arc occurrences.
10 trips contain repeated arc IDs. Repeated occurrences are consistent
with repeated traversal in the documented format; geometry alone cannot prove
that each repeat reflects actual vehicle motion rather than a reference error.
Do not deduplicate the route. A nearest-segment tie cannot locate the visit in time.
- train 00000008: arc 5191 occurs at zero-based positions [143, 172].
- train 00000018: arc 682 occurs at zero-based positions [71, 93].
- train 00000069: arc 4593 occurs at zero-based positions [421, 438, 455].
- train 00000075: arc 632 occurs at zero-based positions [3, 89].
- train 00000084: arc 9092 occurs at zero-based positions [77, 85].
- train 00000086: arc 130144 occurs at zero-based positions [0, 3].
- train 00000090: arc 3177 occurs at zero-based positions [180, 196].
- train 00000092: arc 1251 occurs at zero-based positions [1043, 1111].
- train 00000095: arc 316 occurs at zero-based positions [98, 148].
- student_test 00000056: arc 18422 occurs at zero-based positions [23, 506, 731].

## All 14 directed adjacency breaks
13 boundaries have a directed connector in the supplied road graph; that connector is not represented between these
consecutive route occurrences. This is consistent with skipped connector arcs,
partial route encoding or reference errors, but does not identify the true driven path.
All 14 endpoint pairs are in the same weak component. None is explained merely
by separate weak components or coincident node coordinates.
Trip 00000086 has no forward directed connector at occurrences 1->2, but does
have a reverse connector: a direction/connectivity inconsistency in the supplied
reference/network pair. The data alone cannot establish which source is wrong.
Shortest paths are evidence of graph possibilities, not silent route repairs.
Endpoint-to-nearest-GPS indices/times in the CSV are spatial diagnostics,
not adjudicated temporal alignment (especially on revisited roads).

| Split/trip | Occurrences | Arcs | End/start nodes | Separation m | Directed path m |
|---|---|---|---|---:|---:|
| train/00000002 | 53->54 | 1848->10863 | 5059->5289 | 324.662 | 1057.067 |
| train/00000019 | 94->95 | 85589->51027 | 849->23488 | 241.681 | 364.429 |
| train/00000019 | 151->152 | 55059->55050 | 25413->25408 | 255.509 | 512.961 |
| train/00000019 | 300->301 | 953->23709 | 10107->10862 | 597.723 | 600.430 |
| train/00000033 | 164->165 | 1003->1002 | 1283->563 | 91.965 | 91.965 |
| train/00000041 | 12->13 | 1264->519 | 349->348 | 57.373 | 402.580 |
| train/00000042 | 1212->1213 | 14661->17989 | 1762->8726 | 30.666 | 30.667 |
| train/00000069 | 153->154 | 10165->10172 | 5083->5084 | 40.977 | 62.399 |
| train/00000075 | 31->32 | 507->509 | 250->227 | 10.643 | 925.245 |
| train/00000075 | 62->63 | 134->552 | 229->247 | 73.143 | 1211.509 |
| train/00000080 | 118->119 | 8899->9194 | 4222->4363 | 107.579 | 231.885 |
| train/00000086 | 1->2 | 230068->96025 | 4019->45354 | 97.890 | unreachable |
| train/00000089 | 192->193 | 22715->22703 | 10765->10779 | 29.371 | 180.754 |
| student_test/00000076 | 12->13 | 39548->1122 | 137->509 | 15.064 | 177.175 |

## Timestamp and geometry evidence
All observed timestamps strictly increase. Coordinates and trajectory order allow
distance-based matching and nondecreasing reference-occurrence alignment, with
multiple observations per arc and skipped arcs. They do not uniquely identify
road direction at overlapping geometries or revisit number at repeated arcs.
Trip 00000090 contains a 153064-second gap; the vehicle's route during that interval
cannot be verified from these observations. Speed bounds and progression penalties
are assumptions and are relaxed with elapsed time, not treated as new observations.

## Executed methods, screening and conclusions
All three methods were run on all 80 training and 10 student-testing trips.
See docs/alignment_methodology.md for equations and limitations; frozen settings
and training-only calibration provenance are under experiments/alignment/.
Only 100% tentative assignment coverage is guaranteed by the algorithms; eligibility
and conservative screened coverage must be examined separately.
- student_test A: residual median 5.165m, p95 16.788m; quality eligible 92.69%; order violations 770; unexplained jumps 414.
- student_test B: residual median 5.277m, p95 17.004m; quality eligible 98.67%; order violations 0; unexplained jumps 0.
- student_test C: residual median 5.463m, p95 17.105m; quality eligible 98.70%; order violations 0; unexplained jumps 0.
- train A: residual median 5.614m, p95 140.288m; quality eligible 83.30%; order violations 5601; unexplained jumps 2285.
- train B: residual median 5.698m, p95 142.095m; quality eligible 86.55%; order violations 0; unexplained jumps 101.
- train C: residual median 6.482m, p95 7914.863m; quality eligible 79.21%; order violations 0; unexplained jumps 38.
- B: maximum benchmark arc disagreement under sigma x0.5/x2 was 1.49%.
- C: maximum benchmark arc disagreement under sigma x0.5/x2 was 13.21%.

## Important observed failure of method C
The probabilistic method is not robust enough for unscreened labels.
On the following training trips it locks onto an early occurrence
and remains there as the GPS trajectory progresses. Hard progression
constraints combined with beam/candidate truncation and no unmatched
state can prevent recovery. These are measured alignment failures,
not evidence of incorrect GPS labels or verified correctness of B.
- train 00000032: C terminal occurrence 186, B terminal 818, reference size 819; C p95 70118.98m versus B 14.99m. 1869 observations have C distance >1km while B distance <100m; first at observation 595. Selected for supervised use among these: 0. During the constant suffix, the next occurrence was speed-admissible in 39 steps, but in the nearest candidate set only 28 steps.
- train 00000084: C terminal occurrence 80, B terminal 626, reference size 627; C p95 20594.32m versus B 36.22m. 12982 observations have C distance >1km while B distance <100m; first at observation 3049. Selected for supervised use among these: 0. During the constant suffix, the next occurrence was speed-admissible in 12959 steps, but in the nearest candidate set only 0 steps.
The screened-consensus rule excludes these severe residuals. No algorithm or
parameter was changed after student testing to conceal the failures. Future
training-only work could test explicit unmatched states, gap-aware re-entry,
soft outlier transition mixtures, and broader lookahead. This version does
not claim those remedies were implemented or validated.

Approach B is the better-supported default chronology heuristic for continued
label investigation: it respects occurrence order, avoids C's observed lock-in
failures, and was less sensitive to the tested spatial-scale perturbations.
This is a robustness judgement,
not a pointwise-accuracy ranking. A provides the spatial-residual lower bound;
C adds a constrained probabilistic alternative but does not establish calibrated
confidence or independent correctness.
For initial supervised experiments, use only the screened B/C occurrence-consensus
subset (stored as recommended_supervision_use in C exports), retaining all
uncertain rows for audit. B and C targets are identical on that subset.
These labels are suitable only for explicitly weak-supervised exploratory training.
They are not adequate to claim true pointwise accuracy or F1 on future classifiers.
No model was trained. Independent point labels or human adjudication remain necessary
to validate the target itself. The separate Phase 1 package remains unlocated.
