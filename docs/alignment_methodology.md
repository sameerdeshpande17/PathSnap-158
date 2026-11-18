# Phase 2A alignment methodology

## Intended use

These are reference-assisted label-construction algorithms, not map-matching
predictors. All three consume `.route`. None may be deployed as an inference
feature extractor. Their outputs are weak/inferred supervision, never independent
pointwise truth. No classifier training or final HMM predictor is performed here.

## Geometry and identity

Reuse the Phase 1 strict parser and longitude-first local WGS84 azimuthal
equidistant projection. Every state is a zero-based reference occurrence `j`,
mapping to `route[j]`, which maps to the directed endpoints `arcs[route[j]]`.
Repeated arc IDs remain distinct occurrences with identical segment geometry.
For observation i and occurrence j compute distance d(i,j) to the finite
segment, projected position along that directed segment, and progress s(i,j).
Progress is the sum of prior supplied arc lengths plus projected in-arc distance.
It deliberately omits unknown connectors at reference breaks: those boundaries
are flagged and never silently repaired.

## A: nearest reference segment

Minimize d(i,j) independently for each observation; ties select the earliest
occurrence. All reference occurrences are searched, including repetitions.
This supplies a lower bound on achievable spatial residuals, not necessarily the
correct direction, occurrence or chronological traversal.

## B: monotone dynamic program

Minimize total L1 spatial cost d/sigma plus transition cost:

`w * abs((s_current - s_previous) - GPS_step) / beta(dt)`

plus a soft speed-excess penalty `4 * excess / slack` and 8 per crossed reference
break. Here `beta(dt) = beta * (1 + sqrt(min(max(dt-1,0),300)))`.
Reject backwards occurrence transitions. Permit dwelling on an occurrence,
skipping intermediate occurrences, and small in-segment projection reversals
due to noise. At a supplied reference break, progression mismatch is unknown:
omit that term, penalize the crossing, and exclude the crossing observation
from quality-eligible labels. This does not assert that a connecting path exists.
Initial and terminal occurrence are unconstrained; an observation window need
not cover the complete supplied route.

## C: constrained probabilistic alignment

Use Gaussian-shaped distance emissions proportional to
`exp(-0.5 * (d/sigma)^2)` and transitions proportional to
`exp(-w * abs(delta_progress-GPS_step)/beta(dt) - 8*breaks_crossed)`.
Normalize each transition row over allowed current candidates with logsumexp.
Occurrence decreases are impossible. Occurrence changes exceeding
`max_speed * dt + slack` are forbidden unless crossing a supplied reference
break, where path length is unknown. Self-occurrence transitions remain allowed:
their changing projection is not an independently modelled vehicle position.
Excessive self-occurrence projection changes are still flagged in diagnostics.

Use log-space Viterbi for the most likely retained sequence, subtracting the
best accumulated cost at every step to avoid numerical growth. This is an
observation-conditioned chain: transitions use observed displacement and its
candidate support is observation-dependent. It is not a calibrated generative
GPS model, and is not the later classifier-emission HMM required by the hackathon.

## Practical search and approximation

Compute all N*M point-to-occurrence distances in 512-observation chunks, retaining
distance and progress matrices. For B/C, each step considers the nearest K
occurrences plus all previous beam states, evaluates all previous/current
transitions, and retains the best W current states and traceback parents.
This guarantees a monotone occurrence path through dwelling; it does not
guarantee that the globally optimal unrestricted path survives pruning.
Spatial costs and time-dependent terms are not learned classifier features.

With W and K >= M, the implementation is full-state dynamic programming, tested
against independent exhaustive enumeration on small problems. With smaller
beams, C also renormalizes transitions over a truncated support. Equal outputs
under two widths do not prove global optimality. Memory is O(N*M + N*W);
transition computation is approximately O(N*W*(W+K)), plus candidate sorting.

## Training-only parameter estimation and freeze

Calibration reads only the 80 training directories, guarded in code. It estimates:

- sigma = max(5m, median nearest-reference distance / 0.67448975);
- beta = max(10m, 75th percentile of absolute nearest-reference progress versus
  GPS-step mismatch), using dt<=5s and forward same/next occurrence without breaks;
- max_speed = clipped [15,70] m/s 99th percentile of GPS-derived speeds with dt<=5s.

These are robust scale proxies, not estimates of verified sensor error or true
vehicle maximum speed. Other settings are declared constants: progress weight 1,
break penalty 8, soft speed penalty 4, slack max(30m,3*sigma), residual limit
max(30m,3*sigma), ambiguity margin 2m, gap threshold 5s. Seed 0 is recorded,
but there is no random sampling.

Benchmark training trips selected by maximum N*M, largest gap, most repeated
occurrences and most reference breaks. Compare base W=32/K=24 with W=64/K=48 and
sigma multipliers 0.5 and 2. Predeclared width rule: widen if any benchmark trip's
arc assignments change by >1%; otherwise keep base width. Sigma sensitivity is
reported, not optimized against heuristic agreement. Freeze the method JSON
and hash before running student-testing alignments; verify the hash afterwards.
Student-test findings do not modify parameters or label-screening thresholds.

## Diagnostics and screening

- Assigned coverage: fraction given a tentative occurrence; this can be 100%
  even when an alignment is implausible.
- Quality-eligible fraction: residual <= threshold, no order violation, no
  unexplained jump, no large backwards progress, no reference-break crossing,
  not immediately before/after a >5s timestamp gap, and not a zero-length arc.
- Geometric ambiguity: multiple occurrences within 2m of the minimum distance.
  Includes shared endpoints, overlapping/reversed arcs and repeated traversals.
- Unexplained jump: route progress exceeds speed*dt+slack without a documented
  reference break. Break crossings are reported separately; they are not assumed
  valid. Backwards progress means a decrease exceeding slack metres.
- Route-order violation: assigned occurrence index decreases.
- Occurrence coverage: fraction of supplied occurrences assigned at least one
  observation. Low coverage can reflect sparse sampling or irrelevant supplied
  route portions; high coverage does not establish correctness.
- Sequence agreement diagnostic: longest nondecreasing subsequence of the
  run-length-collapsed assigned occurrence sequence divided by its length.
  B/C satisfy this by construction. It is not independent accuracy.
- Distances: median, p95, maximum and RMSE of projected spatial residuals;
  summary quantiles pool observations, while occurrence coverage averages trips.
- Timestamp sensitivity: separate residuals for dt<=1, (1,5], (5,30], >30s,
  and gap-specific order/jump counts. First observations have no measured dt
  and are excluded from interval strata. No synthetic time perturbation is claimed.
- Method differences compare both arc IDs and occurrence indices; agreement
  between methods sharing inputs and constraints is not independent validation.
- Runtime: separate projection/geometry, method alignment and end-to-end export.

Every output keeps tentative assignments, even rejected ones. The
`recommended_supervision_use` column is true only in method C files where B/C
agree on the occurrence, both are quality eligible, and geometric ambiguity is
absent. B and C therefore give exactly the same target on this subset. This is
a conservative screening rule, not a confidence guarantee.

`spatial_compatibility_score = exp(-0.5*(distance/sigma)^2)` is a distance-only
diagnostic shared by all methods, not a posterior or probability of correctness.
Quality flags and all excluded counts are exported, not silently discarded.

## Limits for subsequent supervised training

Use screened labels only for exploratory weak-supervision experiments. Preserve
trip groups and occurrence metadata; train candidate-relative classifiers, not
global trip-local arc IDs. Candidate generation and prediction features must
use GPS and road network only. Keep label construction in a separate offline job.
During rigorous group-CV, refit alignment calibration within training folds (or
explicitly disclose a fixed target-generation policy); fit learned feature
preprocessing only within training folds.

Evaluation against these targets measures agreement with the frozen weak-label
policy, not true pointwise accuracy/F1. Candidate misses, uncertain targets,
route residuals/continuity and abstention coverage must remain visible.
Independent organizer-supplied point labels or manually adjudicated observations
would be needed to establish actual correctness and compare label accuracy.
Neither A/B/C consensus nor the supplied route alone provides that evidence.
Small residuals can still attach an observation to the wrong nearby, reversed
or repeated road occurrence, especially around missing reference connectors.
Screening also creates selection bias: easy, unambiguous observations are
overrepresented. The final report records C's severe training lock-in failures;
its unscreened outputs are unsuitable training targets.
