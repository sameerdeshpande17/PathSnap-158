# Phase 2B model evaluation methodology

## Status of the targets

All labels are the frozen Phase 2A weak/inferred supervision (screened
B/C-consensus targets, optionally supplemented with Approach B-only labels).
They are **not independently verified ground truth**. Every metric below
measures agreement with that weak-label policy. True pointwise accuracy,
F1, or log loss against physical ground truth cannot be reported from these
data.

## Experimental design fixed before student testing

1. **Candidate generation** runs on all 80 training trips and all 10
   student-testing trips with one frozen configuration:
   radius 50 m, max 12 candidates per observation, nearest-arc fallback,
   5 s gap threshold. Complete road network is searched; reference routes are
   not used.
2. **Weak-label join** happens after inference-only feature construction.
3. **Supervision strategies** (kept separate):
   - `consensus`: screened B/C-consensus observations only.
   - `consensus_plus_B`: same consensus observations plus up to 5 000 B-only
     quality-eligible, geometrically unambiguous observations.
4. **Cross-validation**: GroupKFold, 3 splits, trip IDs as groups; shuffle
   with seed 42. No observations from the same trip appear in both fitting
   and validation parts of a fold. Validation trips are never used for fitting.
5. **Preprocessing**: StandardScaler fitted on the fold's training candidate
   rows only; serialized with each model artifact.
6. **Candidate misses**: observations whose inferred positive arc is absent
   from the candidate set are excluded from fitting rows but count as
   Top-1 disagreement and reciprocal-rank zero in evaluation.
7. **Negative candidates**: all generated candidates for sampled covered
   observations; no class weights; no label-dependent candidate injection.
8. **Selection rule** (declared before student-test results): primary metric is
   CV mean observation-level Top-1 weak-label agreement (trip-macro over
   folds); tie-breakers are CV candidate Macro F1, CV MRR, then lower
   prediction latency. Student-testing data plays no role in choosing the
   model, thresholds, or search spaces.
9. **Final model**: retrained on the full sampled training corpus under the
   winning strategy, then evaluated once on the 10 student-testing trips.

## Metrics

Candidate level (every generated candidate row is one record):

- `candidate_precision/recall/f1_class1` and class 0, plus Macro F1.
- `candidate_log_loss` (binary, labels [0,1]).
- `candidate_average_precision`.
- `candidate_accuracy` (not used for selection because negatives dominate).
- `weak-label candidate recall`: fraction of labeled observations whose
  inferred positive arc appears in the candidate set (~99.9% in train).

Observation level (one Top-1 candidate per observation, highest score):

- `obs_top1_agreement`: fraction of labeled observations whose Top-1
  candidate arc equals the weak label arc; candidate misses count as zero.
- `obs_top1_agreement_covered_only`: same, excluding candidate-miss
  observations.
- `obs_mrr`: mean reciprocal rank of the first weak-positive candidate
  (score descending, distance rank ascending); miss = 0.
- `obs_candidate_miss_rate`.

Route-continuity diagnostics (prediction coherence, not accuracy):

- `top1_direct_adjacency`: fraction of consecutive observation pairs whose
  Top-1 arcs are identical or endpoint-connected in the directed graph.
- `top1_same_arc`, `top1_endpoint_adjacent` components.

Costs and artifacts:

- `train_s`: wall-clock training time per fold/model.
- Inference latency per observation (prediction loop).
- Peak RSS delta during fit (MB).
- Saved artifact size (bytes).
- Device: CPU for scikit-learn models (none has a CUDA parameter); CUDA for
  the MLP when torch CUDA is available, else CPU.

## What is not reported

- Pointwise accuracy/F1 against verified ground truth (no such labels exist).
- Weak-label agreement described as true map-matching accuracy.
- Model selection influenced by student-testing results.
- Features derived from `.route` at inference.
